/*
 *    iPFIClassifier.java
 *    Copyright (C) 2026 University of Waikato, Hamilton, New Zealand
 *
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program. If not, see <http://www.gnu.org/licenses/>.
 *
 */
package moa.learners.featureanalysis;

import com.github.javacliparser.FlagOption;
import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.github.javacliparser.MultiChoiceOption;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import moa.capabilities.CapabilitiesHandler;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.classifiers.MultiClassClassifier;
import moa.classifiers.meta.AdaptiveRandomForest;
import moa.core.Measurement;
import moa.core.Utils;
import moa.options.ClassOption;

/**
 * Incremental Permutation Feature Importance (iPFI).
 *
 * <p>A meta classifier that trains any base classifier and, at the same time, incrementally
 * estimates the permutation feature importance of every input feature. For each incoming
 * instance the loss of the base learner is measured on the original instance and on
 * {@code nInteractions} perturbed copies, where the value of a single feature is replaced by
 * the value observed for that feature in a randomly drawn instance from a sliding window of
 * recent instances (sampling-based imputation). The difference between the perturbed loss and
 * the original loss is the importance contribution of that feature, and it is aggregated over
 * time with an exponentially weighted moving average controlled by {@code smoothingAlpha}.</p>
 *
 * <p>Importances can be tracked globally (one score per feature) or class-wise (one score per
 * feature per class), which is useful under class imbalance. Predictions are simply delegated
 * to the base learner, so wrapping a classifier with this class does not change its
 * predictive behaviour, only its cost.</p>
 *
 * <p>See details in:<br> Fabian Fumagalli, Maximilian Muschalik, Eyke Hüllermeier,
 * Barbara Hammer. Incremental Permutation Feature Importance (iPFI): Towards Online
 * Explanations on Data Streams. Machine Learning, 2023.</p>
 *
 * <p>Parameters:</p> <ul>
 * <li>-l : Classifier to train and to be analyzed.</li>
 * <li>-f : Loss function used to score the (perturbed) predictions: cross entropy, focal loss,
 * 0/1 accuracy, Brier score, hinge or class-balanced 0/1 accuracy.</li>
 * <li>-g : Focusing parameter of the focal loss.</li>
 * <li>-s : Smoothing factor of the exponentially weighted moving average.</li>
 * <li>-n : Number of perturbations sampled per feature per instance.</li>
 * <li>-w : Size of the sliding window the replacement values are drawn from.</li>
 * <li>-c : If set, a single importance value is tracked per feature instead of one per
 * feature per class.</li>
 * </ul>
 *
 * @author Daniel Nowak Assis (daniel dot nowak-assis at lip6 dot fr)
 */
public class iPFIClassifier extends AbstractClassifier implements MultiClassClassifier,
        CapabilitiesHandler, FeatureImportanceLearner {

    private static final long serialVersionUID = 1L;

    protected static final int LOSS_CROSS_ENTROPY = 0;
    protected static final int LOSS_FOCAL = 1;
    protected static final int LOSS_ACCURACY = 2;
    protected static final int LOSS_BRIER = 3;
    protected static final int LOSS_HINGE = 4;
    protected static final int LOSS_BALANCED_ACCURACY = 5;

    /**
     * Probabilities are clipped to this floor before being fed to a logarithm, so that a
     * confidently wrong prediction yields a large finite loss instead of an infinity.
     */
    protected static final double PROBABILITY_FLOOR = 1e-12;

    @Override
    public String getPurposeString() {
        return "Trains a base classifier and incrementally estimates the permutation "
                + "importance of each feature (iPFI).";
    }

    public ClassOption baseLearnerOption = new ClassOption("baseLearner", 'l',
            "Classifier to train.", Classifier.class, "trees.HoeffdingTree");

    public MultiChoiceOption lossFunctionOption = new MultiChoiceOption("lossFunction", 'f',
            "Which loss to use. Cross entropy and focal loss react to shifts in confidence, "
                    + "Brier score does the same but stays bounded, hinge reacts to the margin "
                    + "between the true class and its closest competitor, and the two accuracy "
                    + "losses only react to perturbations that flip the predicted label.",
            // The fourth argument holds the labels accepted on the command line and the fifth
            // their descriptions, so the short codes belong in the first of the two.
            new String[]{"cce", "fl", "acc", "brier", "hinge", "bacc"},
            new String[]{"Cross Entropy", "Focal Loss", "Accuracy", "Brier Score", "Hinge",
                    "Balanced Accuracy"}, LOSS_CROSS_ENTROPY);

    public FloatOption focalGammaOption = new FloatOption("focalGamma", 'g',
            "Focusing parameter of the focal loss. 0 makes it identical to cross entropy, "
                    + "larger values down-weight instances that are already classified well.",
            2.0, 0.0, 10.0);

    public FloatOption smoothingAlphaOption = new FloatOption("smoothingAlpha", 's',
            "Smoothing factor of the exponentially weighted moving average used to aggregate "
                    + "the per-instance importance contributions.", 0.001, 0.0, 1.0);

    public IntOption nInteractionsOption = new IntOption("nInteractions", 'n',
            "Number of perturbations sampled per feature per instance.", 1, 1, Integer.MAX_VALUE);

    public IntOption maxWindowSizeOption = new IntOption("maxWindowSize", 'w',
            "Size of the sliding window the replacement values are sampled from.",
            500, 1, Integer.MAX_VALUE);

    public FlagOption deactivateClasswiseImportanceOption = new FlagOption(
            "deactivateClasswiseImportance", 'c',
            "If set, track a single importance value per feature instead of one per feature per class.");

    /** Importance estimates, [class][feature] (a single row when class-wise tracking is off). */
    protected double[][] importanceTracker;

    /** Attribute index of each tracked feature, i.e. every attribute but the class attribute. */
    protected int[] featureIndices;

    /** Number of instances seen per class, used to weight the balanced accuracy loss. */
    protected double[] classCounts;

    /** Running total of {@link #classCounts}, kept alongside it to avoid re-summing per call. */
    protected double classCountTotal;

    /** How many distinct classes have been observed so far. */
    protected int observedClasses;

    /** Sliding window of recent instances, used as the source of the replacement values. */
    protected Instances window;

    protected Classifier classifier;

    protected long instancesSeen;

    @Override
    public void setModelContext(InstancesHeader context) {
        super.setModelContext(context);
        if (context != null) {
            this.window = new Instances(context, 0);
            this.window.setClassIndex(context.classIndex());
            initTracker(context.numClasses(), context.numAttributes(), context.classIndex());
        }
        if (this.classifier != null) {
            this.classifier.setModelContext(context);
        }
    }

    @Override
    public void resetLearningImpl() {
        this.classifier = ((Classifier) getPreparedClassOption(this.baseLearnerOption)).copy();

        // The importance estimation queries the base learner once per feature per perturbation,
        // so let ensembles that support it use every available core.
        if (this.classifier instanceof AdaptiveRandomForest) {
            ((AdaptiveRandomForest) this.classifier).numberOfJobsOption.setValue(-1);
        }

        if (this.getModelContext() != null) {
            this.classifier.setModelContext(this.getModelContext());
        }
        this.classifier.resetLearning();

        this.instancesSeen = 0;
        this.importanceTracker = null;
        this.featureIndices = null;
        this.classCounts = null;
        this.classCountTotal = 0.0;
        this.observedClasses = 0;
    }

    @Override
    public double[] getVotesForInstance(Instance inst) {
        return this.classifier.getVotesForInstance(inst);
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {
        if (this.window == null) {
            this.window = new Instances(inst.dataset(), 0);
            this.window.setClassIndex(inst.classIndex());
        }
        initTracker(inst.numClasses(), inst.numAttributes(), inst.classIndex());
        // Counted before the importances are updated, so the weight of the current class is
        // always defined. Only real instances are counted, never the perturbed copies.
        countClass((int) inst.classValue());

        // The first instance only seeds the window: there is nothing to sample replacements from yet.
        if (this.instancesSeen > 0) {
            updateImportances(inst);
        }

        this.instancesSeen++;
        this.window.add(inst);
        while (this.window.size() > this.maxWindowSizeOption.getValue()) {
            this.window.delete(0);
        }

        this.classifier.trainOnInstance(inst);
    }

    /**
     * Estimate the importance contribution of every feature for a single instance and fold it
     * into the exponentially weighted moving average.
     */
    protected void updateImportances(Instance inst) {
        int numClasses = inst.numClasses();
        double[] votes = toDistribution(this.classifier.getVotesForInstance(inst), numClasses);
        double loss = calculateLoss(inst, votes);

        double alpha = this.smoothingAlphaOption.getValue();
        int trackerRow = this.deactivateClasswiseImportanceOption.isSet() ? 0 : (int) inst.classValue();
        if (trackerRow < 0 || trackerRow >= this.importanceTracker.length) {
            return;
        }

        for (int i = 0; i < this.featureIndices.length; i++) {
            double avgLoss = 0.0;
            for (int j = 0; j < this.nInteractionsOption.getValue(); j++) {
                Instance imputed = inst.copy();
                double newValue = this.window.get(this.classifierRandom.nextInt(this.window.size()))
                        .value(this.featureIndices[i]);
                imputed.setValue(this.featureIndices[i], newValue);
                avgLoss += calculateLoss(inst,
                        toDistribution(this.classifier.getVotesForInstance(imputed), numClasses));
            }
            avgLoss /= this.nInteractionsOption.getValue();

            this.importanceTracker[trackerRow][i] =
                    (1 - alpha) * this.importanceTracker[trackerRow][i] + alpha * (avgLoss - loss);
        }
    }

    /**
     * Loss of a prediction with respect to the true class of {@code inst}. The returned value
     * grows as the prediction gets worse, so that the difference between the loss under
     * perturbation and the original loss is positive for informative features.
     *
     * <p>Note that only differences of this quantity are ever used, so a loss is defined up to
     * an additive constant. That is why the two accuracy losses score a correct prediction as
     * negative and a wrong one as zero rather than the other way round: the importances are
     * identical either way.</p>
     */
    public double calculateLoss(Instance inst, double[] votes) {
        int trueClass = (int) inst.classValue();
        if (trueClass < 0 || trueClass >= votes.length) {
            return 0.0;
        }
        switch (this.lossFunctionOption.getChosenIndex()) {
            case LOSS_CROSS_ENTROPY:
                return -Math.log(clipProbability(votes[trueClass]));

            case LOSS_FOCAL: {
                double p = clipProbability(votes[trueClass]);
                return -Math.pow(1 - p, this.focalGammaOption.getValue()) * Math.log(p);
            }
            case LOSS_ACCURACY:
                return Utils.maxIndex(votes) == trueClass ? -1.0 : 0.0;

            case LOSS_BRIER: {
                double sum = 0.0;
                for (int k = 0; k < votes.length; k++) {
                    double residual = votes[k] - (k == trueClass ? 1.0 : 0.0);
                    sum += residual * residual;
                }
                return sum;
            }
            case LOSS_HINGE: {
                double bestCompetitor = 0.0;
                for (int k = 0; k < votes.length; k++) {
                    if (k != trueClass && votes[k] > bestCompetitor) {
                        bestCompetitor = votes[k];
                    }
                }
                return Math.max(0.0, 1.0 - (votes[trueClass] - bestCompetitor));
            }
            case LOSS_BALANCED_ACCURACY:
                return Utils.maxIndex(votes) == trueClass ? -classWeight(trueClass) : 0.0;

            default:
                return 0.0;
        }
    }

    /**
     * Clamp a probability into the range where the logarithm is finite and negative.
     *
     * <p>Without this a vote of zero for the true class -- the worst prediction there is --
     * would produce an infinite loss, and the previous workaround of returning 0.0 in that case
     * made it score the same as a perfect prediction. That inverted the ranking: a perturbation
     * that drove the true-class probability to zero registered as no importance at all.</p>
     */
    protected static double clipProbability(double p) {
        if (Double.isNaN(p)) {
            return PROBABILITY_FLOOR;
        }
        return Math.min(1.0, Math.max(PROBABILITY_FLOOR, p));
    }

    /** Record one more observation of {@code classIndex} for the balanced accuracy loss. */
    protected void countClass(int classIndex) {
        if (this.classCounts == null || classIndex < 0 || classIndex >= this.classCounts.length) {
            return;
        }
        if (this.classCounts[classIndex] == 0.0) {
            this.observedClasses++;
        }
        this.classCounts[classIndex]++;
        this.classCountTotal++;
    }

    /**
     * Weight of {@code classIndex} under the balanced accuracy loss: the mean count over the
     * classes observed so far divided by this class's own count. A class of average frequency
     * weighs 1, a rare class weighs more, so that a minority class is not drowned out of the
     * importance estimates by the majority one.
     */
    protected double classWeight(int classIndex) {
        if (this.classCounts == null || classIndex < 0 || classIndex >= this.classCounts.length
                || this.observedClasses == 0 || this.classCounts[classIndex] <= 0.0) {
            return 1.0;
        }
        return (this.classCountTotal / this.observedClasses) / this.classCounts[classIndex];
    }

    /**
     * Turn a raw vote array into a distribution over exactly {@code numClasses} entries.
     *
     * <p>Two things are repaired here. First, learners report votes as the class distribution
     * they have observed so far, which is truncated after the highest class index they have
     * actually seen: a leaf that is pure for class 0 returns an array of length one. The
     * missing classes have a vote of zero, so the array is padded rather than discarded.
     * Discarding it would bias the estimates for exactly those features whose perturbation
     * sends the instance into a pure leaf, which are the informative ones. Second, unlike
     * {@link Utils#normalize(double[])} this does not throw when the learner abstains with an
     * all-zero vector, which happens routinely at the start of a stream; the resulting flat
     * scores cancel out between the original and the perturbed prediction, so an uninformed
     * learner contributes nothing instead of noise.</p>
     */
    protected double[] toDistribution(double[] votes, int numClasses) {
        double[] distribution = new double[Math.max(numClasses, 0)];
        if (votes == null) {
            return distribution;
        }
        double sum = 0.0;
        for (int i = 0; i < votes.length && i < distribution.length; i++) {
            distribution[i] = votes[i];
            sum += votes[i];
        }
        // A NaN sum fails the first test already, so only infinity needs ruling out explicitly.
        if (sum > 0.0 && !Double.isInfinite(sum)) {
            for (int i = 0; i < distribution.length; i++) {
                distribution[i] /= sum;
            }
        }
        return distribution;
    }

    /** Allocate the importance matrix and the feature index mapping on the first opportunity. */
    protected void initTracker(int numClasses, int numAttributes, int classIndex) {
        if (this.importanceTracker != null) {
            return;
        }
        int numFeatures = numAttributes - 1;
        if (numFeatures <= 0) {
            return;
        }
        this.featureIndices = new int[numFeatures];
        int next = 0;
        for (int i = 0; i < numAttributes && next < numFeatures; i++) {
            if (i != classIndex) {
                this.featureIndices[next++] = i;
            }
        }
        int numRows = this.deactivateClasswiseImportanceOption.isSet() ? 1 : Math.max(1, numClasses);
        this.importanceTracker = new double[numRows][numFeatures];
        this.classCounts = new double[Math.max(1, numClasses)];
    }

    /**
     * The current importance estimates, one value per feature. When class-wise tracking is
     * enabled the per-class estimates are averaged.
     *
     * <p>Note that permutation importances are signed: a feature the model does not use can end
     * up slightly negative. Normalization therefore divides by the sum of the absolute values
     * rather than by the plain sum, which keeps the sign and bounds the scores to [-1, 1].</p>
     */
    @Override
    public double[] getFeatureImportances(boolean normalize) {
        if (this.importanceTracker == null) {
            return null;
        }
        double[] importances = new double[this.importanceTracker[0].length];
        for (int i = 0; i < importances.length; i++) {
            double sum = 0.0;
            for (int c = 0; c < this.importanceTracker.length; c++) {
                sum += this.importanceTracker[c][i];
            }
            importances[i] = sum / this.importanceTracker.length;
        }

        if (normalize) {
            double absSum = 0.0;
            for (double importance : importances) {
                absSum += Math.abs(importance);
            }
            if (absSum > 0.0) {
                for (int i = 0; i < importances.length; i++) {
                    importances[i] /= absSum;
                }
            }
        }
        return importances;
    }

    @Override
    public int[] getTopKFeatures(int k, boolean normalize) {
        double[] importances = getFeatureImportances(normalize);
        if (importances == null) {
            return null;
        }
        if (k > importances.length) {
            k = importances.length;
        }

        double[] remaining = importances.clone();
        int[] topK = new int[k];
        for (int i = 0; i < k; i++) {
            int currentTop = Utils.maxIndex(remaining);
            topK[i] = currentTop;
            remaining[currentTop] = Double.NEGATIVE_INFINITY;
        }
        return topK;
    }

    /**
     * The raw importance matrix, indexed by [class][feature] (a single row when class-wise
     * tracking is disabled). Returns null before the first instance is seen.
     */
    public double[][] getImportanceMatrix() {
        return this.importanceTracker;
    }

    @Override
    public Classifier[] getSubClassifiers() {
        return this.classifier == null ? null : new Classifier[]{this.classifier};
    }

    /**
     * One measurement per tracked importance value, named after the attribute it belongs to
     * (and, when class-wise tracking is on, after the class label) instead of by raw index.
     *
     * <p>The measurements and their order have to stay identical across calls, since MOA
     * derives the CSV header from the first invocation only. Hence the names come from the
     * model context, which is fixed for the run, and the values are never reordered by rank.</p>
     */
    @Override
    public Measurement[] getModelMeasurementsImpl() {
        if (this.importanceTracker == null || this.importanceTracker[0].length == 0) {
            return new Measurement[0];
        }

        int numRows = this.importanceTracker.length;
        int numFeatures = this.importanceTracker[0].length;
        boolean classwise = numRows > 1;

        Measurement[] measurements = new Measurement[numRows * numFeatures];
        int next = 0;
        for (int c = 0; c < numRows; c++) {
            for (int f = 0; f < numFeatures; f++) {
                String name = classwise
                        ? "importance(" + featureName(f) + " | " + classLabel(c) + ")"
                        : "importance(" + featureName(f) + ")";
                measurements[next++] = new Measurement(name, this.importanceTracker[c][f]);
            }
        }
        return measurements;
    }

    /** Display name of tracked feature {@code i}: its attribute name when known, else its index. */
    protected String featureName(int i) {
        InstancesHeader context = this.getModelContext();
        if (context != null && this.featureIndices != null && i < this.featureIndices.length
                && this.featureIndices[i] < context.numAttributes()) {
            return displayable(context.attribute(this.featureIndices[i]).name(), "att" + i);
        }
        return "att" + i;
    }

    /** Display name of class {@code c}: its class label when known, else its index. */
    protected String classLabel(int c) {
        InstancesHeader context = this.getModelContext();
        if (context != null && context.classAttribute() != null
                && context.classAttribute().isNominal() && c < context.numClasses()) {
            return displayable(context.classAttribute().value(c), "class" + c);
        }
        return "class" + c;
    }

    /** Measurement names end up as CSV column headers, so keep the separators out of them. */
    private static String displayable(String name, String fallback) {
        if (name == null || name.trim().isEmpty()) {
            return fallback;
        }
        return name.trim().replaceAll("[,\"\r\n]", "_");
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {
    }

    @Override
    public boolean isRandomizable() {
        return true;
    }
}
