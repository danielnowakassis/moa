/*
 *    iPFIRegressor.java
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
import moa.classifiers.Regressor;
import moa.core.Measurement;
import moa.core.Utils;
import moa.options.ClassOption;

/**
 * Incremental Permutation Feature Importance (iPFI) for regression.
 *
 * <p>The regression counterpart of {@link iPFIClassifier}. A meta regressor that trains any base
 * regressor and, at the same time, incrementally estimates the permutation feature importance of
 * every input feature. For each incoming instance the error of the base learner is measured on
 * the original instance and on {@code nInteractions} perturbed copies, where the value of a
 * single feature is replaced by the value observed for that feature in a randomly drawn instance
 * from a sliding window of recent instances (sampling-based imputation). The difference between
 * the perturbed error and the original error is the importance contribution of that feature, and
 * it is aggregated over time with an exponentially weighted moving average controlled by
 * {@code smoothingAlpha}.</p>
 *
 * <p>Unlike the classification case there is no class to condition on, so a single importance
 * value is tracked per feature. Predictions are simply delegated to the base learner, so
 * wrapping a regressor with this class does not change its predictive behaviour, only its
 * cost.</p>
 *
 * <p>Importances are expressed in the units of the loss, and therefore in the units of the
 * target. That makes them incomparable across streams with different target scales, and it lets
 * a single outlier dominate the average for a long time under squared error. Absolute error is
 * the default for that reason, and {@code -z} additionally divides every error by a running
 * estimate of the standard deviation of the target, which makes the scores scale-free.</p>
 *
 * <p>See details in:<br> Fabian Fumagalli, Maximilian Muschalik, Eyke H&uuml;llermeier,
 * Barbara Hammer. Incremental Permutation Feature Importance (iPFI): Towards Online
 * Explanations on Data Streams. Machine Learning, 2023.</p>
 *
 * <p>Parameters:</p> <ul>
 * <li>-l : Regressor to train and to be analyzed.</li>
 * <li>-f : Loss function used to score the (perturbed) predictions: absolute error, squared
 * error or Huber loss.</li>
 * <li>-d : Transition point of the Huber loss, in units of the (possibly standardized) error.</li>
 * <li>-z : If set, divide the errors by a running estimate of the standard deviation of the
 * target, so that the importances no longer depend on the scale of the target.</li>
 * <li>-s : Smoothing factor of the exponentially weighted moving average.</li>
 * <li>-n : Number of perturbations sampled per feature per instance.</li>
 * <li>-w : Size of the sliding window the replacement values are drawn from.</li>
 * </ul>
 *
 * @author Daniel Nowak Assis (daniel dot nowak-assis at lip6 dot fr)
 */
public class iPFIRegressor extends AbstractClassifier implements FeatureImportanceLearner,
        Regressor, CapabilitiesHandler {

    private static final long serialVersionUID = 1L;

    protected static final int LOSS_ABSOLUTE_ERROR = 0;
    protected static final int LOSS_SQUARED_ERROR = 1;
    protected static final int LOSS_HUBER = 2;

    /** Smallest target standard deviation accepted as a divisor, to keep a constant target safe. */
    protected static final double MIN_TARGET_SCALE = 1e-12;

    @Override
    public String getPurposeString() {
        return "Trains a base regressor and incrementally estimates the permutation "
                + "importance of each feature (iPFI).";
    }

    public ClassOption baseLearnerOption = new ClassOption("baseLearner", 'l',
            "Regressor to train.", Regressor.class, "moa.classifiers.trees.FIMTDD");

    public MultiChoiceOption lossFunctionOption = new MultiChoiceOption("lossFunction", 'f',
            "Which loss to use. Absolute error keeps the importances in the units of the target "
                    + "and is robust to outliers, squared error is the textbook choice but lets a "
                    + "single large residual dominate the average, and Huber interpolates between "
                    + "the two.",
            // The fourth argument holds the labels accepted on the command line and the fifth
            // their descriptions, so the short codes belong in the first of the two.
            new String[]{"ae", "se", "huber"},
            new String[]{"Absolute Error", "Squared Error", "Huber Loss"}, LOSS_ABSOLUTE_ERROR);

    public FloatOption huberDeltaOption = new FloatOption("huberDelta", 'd',
            "Residual at which the Huber loss switches from squared to absolute, expressed in "
                    + "units of the error actually fed to the loss.", 1.0, 0.0, Double.MAX_VALUE);

    public FlagOption standardizeErrorsOption = new FlagOption("standardizeErrors", 'z',
            "If set, divide every error by a running estimate of the standard deviation of the "
                    + "target, making the importances independent of the scale of the target.");

    public FloatOption smoothingAlphaOption = new FloatOption("smoothingAlpha", 's',
            "Smoothing factor of the exponentially weighted moving average used to aggregate "
                    + "the per-instance importance contributions.", 0.001, 0.0, 1.0);

    public IntOption nInteractionsOption = new IntOption("nInteractions", 'n',
            "Number of perturbations sampled per feature per instance.", 1, 1, Integer.MAX_VALUE);

    public IntOption maxWindowSizeOption = new IntOption("maxWindowSize", 'w',
            "Size of the sliding window the replacement values are sampled from.",
            500, 1, Integer.MAX_VALUE);

    /** Importance estimate of each tracked feature. */
    protected double[] importanceTracker;

    /** Attribute index of each tracked feature, i.e. every attribute but the target attribute. */
    protected int[] featureIndices;

    /** Sliding window of recent instances, used as the source of the replacement values. */
    protected Instances window;

    protected Classifier regressor;

    protected long instancesSeen;

    // Welford accumulators for the running standard deviation of the target.
    protected double targetMean;
    protected double targetM2;
    protected long targetCount;

    @Override
    public void setModelContext(InstancesHeader context) {
        super.setModelContext(context);
        if (context != null) {
            this.window = new Instances(context, 0);
            this.window.setClassIndex(context.classIndex());
            initTracker(context.numAttributes(), context.classIndex());
        }
        if (this.regressor != null) {
            this.regressor.setModelContext(context);
        }
    }

    @Override
    public void resetLearningImpl() {
        Object baseLearner = getPreparedClassOption(this.baseLearnerOption);
        if (!(baseLearner instanceof Classifier)) {
            throw new IllegalArgumentException("The base learner of " + this.getClass().getName()
                    + " must be a trainable regressor, but " + baseLearner.getClass().getName()
                    + " is not a moa.classifiers.Classifier.");
        }
        this.regressor = ((Classifier) baseLearner).copy();

        if (this.getModelContext() != null) {
            this.regressor.setModelContext(this.getModelContext());
        }
        this.regressor.resetLearning();

        this.instancesSeen = 0;
        this.importanceTracker = null;
        this.featureIndices = null;
        this.targetMean = 0.0;
        this.targetM2 = 0.0;
        this.targetCount = 0;
    }

    @Override
    public double[] getVotesForInstance(Instance inst) {
        return this.regressor.getVotesForInstance(inst);
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {
        if (this.window == null) {
            this.window = new Instances(inst.dataset(), 0);
            this.window.setClassIndex(inst.classIndex());
        }
        initTracker(inst.numAttributes(), inst.classIndex());
        // Updated before the importances are, so that the scale is already defined for the
        // current instance. Only real instances count, never the perturbed copies.
        updateTargetScale(inst.classValue());

        // The first instance only seeds the window: there is nothing to sample replacements from yet.
        if (this.instancesSeen > 0) {
            updateImportances(inst);
        }

        this.instancesSeen++;
        this.window.add(inst);
        while (this.window.size() > this.maxWindowSizeOption.getValue()) {
            this.window.delete(0);
        }

        this.regressor.trainOnInstance(inst);
    }

    /**
     * Estimate the importance contribution of every feature for a single instance and fold it
     * into the exponentially weighted moving average.
     */
    protected void updateImportances(Instance inst) {
        double target = inst.classValue();
        double loss = calculateLoss(target, predict(inst));
        double alpha = this.smoothingAlphaOption.getValue();

        for (int i = 0; i < this.featureIndices.length; i++) {
            double avgLoss = 0.0;
            for (int j = 0; j < this.nInteractionsOption.getValue(); j++) {
                Instance imputed = inst.copy();
                double newValue = this.window.get(this.classifierRandom.nextInt(this.window.size()))
                        .value(this.featureIndices[i]);
                imputed.setValue(this.featureIndices[i], newValue);
                avgLoss += calculateLoss(target, predict(imputed));
            }
            avgLoss /= this.nInteractionsOption.getValue();

            this.importanceTracker[i] =
                    (1 - alpha) * this.importanceTracker[i] + alpha * (avgLoss - loss);
        }
    }

    /**
     * The scalar prediction of the base learner. Regression learners report it as the single
     * entry of the vote array; an empty array means the learner has nothing to say yet, and
     * since the same fallback is then used for the original and the perturbed instance alike,
     * the two losses cancel and such an instance contributes no importance.
     */
    protected double predict(Instance inst) {
        double[] votes = this.regressor.getVotesForInstance(inst);
        if (votes == null || votes.length == 0 || Double.isNaN(votes[0])) {
            return 0.0;
        }
        return votes[0];
    }

    /**
     * Error of a prediction with respect to the true target. The returned value grows as the
     * prediction gets worse, so that the difference between the error under perturbation and the
     * original error is positive for informative features.
     */
    public double calculateLoss(double target, double prediction) {
        double error = prediction - target;
        if (Double.isNaN(error) || Double.isInfinite(error)) {
            return 0.0;
        }
        if (this.standardizeErrorsOption.isSet()) {
            error /= targetScale();
        }

        switch (this.lossFunctionOption.getChosenIndex()) {
            case LOSS_ABSOLUTE_ERROR:
                return Math.abs(error);

            case LOSS_SQUARED_ERROR:
                return error * error;

            case LOSS_HUBER: {
                double delta = this.huberDeltaOption.getValue();
                double magnitude = Math.abs(error);
                return magnitude <= delta
                        ? 0.5 * error * error
                        : delta * (magnitude - 0.5 * delta);
            }
            default:
                return Math.abs(error);
        }
    }

    /** Fold one target value into the running mean and variance (Welford's algorithm). */
    protected void updateTargetScale(double target) {
        if (Double.isNaN(target) || Double.isInfinite(target)) {
            return;
        }
        this.targetCount++;
        double delta = target - this.targetMean;
        this.targetMean += delta / this.targetCount;
        this.targetM2 += delta * (target - this.targetMean);
    }

    /**
     * Running standard deviation of the target, floored so that it is always a safe divisor.
     * Returns 1 until there are enough observations to estimate it, which leaves the early
     * errors on their raw scale rather than exaggerating them.
     */
    protected double targetScale() {
        if (this.targetCount < 2) {
            return 1.0;
        }
        double variance = this.targetM2 / (this.targetCount - 1);
        double deviation = Math.sqrt(variance);
        return deviation > MIN_TARGET_SCALE ? deviation : 1.0;
    }

    /** Allocate the importance array and the feature index mapping on the first opportunity. */
    protected void initTracker(int numAttributes, int classIndex) {
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
        this.importanceTracker = new double[numFeatures];
    }

    /**
     * The current importance estimates, one value per feature.
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
        double[] importances = this.importanceTracker.clone();

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

    /** The raw importance estimates. Returns null before the first instance is seen. */
    public double[] getImportanceVector() {
        return this.importanceTracker;
    }

    @Override
    public Classifier[] getSubClassifiers() {
        return this.regressor == null ? null : new Classifier[]{this.regressor};
    }

    /**
     * One measurement per feature, named after the attribute it belongs to instead of by raw
     * index.
     *
     * <p>The measurements and their order have to stay identical across calls, since MOA
     * derives the CSV header from the first invocation only. Hence the names come from the
     * model context, which is fixed for the run, and the values are never reordered by rank.</p>
     */
    @Override
    public Measurement[] getModelMeasurementsImpl() {
        if (this.importanceTracker == null || this.importanceTracker.length == 0) {
            return new Measurement[0];
        }

        Measurement[] measurements = new Measurement[this.importanceTracker.length];
        for (int f = 0; f < this.importanceTracker.length; f++) {
            measurements[f] = new Measurement("importance(" + featureName(f) + ")",
                    this.importanceTracker[f]);
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
