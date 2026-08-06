/*
 *    RandomSearchClassifier.java
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
 */
package moa.classifiers.AutoML;

import com.github.javacliparser.FileOption;
import com.github.javacliparser.FlagOption;
import com.github.javacliparser.IntOption;
import com.github.javacliparser.MultiChoiceOption;
import com.github.javacliparser.StringOption;
import com.yahoo.labs.samoa.instances.Instance;
import moa.capabilities.CapabilitiesHandler;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.AutoML.Parameters.CategoricalParameter;
import moa.classifiers.AutoML.Parameters.DoubleParameter;
import moa.classifiers.AutoML.Parameters.IntParameter;
import moa.classifiers.AutoML.Parameters.Parameter;
import moa.classifiers.AutoML.space.ConfigurationSpace;
import moa.classifiers.AutoML.space.LearnerConfigurator;
import moa.classifiers.AutoML.space.ParameterSpec;
import moa.classifiers.Classifier;
import moa.classifiers.core.driftdetection.ChangeDetector;
import moa.classifiers.MultiClassClassifier;
import moa.core.InstanceExample;
import moa.core.Measurement;
import moa.core.SizeOf;
import moa.options.ClassOption;
import moa.evaluation.BasicClassificationPerformanceEvaluator;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Random search over a streaming hyperparameter space: a pool of candidates
 * carries randomly drawn configurations, and whenever a candidate beats the
 * incumbent over an evaluation window it takes its place.
 *
 * <p>Candidates are configured through {@link LearnerConfigurator}, i.e. by
 * writing MOA {@link com.github.javacliparser.Option}s, so any MOA classifier
 * can be tuned as shipped.
 *
 * @author Daniel Nowak Assis (daniel dot nowak-assis at lip6 dot fr)
 */
public class RandomSearchClassifier extends AbstractClassifier implements MultiClassClassifier,
        CapabilitiesHandler, Serializable, HPOMethod {

    private static final long serialVersionUID = 1L;

    public FileOption configurationFileOption = new FileOption("configurationFile", 'f',
            "Search space in JSON format.", null, ".json", false);

    public StringOption searchSpaceOption = new StringOption("searchSpace", 's',
            "Search space as inline JSON. Takes precedence over configurationFile when set,"
            + " so that a caller holding the space in memory need not write a file.", "");

    public IntOption periodicityOption = new IntOption("periodicity", 'g',
            "Number of instances between candidate evaluations.", 1000, 1, Integer.MAX_VALUE);

    public FlagOption randomInitialParametersOption = new FlagOption("randomInitialParameters", 'R',
            "Draw the initial configuration at random instead of using the values declared in the search space.");

    public IntOption numberOfCandidatesOption = new IntOption("numberOfCandidates", 'n',
            "Number of candidate models in the pool.", 10, 1, Integer.MAX_VALUE);

    public MultiChoiceOption metricOption = new MultiChoiceOption("metric", 'm',
            "Metric to optimize the model.", MetricUtils.METRIC_NAMES, MetricUtils.METRIC_DESCRIPTIONS, 4);

    public FlagOption resetLearningOption = new FlagOption("resetLearning", 'L',
            "Reset candidate learning instead of copying internal model state from the best classifier.");

    public FlagOption driftDetectionOption = new FlagOption("driftDetection", 'd',
            "Enable drift detection on the incumbent's prediction error.");

    public ClassOption driftDetectorOption = new ClassOption("driftDetector", 'D',
            "Change detector to use when drift detection is enabled; on a detected"
            + " drift the whole search is reinitialized from the search space.",
            ChangeDetector.class, "ADWINChangeDetector");

    public IntOption numberOfJobsOption = new IntOption("numberOfJobs", 'j',
            "Total number of concurrent jobs used for processing (-1 = as much as possible, 0 = do not use multithreading)",
            1, -1, Integer.MAX_VALUE);

    protected static final int SINGLE_THREAD = 0;

    public long instanceCount;

    public long statesEvaluated;

    public Classifier classifier;

    protected Classifier[] candidates;

    ArrayList<Parameter> classifierParameters;

    ArrayList<ArrayList<Parameter>> candidatesParameters;

    int numericalParameters;

    protected BasicClassificationPerformanceEvaluator evaluatorClassifier;

    protected BasicClassificationPerformanceEvaluator[] evaluatorCandidates;

    int evaluationInstances = 0;

    /** Detector on the incumbent's error; {@code null} unless drift detection is on. */
    protected ChangeDetector driftDetector;

    /** Cumulative number of drifts signalled over the run; not reset by a restart. */
    protected long driftsDetected;

    protected boolean[] optimizeMask;

    /** Search space and the boundary through which every learner is configured. */
    protected LearnerConfigurator configurator;

    /**
     * Parameter list most recently written onto each candidate. Needed when a
     * candidate has to be rebuilt rather than reconfigured in place.
     */
    protected ArrayList<Parameter>[] appliedParameters;

    /**
     * Candidates actually trained and evaluated, i.e. the active prefix of the
     * pool. The pool itself is always allocated at {@code numberOfCandidatesOption}
     * so the budget can move without reallocating anything.
     */
    protected int activeCandidates;

    /** Budget requested by an external controller, applied at the next window boundary. */
    protected int pendingActiveCandidates = -1;

    /**
     * Pool training the incumbent and the candidates concurrently; {@code null}
     * when running single-threaded. Transient because an executor cannot be
     * serialized, so it is (re)created on demand by {@link #initExecutor()}.
     */
    protected transient ExecutorService executor;

    @Override
    public void setOptimizableParameters(boolean[] mask) { this.optimizeMask = mask; }

    @Override
    public int getActiveCandidates() { return this.activeCandidates; }

    @Override
    public int setActiveCandidates(int n) {
        this.pendingActiveCandidates = Math.max(1, Math.min(n, this.numberOfCandidatesOption.getValue()));
        return this.pendingActiveCandidates;
    }

    /**
     * Pin every frozen hyperparameter (optimizeMask[k] == false) of every
     * candidate to the incumbent value, and re-apply it to the candidate model
     * so the frozen value is actually used.
     */
    private void applyFreezeMask() {
        if (this.optimizeMask == null) return;
        for (int i = 0; i < this.activeCandidates; i++) {
            ArrayList<Parameter> params = this.candidatesParameters.get(i);
            HPOMethod.freezeToIncumbent(this.optimizeMask, params, this.classifierParameters);
            for (int k = 0; k < params.size() && k < this.optimizeMask.length; k++) {
                if (this.optimizeMask[k]) continue;
                this.configurator.applyLive(this.candidates[i], params.get(k));
            }
        }
    }

    @Override
    public boolean isRandomizable() {
        return true;
    }

    @Override
    public double[] getVotesForInstance(Instance inst) {
        //Main classifiers votes
        return this.classifier.getVotesForInstance(inst);
    }

    /**
     * Creates the training pool on first use, and again after deserialization.
     * A pool larger than the batch trained per instance would leave threads idle,
     * so the requested job count is capped at the incumbent plus the whole pool.
     */
    protected void initExecutor() {
        if (this.executor != null) return;
        int numberOfJobs = this.numberOfJobsOption.getValue() == -1
                ? Runtime.getRuntime().availableProcessors()
                : this.numberOfJobsOption.getValue();
        numberOfJobs = Math.min(numberOfJobs, this.numberOfCandidatesOption.getValue() + 1);
        // SINGLE_THREAD and requesting a single thread are equivalent: training
        // then happens in-place and this.executor stays null.
        if (numberOfJobs != SINGLE_THREAD && numberOfJobs != 1) {
            // Daemon threads: a live pool must not keep the JVM alive after the
            // task that ran this learner has finished.
            this.executor = Executors.newFixedThreadPool(numberOfJobs, runnable -> {
                Thread thread = new Thread(runnable);
                thread.setDaemon(true);
                return thread;
            });
        }
    }

    @Override
    public void cleanThreads() {
        if (this.executor != null) {
            this.executor.shutdownNow();
            this.executor = null;
        }
    }

    @Override
    @SuppressWarnings("unchecked")
    public void resetLearningImpl() {
        cleanThreads(); // shut down any pool from a previous reset before creating a new one
        driftDetector = driftDetectionOption.isSet()
                ? ((ChangeDetector) getPreparedClassOption(driftDetectorOption)).copy()
                : null;
        this.classifierParameters = new ArrayList<Parameter>();
        this.candidatesParameters = new ArrayList<>();
        for (int i = 0; i < this.numberOfCandidatesOption.getValue(); i++) {
            this.candidatesParameters.add(new ArrayList<Parameter>());
        }


        this.evaluatorClassifier = new BasicClassificationPerformanceEvaluator();
        MetricUtils.configure(this.evaluatorClassifier);
        this.evaluatorCandidates = new BasicClassificationPerformanceEvaluator[this.numberOfCandidatesOption.getValue()];
        for (int i = 0; i < this.numberOfCandidatesOption.getValue(); i++) {
            this.evaluatorCandidates[i] = new BasicClassificationPerformanceEvaluator();
            MetricUtils.configure(this.evaluatorCandidates[i]);
        }

        this.appliedParameters = new ArrayList[this.numberOfCandidatesOption.getValue()];

        this.setConfigurations();
        this.instanceCount = 0;
        this.statesEvaluated = 0;
        this.activeCandidates = this.numberOfCandidatesOption.getValue();
        this.pendingActiveCandidates = -1;
    }

    /**
     * Reads the search space, creates the incumbent and the candidate pool, and
     * seeds the per-parameter random generators.
     */
    public void setConfigurations() {
        try {
            this.numericalParameters = 0;

            ConfigurationSpace space = ConfigurationSpace.resolve(this.configurationFileOption.getValue(), this.searchSpaceOption.getValue());
            this.configurator = new LearnerConfigurator(space);
            this.configurator.validate();

            this.classifier = this.configurator.newLearner(this.classifierRandom.nextInt());

            this.candidates = new Classifier[this.numberOfCandidatesOption.getValue()];
            for (int i = 0; i < this.numberOfCandidatesOption.getValue(); i++) {
                // Built one by one rather than copied from a single base, so that
                // each candidate carries its own random seed.
                this.candidates[i] = this.configurator.newLearner(this.classifierRandom.nextInt());
            }

            for (ParameterSpec spec : space.parameters) {
                switch (spec.type) {
                    case ParameterSpec.TYPE_CATEGORICAL: {
                        int active;
                        if (!this.randomInitialParametersOption.isSet()) {
                            active = spec.active;
                        } else {
                            active = this.classifierRandom.nextInt(spec.values.length);
                        }
                        this.classifierParameters.add(new CategoricalParameter(spec.name, spec.values,
                                active, new Random(this.classifierRandom.nextLong())));
                        for (int i = 0; i < this.numberOfCandidatesOption.getValue(); i++) {
                            this.candidatesParameters.get(i).add(new CategoricalParameter(spec.name,
                                    spec.values, active, new Random(this.classifierRandom.nextLong())));
                        }
                        break;
                    }
                    case ParameterSpec.TYPE_INT: {
                        int[] range = new int[]{(int) spec.range[0], (int) spec.range[1]};
                        int value;
                        if (!this.randomInitialParametersOption.isSet()) {
                            value = (int) spec.value;
                        } else {
                            value = this.classifierRandom.nextInt(range[1] + 1 - range[0]) + range[0];
                        }
                        this.classifierParameters.add(new IntParameter(spec.name, value, range,
                                new Random(this.classifierRandom.nextLong())));
                        for (int i = 0; i < this.numberOfCandidatesOption.getValue(); i++) {
                            this.candidatesParameters.get(i).add(new IntParameter(spec.name, value, range,
                                    new Random(this.classifierRandom.nextLong())));
                        }
                        this.numericalParameters++;
                        break;
                    }
                    case ParameterSpec.TYPE_DOUBLE: {
                        double[] range = new double[]{spec.range[0], spec.range[1]};
                        double value;
                        if (!this.randomInitialParametersOption.isSet()) {
                            value = spec.value;
                        } else {
                            value = range[0] + (range[1] - range[0]) * this.classifierRandom.nextDouble();
                        }
                        this.classifierParameters.add(new DoubleParameter(spec.name, value, range,
                                new Random(this.classifierRandom.nextLong())));
                        for (int i = 0; i < this.numberOfCandidatesOption.getValue(); i++) {
                            this.candidatesParameters.get(i).add(new DoubleParameter(spec.name, value, range,
                                    new Random(this.classifierRandom.nextLong())));
                        }
                        this.numericalParameters++;
                        break;
                    }
                    default:
                        break;
                }
            }

            // Put the incumbent on its declared starting configuration.
            this.classifier = this.configurator.instantiate(this.classifierParameters, this.classifierRandom.nextInt());

        } catch (Exception e) {
            throw new IllegalStateException("Could not set up " + getClass().getSimpleName()
                    + " from " + ConfigurationSpace.describeSource(this.configurationFileOption.getValue(), this.searchSpaceOption.getValue()) + ": " + e.getMessage(), e);
        }
    }

    public void deepCopyList(int bestPerforming) {
        //Swap parameters from candidates to main classifier
        for (int i = 0; i < this.classifierParameters.size(); i++) {
            Parameter parameterS = this.candidatesParameters.get(bestPerforming).get(i);
            Parameter parameterC = this.classifierParameters.get(i);
            switch (parameterS.type) {
                case Parameter.TYPE_INT:
                    ((IntParameter) parameterC).value = ((IntParameter) parameterS).value;
                    ((IntParameter) parameterC).range = ((IntParameter) parameterS).range;
                    break;
                case Parameter.TYPE_DOUBLE:
                    ((DoubleParameter) parameterC).value = ((DoubleParameter) parameterS).value;
                    ((DoubleParameter) parameterC).range = ((DoubleParameter) parameterS).range;
                    break;
                case Parameter.TYPE_CATEGORICAL:
                    ((CategoricalParameter) parameterC).values = ((CategoricalParameter) parameterS).values;
                    ((CategoricalParameter) parameterC).active = ((CategoricalParameter) parameterS).active;
                    break;
            }
        }
    }

    public void checkParameterChange() {
        try {
            double maxAccuracy = 0;
            double classifierAcc = 0;
            int bestPerforming = 0;

            if (this.statesEvaluated != 0) {
                maxAccuracy = MetricUtils.getScore(this.evaluatorCandidates[0].getPerformanceMeasurements(),
                        this.metricOption.getChosenIndex());
                for (int i = 1; i < this.activeCandidates; i++) {
                    double currentAcc = MetricUtils.getScore(this.evaluatorCandidates[i].getPerformanceMeasurements(),
                            this.metricOption.getChosenIndex());
                    if (currentAcc > maxAccuracy) {
                        bestPerforming = i;
                        maxAccuracy = currentAcc;
                    }
                }

                classifierAcc = MetricUtils.getScore(this.evaluatorClassifier.getPerformanceMeasurements(),
                        this.metricOption.getChosenIndex());
            }

            // Swapping classifiers if candidates acc is greater than main classifier
            if (maxAccuracy > classifierAcc) {
                this.swapClassifiers(bestPerforming);
            }

            // Apply an externally requested budget only here: the window is
            // scored with the budget it ran under, and the new one takes effect
            // from the candidates regenerated below.
            if (this.pendingActiveCandidates > 0) {
                this.activeCandidates = this.pendingActiveCandidates;
                this.pendingActiveCandidates = -1;
            }

            boolean live = this.configurator.canApplyLive();

            if (live) {
                for (int i = 0; i < this.activeCandidates; i++) {
                    if (resetLearningOption.isSet()) {
                        this.candidates[i].resetLearning();
                    } else {
                        this.candidates[i] = this.classifier.copy();
                        LearnerConfigurator.reseedCopy(this.candidates[i], this.classifierRandom.nextInt());
                    }
                }
            }

            // Parameter change of candidates. Candidate i is drawn from, and
            // configured out of, its own parameter list, so its model and its
            // bookkeeping agree and deepCopyList records the winner's real
            // configuration.
            for (int i = 0; i < this.activeCandidates; i++) {
                ArrayList<Parameter> listp = this.candidatesParameters.get(i);
                for (Parameter parameter : listp) {
                    this.changeStateParameter(parameter, i);
                }
                this.appliedParameters[i] = listp;
            }

            if (!live) {
                // At least one hyperparameter is only read when the learner is
                // built, so the candidate has to be rebuilt around it.
                for (int i = 0; i < this.activeCandidates; i++) {
                    this.candidates[i] = this.configurator.instantiate(this.appliedParameters[i], this.classifierRandom.nextInt());
                }
            }

            // Freeze the hyperparameters outside the optimized subset to the incumbent.
            this.applyFreezeMask();

            //reset evaluators
            this.evaluatorClassifier.reset();

            for (int i = 0; i < this.activeCandidates; i++) {
                this.evaluatorCandidates[i].reset();
            }

        } catch (Exception e) {
            throw new IllegalStateException("Failed to update candidates in "
                    + getClass().getSimpleName() + ": " + e.getMessage(), e);
        }
    }

    public void swapClassifiers(int bestPerforming) {
        this.classifier = this.candidates[bestPerforming].copy();
        this.deepCopyList(bestPerforming);
    }

    @Override
    public void changeStateParameter(Parameter parameter, int index) {
        //Draw a new value and push it into the candidate
        parameter.changeParameter();
        if (this.configurator.canApplyLive()) {
            this.configurator.applyLive(this.candidates[index], parameter);
        }
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {
        double[] incumbentVotes = driftDetector != null ? getVotesForInstance(inst) : null;
        initExecutor();
        //check for change in parameters
        this.evaluationInstances++;

        if ((this.statesEvaluated == 0) || this.evaluationInstances >= this.periodicityOption.getValue()) {
            this.checkParameterChange();
            this.statesEvaluated++;
            this.evaluationInstances = 0;
        }

        //update evaluators (update metrics)
        InstanceExample example = new InstanceExample(inst);
        this.evaluatorClassifier.addResult(example, this.classifier.getVotesForInstance(inst));
        for (int i = 0; i < this.activeCandidates; i++) {
            this.evaluatorCandidates[i].addResult(example, this.candidates[i].getVotesForInstance(inst));
        }

        //train classifiers
        if (this.executor == null) {
            this.classifier.trainOnInstance(inst);
            for (int i = 0; i < this.activeCandidates; i++) {
                this.candidates[i].trainOnInstance(inst);
            }
        } else {
            // The incumbent is an independent model, so it joins the batch as one
            // more task. Every model owns its state, hence no synchronization here.
            Collection<TrainingRunnable> trainers = new ArrayList<TrainingRunnable>();
            trainers.add(new TrainingRunnable(this.classifier, inst));
            for (int i = 0; i < this.activeCandidates; i++) {
                trainers.add(new TrainingRunnable(this.candidates[i], inst));
            }
            try {
                this.executor.invokeAll(trainers);
            } catch (InterruptedException ex) {
                throw new RuntimeException("Could not call invokeAll() on training threads.");
            }
        }

        this.instanceCount++;

        if (incumbentVotes != null) checkDrift(inst, incumbentVotes);
    }

    @Override
    public long measureByteSize() {
        long candidateSize = 0;
        for (int i = 0; i < this.activeCandidates; i++) {
            candidateSize += this.candidates[i].measureByteSize();
        }
        return SizeOf.sizeOf(this) + this.classifier.measureByteSize() + candidateSize;
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {

        for (Classifier c : candidates) {
            c.getModelMeasurements();
        }

        ArrayList<Measurement> parameters = new ArrayList<>();
        parameters.add(new Measurement("driftsDetected", driftsDetected));

        for (Measurement m : this.classifier.getModelMeasurements()) {
            parameters.add(m);
        }
        for (Parameter p : this.classifierParameters) {
            switch (p.type) {
                case Parameter.TYPE_INT:
                    parameters.add(new Measurement(p.name, ((IntParameter) p).value));
                    break;
                case Parameter.TYPE_DOUBLE:
                    parameters.add(new Measurement(p.name, ((DoubleParameter) p).value));
                    break;
                case Parameter.TYPE_CATEGORICAL:
                    parameters.add(new Measurement(p.name, ((CategoricalParameter) p).active));
                    break;
            }
        }
        Measurement[] measurements = new Measurement[parameters.size()];
        measurements = parameters.toArray(measurements);
        return measurements;
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {
    }

    @Override
    public ArrayList<Parameter> getReferenceParameters() { return this.classifierParameters; }

    @Override
    public ArrayList<ArrayList<Parameter>> getCandidateParameters() { return this.candidatesParameters; }

    @Override
    public double getCandidateScore(int i) {
        return MetricUtils.getScore(this.evaluatorCandidates[i].getPerformanceMeasurements(),
                this.metricOption.getChosenIndex());
    }

    @Override
    public double getClassifierScore() {
        return MetricUtils.getScore(this.evaluatorClassifier.getPerformanceMeasurements(),
                this.metricOption.getChosenIndex());
    }

    @Override
    public int getNumberOfCandidates() { return this.numberOfCandidatesOption.getValue(); }

    @Override
    public long getStatesEvaluatedCount() { return this.statesEvaluated; }

    @Override
    public int getEvaluationInstancesCount() { return this.evaluationInstances; }

    @Override
    public int getPeriodicity() { return this.periodicityOption.getValue(); }

    @Override
    public Classifier getMainClassifier() { return this.classifier; }

    @Override
    public String getConfigurationFile() { return this.configurationFileOption.getValue(); }

    private int argmax(double[] arr) {
        int best = 0;
        for (int i = 1; i < arr.length; i++)
            if (arr[i] > arr[best]) best = i;
        return best;
    }

    /**
     * Feeds the incumbent's prediction error to the change detector and, when a
     * drift is signalled, reinitializes the whole search from the search space.
     * The signal is the 0/1 misclassification indicator.
     */
    private void checkDrift(Instance inst, double[] incumbentVotes) {
        driftDetector.input(argmax(incumbentVotes) != (int) inst.classValue() ? 1.0 : 0.0);
        if (driftDetector.getChange()) {
            driftsDetected++;
            resetLearningImpl();
        }
    }

    /**
     * Inner class to assist with the multi-thread execution.
     */
    protected class TrainingRunnable implements Runnable, Callable<Integer> {
        final private Classifier learner;
        final private Instance instance;

        public TrainingRunnable(Classifier learner, Instance instance) {
            this.learner = learner;
            this.instance = instance;
        }

        @Override
        public void run() {
            this.learner.trainOnInstance(this.instance);
        }

        @Override
        public Integer call() {
            run();
            return 0;
        }
    }
}
