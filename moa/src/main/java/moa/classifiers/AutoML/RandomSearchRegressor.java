/*
 *    RandomSearchRegressor.java
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
import moa.classifiers.Regressor;
import moa.core.InstanceExample;
import moa.core.Measurement;
import moa.core.SizeOf;
import moa.options.ClassOption;
import moa.evaluation.BasicRegressionPerformanceEvaluator;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Regression counterpart of {@link RandomSearchClassifier}: the same
 * incumbent-plus-candidate-pool search, scored with a regression metric.
 *
 * @author Daniel Nowak Assis (daniel dot nowak-assis at lip6 dot fr)
 */
public class RandomSearchRegressor extends AbstractClassifier implements Regressor,
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
            "Metric to optimize the model.", MetricUtils.REGRESSION_METRIC_NAMES,
            MetricUtils.REGRESSION_METRIC_DESCRIPTIONS, 0);

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

    protected BasicRegressionPerformanceEvaluator evaluatorClassifier;

    protected BasicRegressionPerformanceEvaluator[] evaluatorCandidates;

    int evaluationInstances = 0;

    /** Detector on the incumbent's error; {@code null} unless drift detection is on. */
    protected ChangeDetector driftDetector;

    /** Cumulative number of drifts signalled over the run; not reset by a restart. */
    protected long driftsDetected;

    protected boolean[] optimizeMask;

    protected LearnerConfigurator configurator;

    protected ArrayList<Parameter>[] appliedParameters;

    protected int activeCandidates;

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


        this.evaluatorClassifier = new BasicRegressionPerformanceEvaluator();
        this.evaluatorCandidates = new BasicRegressionPerformanceEvaluator[this.numberOfCandidatesOption.getValue()];
        for (int i = 0; i < this.numberOfCandidatesOption.getValue(); i++) {
            this.evaluatorCandidates[i] = new BasicRegressionPerformanceEvaluator();
        }

        this.appliedParameters = new ArrayList[this.numberOfCandidatesOption.getValue()];

        this.setConfigurations();
        this.instanceCount = 0;
        this.statesEvaluated = 0;
        this.activeCandidates = this.numberOfCandidatesOption.getValue();
        this.pendingActiveCandidates = -1;
    }

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
                        int active = this.randomInitialParametersOption.isSet()
                                ? this.classifierRandom.nextInt(spec.values.length)
                                : spec.active;
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
                        int value = this.randomInitialParametersOption.isSet()
                                ? this.classifierRandom.nextInt(range[1] + 1 - range[0]) + range[0]
                                : (int) spec.value;
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
                        double value = this.randomInitialParametersOption.isSet()
                                ? range[0] + (range[1] - range[0]) * this.classifierRandom.nextDouble()
                                : spec.value;
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

            this.classifier = this.configurator.instantiate(this.classifierParameters, this.classifierRandom.nextInt());

        } catch (Exception e) {
            throw new IllegalStateException("Could not set up " + getClass().getSimpleName()
                    + " from " + ConfigurationSpace.describeSource(this.configurationFileOption.getValue(), this.searchSpaceOption.getValue()) + ": " + e.getMessage(), e);
        }
    }

    public void deepCopyList(int bestPerforming) {
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
            double best = -Double.MAX_VALUE;
            double incumbent = -Double.MAX_VALUE;
            int bestPerforming = 0;

            if (this.statesEvaluated != 0) {
                best = getCandidateScore(0);
                for (int i = 1; i < this.activeCandidates; i++) {
                    double current = getCandidateScore(i);
                    if (current > best) {
                        bestPerforming = i;
                        best = current;
                    }
                }
                incumbent = getClassifierScore();
            }

            if (!Double.isNaN(best) && best > incumbent) {
                this.swapClassifiers(bestPerforming);
            }

            if (this.pendingActiveCandidates > 0) {
                this.activeCandidates = this.pendingActiveCandidates;
                this.pendingActiveCandidates = -1;
            }

            boolean live = this.configurator.canApplyLive();

            if (live) {
                for (int i = 0; i < this.activeCandidates; i++) {
                    if (this.resetLearningOption.isSet()) {
                        this.candidates[i].resetLearning();
                    } else {
                        this.candidates[i] = this.classifier.copy();
                        LearnerConfigurator.reseedCopy(this.candidates[i], this.classifierRandom.nextInt());
                    }
                }
            }

            for (int i = 0; i < this.activeCandidates; i++) {
                ArrayList<Parameter> listp = this.candidatesParameters.get(i);
                for (Parameter parameter : listp) {
                    this.changeStateParameter(parameter, i);
                }
                this.appliedParameters[i] = listp;
            }

            if (!live) {
                for (int i = 0; i < this.activeCandidates; i++) {
                    this.candidates[i] = this.configurator.instantiate(this.appliedParameters[i], this.classifierRandom.nextInt());
                }
            }

            this.applyFreezeMask();

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
        parameter.changeParameter();
        if (this.configurator.canApplyLive()) {
            this.configurator.applyLive(this.candidates[index], parameter);
        }
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {
        double[] incumbentVotes = driftDetector != null ? getVotesForInstance(inst) : null;
        initExecutor();
        this.evaluationInstances++;

        if ((this.statesEvaluated == 0) || this.evaluationInstances >= this.periodicityOption.getValue()) {
            this.checkParameterChange();
            this.statesEvaluated++;
            this.evaluationInstances = 0;
        }

        InstanceExample example = new InstanceExample(inst);
        this.evaluatorClassifier.addResult(example, this.classifier.getVotesForInstance(inst));
        for (int i = 0; i < this.activeCandidates; i++) {
            this.evaluatorCandidates[i].addResult(example, this.candidates[i].getVotesForInstance(inst));
        }

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
        return MetricUtils.getRegressionScore(this.evaluatorCandidates[i].getPerformanceMeasurements(),
                this.metricOption.getChosenIndex());
    }

    @Override
    public double getClassifierScore() {
        return MetricUtils.getRegressionScore(this.evaluatorClassifier.getPerformanceMeasurements(),
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

    /**
     * Feeds the incumbent's prediction error to the change detector and, when a
     * drift is signalled, reinitializes the whole search from the search space.
     * The signal is the absolute error |y - y_pred|.
     */
    private void checkDrift(Instance inst, double[] incumbentVotes) {
        double predicted = incumbentVotes.length > 0 ? incumbentVotes[0] : 0.0;
        driftDetector.input(Math.abs(predicted - inst.classValue()));
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
