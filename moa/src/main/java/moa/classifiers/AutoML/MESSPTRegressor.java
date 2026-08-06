/*
 *    MESSPTRegressor.java
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
import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.github.javacliparser.MultiChoiceOption;
import com.github.javacliparser.StringOption;
import com.yahoo.labs.samoa.instances.Instance;
import moa.capabilities.CapabilitiesHandler;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.AutoML.Parameters.*;
import moa.classifiers.AutoML.space.ConfigurationSpace;
import moa.classifiers.AutoML.space.LearnerConfigurator;
import moa.classifiers.AutoML.space.ParameterSpec;
import moa.classifiers.Classifier;
import moa.classifiers.core.driftdetection.ChangeDetector;
import moa.classifiers.Regressor;
import moa.core.InstanceExample;
import moa.core.Measurement;
import moa.core.SizeOf;
import moa.evaluation.BasicRegressionPerformanceEvaluator;
import moa.options.ClassOption;

import java.io.Serializable;
import java.util.*;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Regression counterpart of {@link MESSPTClassifier}: differential evolution
 * over the streaming hyperparameter space, scored with the regression metrics
 * of {@link MetricUtils}.
 *
 * <p>See details in:<br> Antonio R. Moya, Bruno Veloso, Joao Gama, Sebastian
 * Ventura. Improving hyper-parameter self-tuning for data streams by adapting
 * an evolutionary approach. In Data Mining and Knowledge Discovery,
 * 38(3):1289-1315, Springer, 2023.</p>
 *
 * @author Daniel Nowak Assis (daniel dot nowak-assis at lip6 dot fr)
 */
public class MESSPTRegressor extends AbstractClassifier implements Regressor,
        CapabilitiesHandler, Serializable, HPOMethod {

    public FileOption configurationFileOption = new FileOption("configurationFile", 'f',
            "Search space in JSON format.", null, ".json", false);

    public StringOption searchSpaceOption = new StringOption("searchSpace", 's',
            "Search space as inline JSON. Takes precedence over configurationFile when set,"
            + " so that a caller holding the space in memory need not write a file.", "");

    public IntOption periodicityOption = new IntOption("periodicity", 'g',
            "Number of instances between DE updates.", 1000, 1, Integer.MAX_VALUE);

    public IntOption populationSizeOption = new IntOption("populationSize", 'p',
            "Number of candidate models in the population.", 4, 2, Integer.MAX_VALUE);

    public FloatOption convergenceSphereOption = new FloatOption("convergenceSphere", 'c',
            "Convergence threshold: squared distance between best params across generations.", 0.001, 0.0, Double.MAX_VALUE);

    public FloatOption initialMutationFactorOption = new FloatOption("initialMutationFactor", 'F',
            "Initial DE mutation factor.", 0.5, 0.0, 1.0);

    public FloatOption initialCrossoverRateOption = new FloatOption("initialCrossoverRate", 'C',
            "Initial DE crossover rate.", 0.5, 0.0, 1.0);

    public FloatOption augmentationStepOption = new FloatOption("augmentationStep", 'a',
            "Step by which the mutation factor decreases and the crossover rate increases each generation.", 0.025, 0.0, 1.0);

    public MultiChoiceOption metricOption = new MultiChoiceOption("metric", 'm',
            "Metric to optimize the model.", MetricUtils.REGRESSION_METRIC_NAMES,
            MetricUtils.REGRESSION_METRIC_DESCRIPTIONS, 0);

    public FlagOption resetModelsOption = new FlagOption("resetModels", 'r',
            "Create fresh regressors for trial points instead of warm-starting from best.");

    public FlagOption verboseOption = new FlagOption("verbose", 'v',
            "Print DE events to stdout.");

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

    // ========== INNER CLASS ==========

    protected static class PopulationEntry implements Serializable {
        Classifier model;
        BasicRegressionPerformanceEvaluator evaluator;
        ArrayList<Parameter> params;

        PopulationEntry(Classifier model, BasicRegressionPerformanceEvaluator evaluator,
                        ArrayList<Parameter> params) {
            this.model = model;
            this.evaluator = evaluator;
            this.params = params;
        }

        double getMetric(int metricIndex) {
            return MetricUtils.getRegressionScore(evaluator.getPerformanceMeasurements(), metricIndex);
        }
    }

    // ========== FIELDS ==========

    protected PopulationEntry[] population;
    protected ArrayList<Parameter> oldBestParams;

    protected ConfigurationSpace space;

    /** Boundary through which every learner is configured. */
    protected LearnerConfigurator configurator;

    protected long instanceCount;
    protected int evaluationInstances;
    protected boolean converged;

    protected double mutationFactor;
    protected double crossoverRate;

    protected boolean[] optimizeMask;

    /** Detector on the incumbent's error; {@code null} unless drift detection is on. */
    protected ChangeDetector driftDetector;

    /** Cumulative number of drifts signalled over the run; not reset by a restart. */
    protected long driftsDetected;

    /**
     * Pool training the population concurrently; {@code null} when running
     * single-threaded. Transient because an executor cannot be serialized, so it
     * is (re)created on demand by {@link #initExecutor()}.
     */
    protected transient ExecutorService executor;

    @Override
    public void setOptimizableParameters(boolean[] mask) { this.optimizeMask = mask; }

    // ========== LIFECYCLE ==========

    @Override
    public boolean isRandomizable() {
        return true;
    }

    /**
     * Creates the training pool on first use, and again after deserialization.
     * A pool larger than the batch trained per instance would leave threads idle,
     * so the requested job count is capped at the population size.
     */
    protected void initExecutor() {
        if (this.executor != null) return;
        int numberOfJobs = this.numberOfJobsOption.getValue() == -1
                ? Runtime.getRuntime().availableProcessors()
                : this.numberOfJobsOption.getValue();
        numberOfJobs = Math.min(numberOfJobs, this.populationSizeOption.getValue());
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
    public void resetLearningImpl() {
        cleanThreads(); // shut down any pool from a previous reset before creating a new one
        converged = false;
        instanceCount = 0;
        evaluationInstances = 0;
        oldBestParams = null;
        mutationFactor = initialMutationFactorOption.getValue();
        crossoverRate = initialCrossoverRateOption.getValue();

        driftDetector = driftDetectionOption.isSet()
                ? ((ChangeDetector) getPreparedClassOption(driftDetectorOption)).copy()
                : null;

        setConfigurations();
        initializePopulation();

    }


    public void setConfigurations() {
        try {
            this.space = ConfigurationSpace.resolve(configurationFileOption.getValue(), searchSpaceOption.getValue());
            this.configurator = new LearnerConfigurator(this.space);
            this.configurator.validate();
        } catch (Exception e) {
            throw new IllegalStateException("Could not set up " + getClass().getSimpleName()
                    + " from " + ConfigurationSpace.describeSource(configurationFileOption.getValue(), searchSpaceOption.getValue()) + ": " + e.getMessage(), e);
        }
    }

    @Override
    public void checkParameterChange() {

    }

    @Override
    public void swapClassifiers(int bestPerforming) {

    }

    @Override
    public void deepCopyList(int bestPerforming) {

    }

    @Override
    public void changeStateParameter(Parameter parameter, int index) {

    }

    private void initializePopulation() {
        int n = populationSizeOption.getValue();
        population = new PopulationEntry[n];
        for (int i = 0; i < n; i++) {
            ArrayList<Parameter> params = createRandomParams();
            Classifier model = createModelWithParams(params);
            population[i] = new PopulationEntry(model, newEvaluator(), params);
        }
        if (verboseOption.isSet())
            System.out.println("MESSPTRegressor: Initialized population with " + n + " models");
    }

    private ArrayList<Parameter> createRandomParams() {
        ArrayList<Parameter> params = new ArrayList<>();
        for (ParameterSpec spec : this.space.parameters) {
            String name = spec.name;
            switch (spec.type) {
                case ParameterSpec.TYPE_INT: {
                    int[] range = {(int) spec.range[0], (int) spec.range[1]};
                    int value = classifierRandom.nextInt(range[1] - range[0] + 1) + range[0];
                    params.add(new IntParameter(name, value, range, new Random(classifierRandom.nextLong())));
                    break;
                }
                case ParameterSpec.TYPE_DOUBLE: {
                    double[] range = {spec.range[0], spec.range[1]};
                    double value = range[0] + classifierRandom.nextDouble() * (range[1] - range[0]);
                    params.add(new DoubleParameter(name, value, range, new Random(classifierRandom.nextLong())));
                    break;
                }
                case ParameterSpec.TYPE_CATEGORICAL: {
                    String[] values = spec.values;
                    int active = classifierRandom.nextInt(values.length);
                    params.add(new CategoricalParameter(name, values, active, new Random(classifierRandom.nextLong())));
                    break;
                }
            }
        }
        return params;
    }

    /** A model built from scratch at {@code params}. */
    private Classifier createModelWithParams(ArrayList<Parameter> params) {
        return this.configurator.instantiate(params, this.classifierRandom.nextInt());
    }

    /**
     * Reconfigure a warm-started model in place, returning the model to use.
     * Falls back to rebuilding when the space contains a hyperparameter the
     * learner only reads at construction time, which by definition cannot take
     * effect on a model that is already training.
     */
    private Classifier applyParamsToModel(Classifier model, ArrayList<Parameter> params) {
        if (!this.configurator.canApplyLive()) {
            return this.configurator.instantiate(params, this.classifierRandom.nextInt());
        }
        this.configurator.applyLive(model, params);
        return model;
    }

    private BasicRegressionPerformanceEvaluator newEvaluator() {
        return new BasicRegressionPerformanceEvaluator();
    }

    // ========== TRAINING ==========

    @Override
    public double[] getVotesForInstance(Instance inst) {
        return population[0].model.getVotesForInstance(inst);
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {
        double[] incumbentVotes = driftDetector != null ? getVotesForInstance(inst) : null;
        initExecutor();
        instanceCount++;
        evaluationInstances++;
        InstanceExample example = new InstanceExample(inst);

        trainNotConverged(inst, example);

        if (incumbentVotes != null) checkDrift(inst, incumbentVotes);
    }

    private void trainConverged(Instance inst, InstanceExample example) {
        PopulationEntry best = population[0];
        double[] votes = best.model.getVotesForInstance(inst);
        best.evaluator.addResult(example, votes);
        best.model.trainOnInstance(inst);
    }

    private void trainNotConverged(Instance inst, InstanceExample example) {
        Collection<TrainingRunnable> trainers = this.executor == null
                ? null : new ArrayList<TrainingRunnable>();

        for (PopulationEntry entry : population) {
            double[] votes = entry.model.getVotesForInstance(inst);
            entry.evaluator.addResult(example, votes);

            // Every individual owns its model, hence no synchronization here.
            if (trainers == null) entry.model.trainOnInstance(inst);
            else trainers.add(new TrainingRunnable(entry.model, inst));
        }

        if (trainers != null) {
            try {
                this.executor.invokeAll(trainers);
            } catch (InterruptedException ex) {
                throw new RuntimeException("Could not call invokeAll() on training threads.");
            }
        }

        if (evaluationInstances >= periodicityOption.getValue()) {
            evaluationInstances = 0;
            updatePopulation();

            if (checkConvergence()) {
                if (verboseOption.isSet())
                    System.out.println("MESSPTRegressor: Converged at instance " + instanceCount);
                converged = true;
            } else {
                mutationFactor = Math.max(0.0, mutationFactor - augmentationStepOption.getValue());
                crossoverRate = Math.min(1.0, crossoverRate + augmentationStepOption.getValue());
            }
        }
    }

    // ========== DE UPDATE ==========

    private void updatePopulation() {
        sortPopulation();
        oldBestParams = cloneParams(population[0].params);

        int n = population.length;
        PopulationEntry[] newPopulation = new PopulationEntry[n];
        newPopulation[0] = population[0]; // elitism: keep best

        for (int i = 1; i < n; i++) {
            ArrayList<Parameter> trialParams = deMutationBest1(i);
            ArrayList<Parameter> newParams = crossover(population[i].params, trialParams);

            // Freeze the hyperparameters outside the optimized subset to the elite (best) individual.
            HPOMethod.freezeToIncumbent(this.optimizeMask, newParams, population[0].params);

            PopulationEntry newEntry;
            if (resetModelsOption.isSet()) {
                Classifier model = createModelWithParams(newParams);
                newEntry = new PopulationEntry(model, newEvaluator(), newParams);
            } else {
                Classifier model = population[0].model.copy();
                LearnerConfigurator.reseedCopy(model, this.classifierRandom.nextInt());
                model = applyParamsToModel(model, newParams);
                newEntry = new PopulationEntry(model, newEvaluator(), newParams);
            }
            newPopulation[i] = newEntry;
        }

        population = newPopulation;
    }

    private void sortPopulation() {
        int metric = metricOption.getChosenIndex();
        Arrays.sort(population, (a, b) -> Double.compare(b.getMetric(metric), a.getMetric(metric)));
    }

    // DE/best/1 mutation: v = best + mutationFactor * (r1 - r2)
    private ArrayList<Parameter> deMutationBest1(int targetIndex) {
        List<Integer> pool = new ArrayList<>();
        for (int i = 1; i < population.length; i++) {
            if (i != targetIndex) pool.add(i);
        }
        Collections.shuffle(pool, classifierRandom);
        int r1Idx = pool.get(0);
        int r2Idx = pool.get(1);

        ArrayList<Parameter> best = population[0].params;
        ArrayList<Parameter> r1 = population[r1Idx].params;
        ArrayList<Parameter> r2 = population[r2Idx].params;

        ArrayList<Parameter> trial = new ArrayList<>();
        for (int i = 0; i < best.size(); i++) {
            Parameter pb = best.get(i);
            Parameter p1 = r1.get(i);
            Parameter p2 = r2.get(i);
            switch (pb.type) {
                case Parameter.TYPE_INT: {
                    int[] range = ((IntParameter) pb).range;
                    double mutated = ((IntParameter) pb).value + mutationFactor * (((IntParameter) p1).value - ((IntParameter) p2).value);
                    int val = clampInt((int) Math.round(mutated), range[0], range[1]);
                    trial.add(new IntParameter(pb.name, val, range.clone(), new Random(classifierRandom.nextLong())));
                    break;
                }
                case Parameter.TYPE_DOUBLE: {
                    double[] range = ((DoubleParameter) pb).range;
                    double mutated = ((DoubleParameter) pb).value + mutationFactor * (((DoubleParameter) p1).value - ((DoubleParameter) p2).value);
                    double val = clampDouble(mutated, range[0], range[1]);
                    trial.add(new DoubleParameter(pb.name, val, range.clone(), new Random(classifierRandom.nextLong())));
                    break;
                }
                case Parameter.TYPE_CATEGORICAL: {
                    CategoricalParameter cp = (CategoricalParameter) pb;
                    int active = classifierRandom.nextInt(cp.values.length);
                    trial.add(new CategoricalParameter(pb.name, cp.values.clone(), active, new Random(classifierRandom.nextLong())));
                    break;
                }
            }
        }
        return trial;
    }

    private ArrayList<Parameter> crossover(ArrayList<Parameter> target, ArrayList<Parameter> trial) {
        ArrayList<Parameter> result = new ArrayList<>();
        for (int i = 0; i < target.size(); i++) {
            Parameter pt = target.get(i);
            Parameter pv = trial.get(i);
            boolean useTrial = classifierRandom.nextDouble() < crossoverRate;
            switch (pt.type) {
                case Parameter.TYPE_INT: {
                    IntParameter ip = (IntParameter) (useTrial ? pv : pt);
                    result.add(new IntParameter(pt.name, ip.value, ((IntParameter) pt).range.clone(), new Random(classifierRandom.nextLong())));
                    break;
                }
                case Parameter.TYPE_DOUBLE: {
                    DoubleParameter dp = (DoubleParameter) (useTrial ? pv : pt);
                    result.add(new DoubleParameter(pt.name, dp.value, ((DoubleParameter) pt).range.clone(), new Random(classifierRandom.nextLong())));
                    break;
                }
                case Parameter.TYPE_CATEGORICAL: {
                    CategoricalParameter ct = (CategoricalParameter) pt;
                    CategoricalParameter cv = (CategoricalParameter) pv;
                    int active = useTrial ? cv.active : ct.active;
                    result.add(new CategoricalParameter(pt.name, ct.values.clone(), active, new Random(classifierRandom.nextLong())));
                    break;
                }
            }
        }
        return result;
    }

    private boolean checkConvergence() {
        if (oldBestParams == null) return false;
        ArrayList<Parameter> current = population[0].params;
        double distSq = 0;
        for (int i = 0; i < oldBestParams.size(); i++) {
            Parameter op = oldBestParams.get(i);
            Parameter cp = current.get(i);
            if (op.type == 0) {
                double diff = ((IntParameter) op).value - ((IntParameter) cp).value;
                distSq += diff * diff;
            } else if (op.type == 1) {
                double diff = ((DoubleParameter) op).value - ((DoubleParameter) cp).value;
                distSq += diff * diff;
            }
        }
        double threshold = convergenceSphereOption.getValue();
        return distSq < threshold * threshold;
    }

    // ========== HELPERS ==========

    private ArrayList<Parameter> cloneParams(ArrayList<Parameter> source) {
        ArrayList<Parameter> copy = new ArrayList<>();
        for (Parameter p : source) {
            switch (p.type) {
                case Parameter.TYPE_INT: {
                    IntParameter ip = (IntParameter) p;
                    copy.add(new IntParameter(ip.name, ip.value, ip.range.clone(), new Random(classifierRandom.nextLong())));
                    break;
                }
                case Parameter.TYPE_DOUBLE: {
                    DoubleParameter dp = (DoubleParameter) p;
                    copy.add(new DoubleParameter(dp.name, dp.value, dp.range.clone(), new Random(classifierRandom.nextLong())));
                    break;
                }
                case Parameter.TYPE_CATEGORICAL: {
                    CategoricalParameter cp = (CategoricalParameter) p;
                    copy.add(new CategoricalParameter(cp.name, cp.values.clone(), cp.active, new Random(classifierRandom.nextLong())));
                    break;
                }
            }
        }
        return copy;
    }

    private int clampInt(int val, int min, int max) {
        return Math.max(min, Math.min(max, val));
    }

    private double clampDouble(double val, double min, double max) {
        return Math.max(min, Math.min(max, val));
    }



    // ========== MOA INTERFACE ==========

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        ArrayList<Measurement> measurements = new ArrayList<>();
        measurements.add(new Measurement("driftsDetected", driftsDetected));
        for (Measurement m : population[0].model.getModelMeasurements())
            measurements.add(m);
        measurements.add(new Measurement("mutationFactor", mutationFactor));
        measurements.add(new Measurement("crossoverRate", crossoverRate));
        for (Parameter p : population[0].params) {
            switch (p.type) {
                case Parameter.TYPE_INT:
                    measurements.add(new Measurement(p.name, ((IntParameter) p).value));
                    break;
                case Parameter.TYPE_DOUBLE:
                    measurements.add(new Measurement(p.name, ((DoubleParameter) p).value));
                    break;
                case Parameter.TYPE_CATEGORICAL:
                    measurements.add(new Measurement(p.name, ((CategoricalParameter) p).active));
                    break;
            }
        }
        return measurements.toArray(new Measurement[0]);
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {
    }

    @Override
    public long measureByteSize() {
        long size = SizeOf.sizeOf(this);
        for (PopulationEntry entry : population)
            size += entry.model.measureByteSize();
        return size;
    }

    @Override
    public double getCandidateScore(int i) {
        return population[i].getMetric(metricOption.getChosenIndex());
    }

    @Override
    public double getClassifierScore() {
        return population[0].getMetric(metricOption.getChosenIndex());
    }

    @Override
    public int getNumberOfCandidates() { return populationSizeOption.getValue(); }

    @Override
    public long getStatesEvaluatedCount() { return instanceCount / periodicityOption.getValue(); }

    @Override
    public int getEvaluationInstancesCount() { return evaluationInstances; }

    @Override
    public int getPeriodicity() { return periodicityOption.getValue(); }

    @Override
    public Classifier getMainClassifier() { return population[0].model; }

    @Override
    public ArrayList<Parameter> getReferenceParameters() {
        return (population != null && population.length > 0) ? population[0].params : null;
    }

    @Override
    public String getConfigurationFile() { return this.configurationFileOption.getValue();}


    @Override
    public ArrayList<ArrayList<Parameter>> getCandidateParameters() {
        ArrayList<ArrayList<Parameter>> list = new ArrayList<>();
        if (population != null) {
            for (PopulationEntry e : population) {
                list.add(e.params);
            }
        }
        return list;
    }


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
            if (verboseOption.isSet())
                System.out.println("MESSPT: Drift detected at instance " + instanceCount
                        + ", reinitializing");
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
