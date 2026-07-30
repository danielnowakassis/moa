/*
 *    FSS_SPTRegressor.java
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
 * Regression counterpart of {@link FSS_SPTClassifier}: Fish School Search over
 * the streaming hyperparameter space, scored with the regression metrics of
 * {@link MetricUtils}.
 *
 * <p>See details in:<br> Bruno Veloso, Hugo Amorim Neto, Fernando Buarque,
 * Joao Gama. Fish swarm parameter self-tuning for data streams. In Data Mining
 * and Knowledge Discovery, 40(1), DOI: 10.1007/s10618-025-01174-8, Springer,
 * 2025.</p>
 *
 * @author Daniel Nowak Assis (daniel dot nowak-assis at lip6 dot fr)
 */
public class FSS_SPTRegressor extends AbstractClassifier implements Regressor,
        CapabilitiesHandler, Serializable, HPOMethod {

    // ========== OPTIONS ==========

    public FileOption configurationFileOption = new FileOption("configurationFile", 'f',
            "Search space in JSON format.", null, ".json", false);

    public StringOption searchSpaceOption = new StringOption("searchSpace", 's',
            "Search space as inline JSON. Takes precedence over configurationFile when set,"
            + " so that a caller holding the space in memory need not write a file.", "");

    public IntOption gracePeriodOption = new IntOption("gracePeriod", 'g',
            "Number of instances between FSS school updates.", 1000, 2, Integer.MAX_VALUE);

    public MultiChoiceOption metricOption = new MultiChoiceOption("metric", 'm',
            "Metric to optimize the model.", MetricUtils.REGRESSION_METRIC_NAMES,
            MetricUtils.REGRESSION_METRIC_DESCRIPTIONS, 0);

    public IntOption numEstimatorsOption = new IntOption("numEstimators", 'n',
            "Number of fish (candidate models) in the school.", 10, 2, Integer.MAX_VALUE);

    public FloatOption initialStepOption = new FloatOption("initialStep", 'a',
            "Initial individual movement step size (fraction of parameter range).", 0.1, 0.0, 1.0);

    public FloatOption finalStepOption = new FloatOption("finalStep", 'e',
            "Final individual movement step size after numIterations updates.", 0.01, 0.0, 1.0);

    public IntOption numIterationsOption = new IntOption("numIterations", 't',
            "Number of step-decay iterations.", 100, 1, Integer.MAX_VALUE);

    public FlagOption resetModelsOption = new FlagOption("resetModels", 'r',
            "Create fresh models after each movement instead of warm-starting from best.");

    public FlagOption verboseOption = new FlagOption("verbose", 'v',
            "Print FSS events to stdout.");

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

    protected static class FishEntry implements Serializable {
        Classifier model;
        BasicRegressionPerformanceEvaluator evaluator;
        ArrayList<Parameter> params;
        long instancesSeen;
        double weight;
        double difFit;
        double oldFit;
        boolean initialized;
        ArrayList<Parameter> oldPos;  // position at first nexteval (never updated after init)
        double[] difDist;             // displacement relative to oldPos when improvement occurred

        FishEntry(Classifier model, BasicRegressionPerformanceEvaluator evaluator,
                  ArrayList<Parameter> params) {
            this.model = model;
            this.evaluator = evaluator;
            this.params = params;
            this.instancesSeen = 0;
            this.weight = 0.1;
            this.difFit = 0.0;
            this.oldFit = Double.NEGATIVE_INFINITY;
            this.initialized = false;
            this.oldPos = null;
            this.difDist = null;
        }

        double getMetric(int metricIndex) {
            if (instancesSeen == 0) return Double.NEGATIVE_INFINITY;
            return MetricUtils.getRegressionScore(evaluator.getPerformanceMeasurements(), metricIndex);
        }

        void addResult(InstanceExample example, double[] votes) {
            evaluator.addResult(example, votes);
            instancesSeen++;
        }
    }

    // ========== FIELDS ==========

    protected FishEntry[] school;
    protected ConfigurationSpace space;

    /** Boundary through which every learner is configured. */
    protected LearnerConfigurator configurator;
    protected double currentStep;
    protected long instanceCount;
    protected int evaluationInstances;

    /** Detector on the incumbent's error; {@code null} unless drift detection is on. */
    protected ChangeDetector driftDetector;

    /** Cumulative number of drifts signalled over the run; not reset by a restart. */
    protected long driftsDetected;

    protected boolean[] optimizeMask;

    /**
     * Pool training the school concurrently; {@code null} when running
     * single-threaded. Transient because an executor cannot be serialized, so it
     * is (re)created on demand by {@link #initExecutor()}.
     */
    protected transient ExecutorService executor;

    @Override
    public void setOptimizableParameters(boolean[] mask) { this.optimizeMask = mask; }

    // ========== LIFECYCLE ==========

    @Override
    public boolean isRandomizable() { return true; }

    /**
     * Creates the training pool on first use, and again after deserialization.
     * A pool larger than the batch trained per instance would leave threads idle,
     * so the requested job count is capped at the school size.
     */
    protected void initExecutor() {
        if (this.executor != null) return;
        int numberOfJobs = this.numberOfJobsOption.getValue() == -1
                ? Runtime.getRuntime().availableProcessors()
                : this.numberOfJobsOption.getValue();
        numberOfJobs = Math.min(numberOfJobs, this.numEstimatorsOption.getValue());
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
        driftDetector = driftDetectionOption.isSet()
                ? ((ChangeDetector) getPreparedClassOption(driftDetectorOption)).copy()
                : null;
        instanceCount = 0;
        evaluationInstances = 0;
        currentStep = initialStepOption.getValue();

        setConfigurations();
        initializeSchool();

    }

    @Override
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

    private void initializeSchool() {
        int n = numEstimatorsOption.getValue();
        school = new FishEntry[n];
        for (int i = 0; i < n; i++) {
            ArrayList<Parameter> params = createRandomParams();
            Classifier model = createModelWithParams(params);
            school[i] = new FishEntry(model, newEvaluator(), params);
        }
        if (verboseOption.isSet())
            System.out.println("FSS_SPT: Initialized school with " + n + " fish");
    }

    @Override
    public void checkParameterChange() {}

    @Override
    public void swapClassifiers(int bestPerforming) {}

    @Override
    public void deepCopyList(int bestPerforming) {}

    @Override
    public void changeStateParameter(Parameter parameter, int index) {}

    // ========== PARAM CREATION ==========

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
        return getBestFish().model.getVotesForInstance(inst);
    }

    private FishEntry getBestFish() {
        int metric = metricOption.getChosenIndex();
        FishEntry best = school[0];
        for (FishEntry f : school)
            if (f.getMetric(metric) > best.getMetric(metric)) best = f;
        return best;
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {
        double[] incumbentVotes = driftDetector != null ? getVotesForInstance(inst) : null;
        initExecutor();
        instanceCount++;
        evaluationInstances++;
        InstanceExample example = new InstanceExample(inst);

        Collection<TrainingRunnable> trainers = this.executor == null
                ? null : new ArrayList<TrainingRunnable>();

        for (FishEntry fish : school) {
            double[] votes = fish.model.getVotesForInstance(inst);
            fish.addResult(example, votes);
            // Every fish owns its model, hence no synchronization here.
            if (trainers == null) fish.model.trainOnInstance(inst);
            else trainers.add(new TrainingRunnable(fish.model, inst));
        }

        if (trainers != null) {
            try {
                this.executor.invokeAll(trainers);
            } catch (InterruptedException ex) {
                throw new RuntimeException("Could not call invokeAll() on training threads.");
            }
        }
        int halfPeriod = gracePeriodOption.getValue() / 2;

        // Phase 1 (halfway): assess improvement, apply individual movement
        if (evaluationInstances == halfPeriod) {
            sortSchool();
            individualMovementNextEval();
            individualMovement();
            rebuildModels();
        }

        // Phase 2 (full period): feeding, instinctive, volitional movements
        if (evaluationInstances >= gracePeriodOption.getValue()) {
            evaluationInstances = 0;
            sortSchool();
            double weightChange = feeding();
            instinctiveMovement();
            double[] barycenter = calculateBarycenter();
            volitionalMovement(barycenter, weightChange);
            updateStep();
            rebuildModels();
            if (verboseOption.isSet())
                System.out.printf("FSS_SPT: Updated at instance %d, step=%.4f, weightChange=%.4f%n",
                        instanceCount, currentStep, weightChange);
        }

        if (incumbentVotes != null) checkDrift(inst, incumbentVotes);
    }

    // Rebuild all fish models with their current params after position updates.
    // Warm-starts from the best fish unless resetModels is set.
    private void rebuildModels() {
        FishEntry best = getBestFish();
        for (FishEntry fish : school) {
            // Freeze the hyperparameters outside the optimized subset to the best fish.
            HPOMethod.freezeToIncumbent(this.optimizeMask, fish.params, best.params);
            if (resetModelsOption.isSet()) {
                fish.model = createModelWithParams(fish.params);
            } else {
                fish.model = best.model.copy();
                LearnerConfigurator.reseedCopy(fish.model, this.classifierRandom.nextInt());
                applyParamsToModel(fish.model, fish.params);
            }
            fish.evaluator = newEvaluator();
            fish.instancesSeen = 0;
        }
    }

    // ========== SCHOOL SORT ==========

    private void sortSchool() {
        int metric = metricOption.getChosenIndex();
        Arrays.sort(school, (a, b) -> Double.compare(b.getMetric(metric), a.getMetric(metric)));
    }

    // ========== INDIVIDUAL MOVEMENT NEXT EVAL ==========
    // Assess whether the fish improved since baseline; record displacement and fitness delta.
    // oldPos is fixed at first call (mirrors Python behaviour where old_pos is never updated).

    private void individualMovementNextEval() {
        int metric = metricOption.getChosenIndex();
        for (FishEntry fish : school) {
            double currentFitness = fish.getMetric(metric);
            double[] currentPos = paramsToDoubleArray(fish.params);

            if (!fish.initialized) {
                fish.oldFit = currentFitness;
                fish.oldPos = cloneParams(fish.params);
                fish.difDist = new double[currentPos.length];
                fish.difFit = 0.0;
                fish.initialized = true;
            } else if (currentFitness > fish.oldFit) {
                fish.difFit = currentFitness - fish.oldFit;
                fish.oldFit = currentFitness;
                double[] oldPosArr = paramsToDoubleArray(fish.oldPos);
                fish.difDist = subtractArrays(currentPos, oldPosArr);
            } else {
                fish.difFit = 0.0;
                fish.difDist = new double[currentPos.length];
            }
        }
    }

    // ========== INDIVIDUAL MOVEMENT ==========
    // Perturb each fish randomly within step * rangeWidth.

    private void individualMovement() {
        double[] rangeWidths = getRangeWidths(school[0].params);
        for (FishEntry fish : school) {
            double[] pos = paramsToDoubleArray(fish.params);
            for (int j = 0; j < pos.length; j++) {
                double direction = classifierRandom.nextDouble() * 2.0 - 1.0;
                pos[j] += currentStep * direction * rangeWidths[j];
            }
            applyDoubleArrayToParams(fish.params, pos);
            for (Parameter p : fish.params)
                if (p.type == 2 && classifierRandom.nextDouble() < currentStep) p.changeParameter();
        }
    }

    // ========== FEEDING ==========
    // Update fish weights proportional to normalised fitness improvement.
    // Returns (sum_before - sum_after): negative when school got heavier.

    private double feeding() {
        double maxDifFit = 0.0;
        for (FishEntry fish : school)
            if (fish.difFit > maxDifFit) maxDifFit = fish.difFit;

        if (maxDifFit == 0.0) return 0.0;

        double weightBefore = 0.0;
        for (FishEntry fish : school) weightBefore += fish.weight;

        for (FishEntry fish : school)
            fish.weight += fish.difFit / maxDifFit;

        double weightAfter = 0.0;
        for (FishEntry fish : school) weightAfter += fish.weight;

        return weightBefore - weightAfter;
    }

    // ========== INSTINCTIVE MOVEMENT ==========
    // Move all fish by the fitness-weighted average of individual displacements.

    private void instinctiveMovement() {
        if (school[0].difDist == null) return;
        double totalDifFit = 0.0;
        for (FishEntry fish : school) totalDifFit += fish.difFit;
        if (totalDifFit == 0.0) return;

        int n = school[0].difDist.length;
        double[] instinctiveVector = new double[n];
        for (FishEntry fish : school)
            for (int j = 0; j < n; j++)
                instinctiveVector[j] += fish.difDist[j] * fish.difFit;
        for (int j = 0; j < n; j++)
            instinctiveVector[j] /= totalDifFit;

        for (FishEntry fish : school) {
            double[] pos = paramsToDoubleArray(fish.params);
            for (int j = 0; j < n; j++) pos[j] += instinctiveVector[j];
            applyDoubleArrayToParams(fish.params, pos);
        }
    }

    // ========== BARYCENTER ==========
    // Compute the weight-averaged centre of the school in parameter space.

    private double[] calculateBarycenter() {
        int n = paramsToDoubleArray(school[0].params).length;
        double[] barycenter = new double[n];
        double totalWeight = 0.0;
        for (FishEntry fish : school) {
            double[] pos = paramsToDoubleArray(fish.params);
            totalWeight += fish.weight;
            for (int j = 0; j < n; j++) barycenter[j] += pos[j] * fish.weight;
        }
        if (totalWeight > 0.0)
            for (int j = 0; j < n; j++) barycenter[j] /= totalWeight;
        return barycenter;
    }

    // ========== VOLITIONAL MOVEMENT ==========
    // weightChange < 0 → school got heavier (improvement) → move AWAY from barycenter (explore).
    // weightChange >= 0 → no improvement → move TOWARD barycenter (converge).
    // This matches the sign convention in the original Python implementation.

    private void volitionalMovement(double[] barycenter, double weightChange) {
        double[] rangeWidths = getRangeWidths(school[0].params);
        int n = barycenter.length;
        for (FishEntry fish : school) {
            double[] pos = paramsToDoubleArray(fish.params);
            double dist = euclideanDistance(pos, barycenter);
            if (dist == 0.0) continue;
            double direction = classifierRandom.nextDouble();
            double[] newPos = new double[n];
            for (int j = 0; j < n; j++) {
                double delta = 2.0 * currentStep * direction * rangeWidths[j]
                        * (barycenter[j] - pos[j]) / dist;
                newPos[j] = (weightChange < 0) ? pos[j] - delta : pos[j] + delta;
            }
            applyDoubleArrayToParams(fish.params, newPos);
        }
    }

    // ========== STEP UPDATE ==========

    private void updateStep() {
        double decay = (initialStepOption.getValue() - finalStepOption.getValue())
                / numIterationsOption.getValue();
        currentStep = Math.max(finalStepOption.getValue(), currentStep - decay);
    }

    // ========== PARAM / ARRAY HELPERS ==========

    private double[] paramsToDoubleArray(ArrayList<Parameter> params) {
        int count = 0;
        for (Parameter p : params) if (p.type == 0 || p.type == 1) count++;
        double[] result = new double[count];
        int idx = 0;
        for (Parameter p : params) {
            if (p.type == 0) result[idx++] = ((IntParameter) p).value;
            else if (p.type == 1) result[idx++] = ((DoubleParameter) p).value;
        }
        return result;
    }

    private void applyDoubleArrayToParams(ArrayList<Parameter> params, double[] vals) {
        int idx = 0;
        for (Parameter p : params) {
            if (p.type == 0) {
                IntParameter ip = (IntParameter) p;
                ip.value = clampInt((int) Math.round(vals[idx++]), ip.range[0], ip.range[1]);
            } else if (p.type == 1) {
                DoubleParameter dp = (DoubleParameter) p;
                dp.value = clampDouble(vals[idx++], dp.range[0], dp.range[1]);
            }
        }
    }

    private double[] getRangeWidths(ArrayList<Parameter> params) {
        int count = 0;
        for (Parameter p : params) if (p.type == 0 || p.type == 1) count++;
        double[] widths = new double[count];
        int idx = 0;
        for (Parameter p : params) {
            if (p.type == 0) widths[idx++] = ((IntParameter) p).range[1] - ((IntParameter) p).range[0];
            else if (p.type == 1) widths[idx++] = ((DoubleParameter) p).range[1] - ((DoubleParameter) p).range[0];
        }
        return widths;
    }

    private double[] subtractArrays(double[] a, double[] b) {
        double[] result = new double[a.length];
        for (int i = 0; i < a.length; i++) result[i] = a[i] - b[i];
        return result;
    }

    private double euclideanDistance(double[] a, double[] b) {
        double sum = 0.0;
        for (int i = 0; i < a.length; i++) { double d = a[i] - b[i]; sum += d * d; }
        return Math.sqrt(sum);
    }

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

    private int clampInt(int val, int min, int max) { return Math.max(min, Math.min(max, val)); }
    private double clampDouble(double val, double min, double max) { return Math.max(min, Math.min(max, val)); }

    // ========== MOA INTERFACE ==========

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        ArrayList<Measurement> measurements = new ArrayList<>();
        measurements.add(new Measurement("driftsDetected", driftsDetected));
        FishEntry best = getBestFish();
        for (Measurement m : best.model.getModelMeasurements())
            measurements.add(m);
        for (Parameter p : best.params) {
            switch (p.type) {
                case Parameter.TYPE_INT: measurements.add(new Measurement(p.name, ((IntParameter) p).value)); break;
                case Parameter.TYPE_DOUBLE: measurements.add(new Measurement(p.name, ((DoubleParameter) p).value)); break;
                case Parameter.TYPE_CATEGORICAL: measurements.add(new Measurement(p.name, ((CategoricalParameter) p).active)); break;
            }
        }
        measurements.add(new Measurement("currentStep", currentStep));
        return measurements.toArray(new Measurement[0]);
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {}

    @Override
    public long measureByteSize() {
        long size = SizeOf.sizeOf(this);
        for (FishEntry fish : school) size += fish.model.measureByteSize();
        return size;
    }

    @Override
    public double getCandidateScore(int i) {
        return school[i].getMetric(metricOption.getChosenIndex());
    }

    @Override
    public double getClassifierScore() {
        return getBestFish().getMetric(metricOption.getChosenIndex());
    }

    @Override
    public int getNumberOfCandidates() { return numEstimatorsOption.getValue(); }

    @Override
    public long getStatesEvaluatedCount() { return instanceCount / gracePeriodOption.getValue(); }

    @Override
    public int getEvaluationInstancesCount() { return evaluationInstances; }

    @Override
    public int getGracePeriod() { return gracePeriodOption.getValue(); }

    @Override
    public Classifier getMainClassifier() { return getBestFish().model; }

    @Override
    public ArrayList<Parameter> getReferenceParameters() {
        return (school != null && school.length > 0) ? getBestFish().params : null;
    }

    @Override
    public String getConfigurationFile() { return this.configurationFileOption.getValue();}


    @Override
    public ArrayList<ArrayList<Parameter>> getCandidateParameters() {
        ArrayList<ArrayList<Parameter>> list = new ArrayList<>();
        if (school != null) {
            for (FishEntry f : school) {
                list.add(f.params);
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
                System.out.println("FSS: Drift detected at instance " + instanceCount
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
