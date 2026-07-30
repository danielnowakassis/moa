/*
 *    SSPTRegressor.java
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
 * Regression counterpart of {@link SSPTClassifier}: a Nelder-Mead simplex over
 * the streaming hyperparameter space, scored with the regression metrics of
 * {@link MetricUtils}.
 *
 * <p>See details in:<br> Bruno Veloso, Joao Gama, Benedita Malheiro, Joao
 * Vinagre. Hyperparameter self-tuning for data streams. In Information Fusion,
 * 76:75-86, DOI: 10.1016/j.inffus.2021.04.011, Elsevier, 2021.</p>
 *
 * @author Daniel Nowak Assis (daniel dot nowak-assis at lip6 dot fr)
 */
public class SSPTRegressor extends AbstractClassifier implements Regressor,
        CapabilitiesHandler, Serializable, HPOMethod {

    public FileOption configurationFileOption = new FileOption("configurationFile", 'f',
            "Search space in JSON format.", null, ".json", false);

    public IntOption gracePeriodOption = new IntOption("gracePeriod", 'g',
            "Number of instances between simplex updates.", 1000, 1, Integer.MAX_VALUE);

    public FloatOption convergenceSphereOption = new FloatOption("convergenceSphere", 'c',
            "Convergence threshold: squared distance between centroids.", 0.001, 0.0, Double.MAX_VALUE);

    public MultiChoiceOption metricOption = new MultiChoiceOption("metric", 'm',
            "Metric to optimize the model.", MetricUtils.REGRESSION_METRIC_NAMES,
            MetricUtils.REGRESSION_METRIC_DESCRIPTIONS, 0);

    public FlagOption resetModelsOption = new FlagOption("resetModels", 'r',
            "Create fresh classifiers for expanded points instead of warm-starting from best.");

    public FlagOption verboseOption = new FlagOption("verbose", 'v',
            "Print simplex events to stdout.");

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

    /** Largest batch trained per instance: the 3 vertices plus the 6 expansion points. */
    protected static final int MAX_MODELS_PER_INSTANCE = 9;

    // ========== INNER CLASS ==========

    protected static class SimplexEntry implements Serializable {
        Classifier model;
        BasicRegressionPerformanceEvaluator evaluator;
        ArrayList<Parameter> params;
        long instancesSeen;

        SimplexEntry(Classifier model, BasicRegressionPerformanceEvaluator evaluator,
                     ArrayList<Parameter> params) {
            this.model = model;
            this.evaluator = evaluator;
            this.params = params;
            this.instancesSeen = 0;
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

    protected SimplexEntry[] simplex;
    protected HashMap<String, SimplexEntry> expanded;

    protected ArrayList<Parameter> lastCentroid;

    protected ConfigurationSpace space;

    /** Boundary through which every learner is configured. */
    protected LearnerConfigurator configurator;

    protected long instanceCount;
    protected int evaluationInstances;
    protected boolean converged;

    protected boolean[] optimizeMask;

    /** Detector on the incumbent's error; {@code null} unless drift detection is on. */
    protected ChangeDetector driftDetector;

    /** Cumulative number of drifts signalled over the run; not reset by a restart. */
    protected long driftsDetected;

    /**
     * Pool training the simplex vertices and expansion points concurrently;
     * {@code null} when running single-threaded. Transient because an executor
     * cannot be serialized, so it is (re)created by {@link #initExecutor()}.
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
     * so the requested job count is capped at {@link #MAX_MODELS_PER_INSTANCE}.
     */
    protected void initExecutor() {
        if (this.executor != null) return;
        int numberOfJobs = this.numberOfJobsOption.getValue() == -1
                ? Runtime.getRuntime().availableProcessors()
                : this.numberOfJobsOption.getValue();
        numberOfJobs = Math.min(numberOfJobs, MAX_MODELS_PER_INSTANCE);
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
        lastCentroid = null;
        expanded = null;

        driftDetector = driftDetectionOption.isSet()
                ? ((ChangeDetector) getPreparedClassOption(driftDetectorOption)).copy()
                : null;

        setConfigurations();
        initializeSimplex();

    }

    @Override
    public void setConfigurations() {
        try {
            this.space = ConfigurationSpace.fromFile(configurationFileOption.getValue());
            this.configurator = new LearnerConfigurator(this.space);
            this.configurator.validate();
        } catch (Exception e) {
            throw new IllegalStateException("Could not set up " + getClass().getSimpleName()
                    + " from \"" + configurationFileOption.getValue() + "\": " + e.getMessage(), e);
        }
    }


    @Override
    public void checkParameterChange() {}

    @Override
    public void swapClassifiers(int bestPerforming) {}

    @Override
    public void deepCopyList(int bestPerforming) {}

    @Override
    public void changeStateParameter(Parameter parameter, int index) {}

    private void initializeSimplex() {
        simplex = new SimplexEntry[3];
        for (int i = 0; i < 3; i++) {
            ArrayList<Parameter> params = createRandomParams();
            Classifier model = createModelWithParams(params);
            simplex[i] = new SimplexEntry(model, newEvaluator(), params);
        }
        if (verboseOption.isSet())
            System.out.println("SSPT: Initialized simplex with 3 models");
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
        return simplex[0].model.getVotesForInstance(inst);
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
        SimplexEntry best = simplex[0];
        double[] votes = best.model.getVotesForInstance(inst);
        best.addResult(example, votes);
        best.model.trainOnInstance(inst);
    }

    private void trainNotConverged(Instance inst, InstanceExample example) {

        // Vertices and expansion points form a single batch: two barriers per
        // instance would halve the gain on such a small number of models.
        Collection<TrainingRunnable> trainers = this.executor == null
                ? null : new ArrayList<TrainingRunnable>();

        for (SimplexEntry entry : simplex) {
            double[] votes = entry.model.getVotesForInstance(inst);
            entry.addResult(example, votes);

            if (trainers == null) entry.model.trainOnInstance(inst);
            else trainers.add(new TrainingRunnable(entry.model, inst));
        }

        if (expanded != null) {
            for (SimplexEntry entry : expanded.values()) {
                double[] votes = entry.model.getVotesForInstance(inst);
                entry.addResult(example, votes);

                if (trainers == null) entry.model.trainOnInstance(inst);
                else trainers.add(new TrainingRunnable(entry.model, inst));
            }
        }

        if (trainers != null) {
            try {
                this.executor.invokeAll(trainers);
            } catch (InterruptedException ex) {
                throw new RuntimeException("Could not call invokeAll() on training threads.");
            }
        }

        if (evaluationInstances >= gracePeriodOption.getValue()) {
            evaluationInstances = 0;
            updateSimplex();
            if (checkConvergence()) {
                if (verboseOption.isSet())
                    System.out.println("SSPT: Converged at instance " + instanceCount);
                converged = true;
            }
        }
    }

    // ========== SIMPLEX UPDATE ==========

    private void updateSimplex() {
        sortSimplex();
        lastCentroid = computeCentroid();

        if (expanded == null || expanded.isEmpty()) {
            expanded = createExpanded();
        }

        applyNelderMeadOperators();
        expanded = null;
    }

    private void sortSimplex() {
        int metric = metricOption.getChosenIndex();
        // Insertion sort (3 elements, descending by metric)
        for (int i = 1; i < 3; i++) {
            SimplexEntry key = simplex[i];
            int j = i - 1;
            while (j >= 0 && simplex[j].getMetric(metric) < key.getMetric(metric)) {
                simplex[j + 1] = simplex[j];
                j--;
            }
            simplex[j + 1] = key;
        }
    }

    private ArrayList<Parameter> computeCentroid() {
        ArrayList<Parameter> centroid = cloneParams(simplex[0].params);
        for (int i = 0; i < centroid.size(); i++) {
            Parameter p = centroid.get(i);
            if (p.type == 0) {
                double sum = 0;
                for (SimplexEntry e : simplex) sum += ((IntParameter) e.params.get(i)).value;
                int val = clampInt((int) Math.round(sum / 3.0),
                        ((IntParameter) p).range[0], ((IntParameter) p).range[1]);
                ((IntParameter) p).value = val;
            } else if (p.type == 1) {
                double sum = 0;
                for (SimplexEntry e : simplex) sum += ((DoubleParameter) e.params.get(i)).value;
                double val = clampDouble(sum / 3.0,
                        ((DoubleParameter) p).range[0], ((DoubleParameter) p).range[1]);
                ((DoubleParameter) p).value = val;
            } else {
                // Mode for categorical
                Map<Integer, Integer> counts = new HashMap<>();
                for (SimplexEntry e : simplex)
                    counts.merge(((CategoricalParameter) e.params.get(i)).active, 1, Integer::sum);
                int mode = Collections.max(counts.entrySet(), Map.Entry.comparingByValue()).getKey();
                ((CategoricalParameter) p).active = mode;
            }
        }
        return centroid;
    }

    private boolean checkConvergence() {
        if (lastCentroid == null) return false;
        ArrayList<Parameter> current = computeCentroid();
        double distSq = 0;
        for (int i = 0; i < lastCentroid.size(); i++) {
            Parameter lp = lastCentroid.get(i);
            Parameter cp = current.get(i);
            if (lp.type == 0) {
                double diff = ((IntParameter) lp).value - ((IntParameter) cp).value;
                distSq += diff * diff;
            } else if (lp.type == 1) {
                double diff = ((DoubleParameter) lp).value - ((DoubleParameter) cp).value;
                distSq += diff * diff;
            }
        }
        double threshold = convergenceSphereOption.getValue();
        return distSq < threshold * threshold;
    }

    // ========== NELDER-MEAD ==========

    private interface DoubleOp {
        double apply(double a, double b);
    }

    private HashMap<String, SimplexEntry> createExpanded() {
        HashMap<String, SimplexEntry> exp = new HashMap<>();

        ArrayList<Parameter> bestP  = simplex[0].params;
        ArrayList<Parameter> goodP  = simplex[1].params;
        ArrayList<Parameter> worstP = simplex[2].params;

        ArrayList<Parameter> midP   = combine(bestP,  goodP,  (a, b) -> (a + b) / 2.0);
        ArrayList<Parameter> reflP  = combine(midP,   worstP, (a, b) -> 2 * a - b);
        ArrayList<Parameter> expP   = combine(reflP,  midP,   (a, b) -> 2 * a - b);
        ArrayList<Parameter> shrP   = combine(bestP,  worstP, (a, b) -> (a + b) / 2.0);
        ArrayList<Parameter> cont1P = combine(midP,   worstP, (a, b) -> (a + b) / 2.0);
        ArrayList<Parameter> cont2P = combine(midP,   reflP,  (a, b) -> (a + b) / 2.0);

        String[] names = {"midpoint", "reflection", "expansion", "shrink", "contraction1", "contraction2"};
        @SuppressWarnings("unchecked")
        ArrayList<Parameter>[] paramSets = new ArrayList[]{midP, reflP, expP, shrP, cont1P, cont2P};

        // Freeze the hyperparameters outside the optimized subset to the best
        // simplex vertex before any model is built or evaluated.
        for (ArrayList<Parameter> ps : paramSets)
            HPOMethod.freezeToIncumbent(this.optimizeMask, ps, simplex[0].params);

        for (int i = 0; i < names.length; i++) {
            SimplexEntry entry;
            if (resetModelsOption.isSet()) {
                Classifier model = createModelWithParams(paramSets[i]);
                entry = new SimplexEntry(model, newEvaluator(), paramSets[i]);
            } else {
                // Warm-start: copy best model's learned state, change parameters
                Classifier model = simplex[0].model.copy();
                LearnerConfigurator.reseedCopy(model, this.classifierRandom.nextInt());
                model = applyParamsToModel(model, paramSets[i]);
                entry = new SimplexEntry(model, newEvaluator(), paramSets[i]);
            }
            exp.put(names[i], entry);
        }
        return exp;
    }

    private void applyNelderMeadOperators() {
        int metric = metricOption.getChosenIndex();
        SimplexEntry b   = simplex[0];
        SimplexEntry g   = simplex[1];
        SimplexEntry w   = simplex[2];
        SimplexEntry r   = expanded.get("reflection");
        SimplexEntry c1  = expanded.get("contraction1");
        SimplexEntry c2  = expanded.get("contraction2");
        SimplexEntry e   = expanded.get("expansion");
        SimplexEntry s   = expanded.get("shrink");
        SimplexEntry mid = expanded.get("midpoint");

        SimplexEntry contraction = c1.getMetric(metric) > c2.getMetric(metric) ? c1 : c2;

        if (r.getMetric(metric) > g.getMetric(metric)) {
            if (b.getMetric(metric) > r.getMetric(metric)) {
                simplex[2] = r;
            } else {
                simplex[2] = e.getMetric(metric) > b.getMetric(metric) ? e : r;
            }
        } else {
            if (r.getMetric(metric) > w.getMetric(metric)) {
                simplex[2] = r;
            } else {
                if (contraction.getMetric(metric) > w.getMetric(metric)) {
                    simplex[2] = contraction;
                } else {
                    simplex[2] = s;
                    simplex[1] = mid;
                }
            }
        }
        sortSimplex();
    }

    // ========== HELPERS ==========

    private ArrayList<Parameter> combine(ArrayList<Parameter> p1, ArrayList<Parameter> p2, DoubleOp op) {
        ArrayList<Parameter> result = new ArrayList<>();
        for (int i = 0; i < p1.size(); i++) {
            Parameter a = p1.get(i);
            Parameter b = p2.get(i);
            switch (a.type) {
                case Parameter.TYPE_INT: {
                    int[] range = ((IntParameter) a).range;
                    double combined = op.apply(((IntParameter) a).value, ((IntParameter) b).value);
                    int val = clampInt((int) Math.round(combined), range[0], range[1]);
                    result.add(new IntParameter(a.name, val, range, new Random(classifierRandom.nextLong())));
                    break;
                }
                case Parameter.TYPE_DOUBLE: {
                    double[] range = ((DoubleParameter) a).range;
                    double val = clampDouble(
                            op.apply(((DoubleParameter) a).value, ((DoubleParameter) b).value),
                            range[0], range[1]);
                    result.add(new DoubleParameter(a.name, val, range, new Random(classifierRandom.nextLong())));
                    break;
                }
                case Parameter.TYPE_CATEGORICAL: {
                    CategoricalParameter ca = (CategoricalParameter) a;
                    CategoricalParameter cb = (CategoricalParameter) b;
                    int active = classifierRandom.nextBoolean() ? ca.active : cb.active;
                    result.add(new CategoricalParameter(a.name, ca.values, active, new Random(classifierRandom.nextLong())));
                    break;
                }
            }
        }
        return result;
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
        for (Measurement m : simplex[0].model.getModelMeasurements())
            measurements.add(m);
        for (Parameter p : simplex[0].params) {
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
        for (SimplexEntry entry : simplex)
            size += entry.model.measureByteSize();
        return size;
    }

    @Override
    public double getCandidateScore(int i) {
        return simplex[i].getMetric(metricOption.getChosenIndex());
    }

    @Override
    public double getClassifierScore() {
        return simplex[0].getMetric(metricOption.getChosenIndex());
    }

    @Override
    public int getNumberOfCandidates() { return 3; }

    @Override
    public long getStatesEvaluatedCount() { return instanceCount / gracePeriodOption.getValue(); }

    @Override
    public int getEvaluationInstancesCount() { return evaluationInstances; }

    @Override
    public int getGracePeriod() { return gracePeriodOption.getValue(); }

    @Override
    public Classifier getMainClassifier() { return simplex[0].model; }

    @Override
    public ArrayList<Parameter> getReferenceParameters() { return simplex[0].params; }

    @Override
    public String getConfigurationFile() { return this.configurationFileOption.getValue();}


    @Override
    public ArrayList<ArrayList<Parameter>> getCandidateParameters() {
        ArrayList<ArrayList<Parameter>> list = new ArrayList<>();
        for (SimplexEntry e : simplex) list.add(e.params);
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
                System.out.println("SSPT: Drift detected at instance " + instanceCount
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
