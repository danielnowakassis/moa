/*
 *    BayesianStreamTunerRegressor.java
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
import com.yahoo.labs.samoa.instances.Attribute;
import com.yahoo.labs.samoa.instances.DenseInstance;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.InstancesHeader;
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
import moa.classifiers.functions.BayesianLinearRegression;
import moa.core.InstanceExample;
import moa.core.Measurement;
import moa.core.SizeOf;
import moa.evaluation.BasicRegressionPerformanceEvaluator;
import moa.options.ClassOption;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Regression counterpart of {@link BayesianStreamTunerClassifier}: Bayesian
 * optimisation over the streaming hyperparameter space, scored with the
 * regression metrics of {@link MetricUtils}.
 *
 * <p>See details in:<br> Nilesh Verma, Albert Bifet, Bernhard Pfahringer,
 * Maroua Bahri. Bayesian Stream Tuner: Dynamic Hyperparameter Optimization for
 * Real-Time Data Streams. In Proceedings of the 31st ACM SIGKDD Conference on
 * Knowledge Discovery and Data Mining V.2 (KDD '25), pages 2871-2882,
 * DOI: 10.1145/3711896.3736852, ACM, 2025.</p>
 *
 * @author Daniel Nowak Assis (daniel dot nowak-assis at lip6 dot fr)
 */
public class BayesianStreamTunerRegressor extends AbstractClassifier
        implements Regressor, CapabilitiesHandler, Serializable, HPOMethod {

    private static final long serialVersionUID = 1L;

    public FileOption configurationFileOption = new FileOption("configurationFile", 'f',
            "Search space in JSON format.", null, ".json", false);

    public StringOption searchSpaceOption = new StringOption("searchSpace", 's',
            "Search space as inline JSON. Takes precedence over configurationFile when set,"
            + " so that a caller holding the space in memory need not write a file.", "");

    public IntOption periodicityOption = new IntOption("periodicity", 'g',
            "Number of instances between model update cycles; also the number of recent"
            + " instances kept to derive the surrogate's stream statistics.",
            1000, 1, Integer.MAX_VALUE);

    public IntOption numberOfCandidatesOption = new IntOption("numberOfCandidates", 'n',
            "Number of candidate models (including default).", 10, 2, Integer.MAX_VALUE);

    // MetricUtils orients every regression metric so that larger is better, so
    // error metrics can be selected here without a separate direction flag.
    public MultiChoiceOption metricOption = new MultiChoiceOption("metric", 'm',
            "Metric to optimize the model.", MetricUtils.REGRESSION_METRIC_NAMES,
            MetricUtils.REGRESSION_METRIC_DESCRIPTIONS, 0);

    public MultiChoiceOption acquisitionFunctionOption = new MultiChoiceOption(
            "acquisitionFunction", 'q',
            "Acquisition function for Bayesian optimization.",
            new String[]{"PI", "EI", "UCB"},
            new String[]{"Probability of Improvement", "Expected Improvement", "Upper Confidence Bound"},
            0);

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

    // ---- Model ensemble ----
    protected Classifier[] candidates;

    /**
     * Pool training the candidates concurrently; {@code null} when running
     * single-threaded. Transient because an executor cannot be serialized, so it
     * is (re)created on demand by {@link #initExecutor()}.
     */
    protected transient ExecutorService executor;

    /** Boundary through which every learner is configured. */
    protected LearnerConfigurator configurator;
    protected BasicRegressionPerformanceEvaluator[] evaluators;
    protected ArrayList<ArrayList<Parameter>> candidatesParameters;
    protected int bestCandidateIndex;
    protected long instanceCount;
    protected String algorithmCLIString;

    // ---- BLR surrogate ----
    protected BayesianLinearRegression surrogate;
    protected InstancesHeader surrogateHeader;
    protected int numParameters;

    // ---- Data window (circular buffer) ----
    protected double[][] dataWindow;
    protected int windowHead;
    protected int windowCount;
    protected int windowFeatures;

    // ---- Subset optimization ----
    /** Detector on the incumbent's error; {@code null} unless drift detection is on. */
    protected ChangeDetector driftDetector;

    /** Cumulative number of drifts signalled over the run; not reset by a restart. */
    protected long driftsDetected;

    protected boolean[] optimizeMask;

    @Override
    public void setOptimizableParameters(boolean[] mask) { this.optimizeMask = mask; }

    @Override
    public boolean isRandomizable() { return true; }

    @Override
    public double[] getVotesForInstance(Instance inst) {
        return candidates[bestCandidateIndex].getVotesForInstance(inst);
    }

    /**
     * Creates the training pool on first use, and again after deserialization.
     * A pool larger than the batch trained per instance would leave threads idle,
     * so the requested job count is capped at the candidate pool size.
     */
    protected void initExecutor() {
        if (this.executor != null) return;
        int numberOfJobs = this.numberOfJobsOption.getValue() == -1
                ? Runtime.getRuntime().availableProcessors()
                : this.numberOfJobsOption.getValue();
        numberOfJobs = Math.min(numberOfJobs, this.numberOfCandidatesOption.getValue());
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
        driftDetector = driftDetectionOption.isSet()
                ? ((ChangeDetector) getPreparedClassOption(driftDetectorOption)).copy()
                : null;
        cleanThreads(); // shut down any pool from a previous reset before creating a new one
        this.candidatesParameters = new ArrayList<>();
        this.instanceCount = 0;
        this.bestCandidateIndex = 0;
        this.dataWindow = null;
        this.windowHead = 0;
        this.windowCount = 0;
        this.surrogate = null;
        this.surrogateHeader = null;
        this.setConfigurations();

    }

    public void setConfigurations() {
        try {
            ConfigurationSpace space = ConfigurationSpace.resolve(configurationFileOption.getValue(), searchSpaceOption.getValue());
            this.configurator = new LearnerConfigurator(space);
            this.configurator.validate();

            this.algorithmCLIString = space.algorithm;
            int n = numberOfCandidatesOption.getValue();

            this.candidates = new Classifier[n];
            this.evaluators = new BasicRegressionPerformanceEvaluator[n];
            for (int i = 0; i < n; i++) {
                // Built one by one rather than copied from a single base, so that
                // each candidate carries its own random seed.
                this.candidates[i] = this.configurator.newLearner(this.classifierRandom.nextInt());
                this.evaluators[i] = new BasicRegressionPerformanceEvaluator();
                this.candidatesParameters.add(new ArrayList<>());
            }

            this.numParameters = 0;
            {
                for (ParameterSpec spec : space.parameters) {
                    String pName = spec.name;
                    switch (spec.type) {
                        case ParameterSpec.TYPE_CATEGORICAL: {
                            String[] vals = spec.values;
                            int defaultActive = spec.active;

                            CategoricalParameter p0 = new CategoricalParameter(pName, vals, defaultActive, new Random(this.classifierRandom.nextLong()));
                            candidatesParameters.get(0).add(p0);
                            setParameter(candidates[0], p0);
                            for (int i = 1; i < n; i++) {
                                int rActive = classifierRandom.nextInt(vals.length);
                                CategoricalParameter pi = new CategoricalParameter(pName, vals, rActive, new Random(classifierRandom.nextLong()));
                                candidatesParameters.get(i).add(pi);
                                setParameter(candidates[i], pi);
                            }
                            // Categoricals contribute one dimension to the parameter
                            // vector (their active index), so they must be counted in
                            // the BLR feature dimension exactly like int/double params.
                            this.numParameters++;
                            break;
                        }
                        case ParameterSpec.TYPE_INT: {
                            int[] range = {(int) spec.range[0], (int) spec.range[1]};
                            int defaultVal = (int) spec.value;

                            IntParameter p0 = new IntParameter(pName, defaultVal, range, new Random(this.classifierRandom.nextLong()));
                            candidatesParameters.get(0).add(p0);
                            setParameter(candidates[0], p0);
                            for (int i = 1; i < n; i++) {
                                int rVal = classifierRandom.nextInt(range[1] - range[0] + 1) + range[0];
                                IntParameter pi = new IntParameter(pName, rVal, range, new Random(classifierRandom.nextLong()));
                                candidatesParameters.get(i).add(pi);
                                setParameter(candidates[i], pi);
                            }
                            this.numParameters++;
                            break;
                        }
                        case ParameterSpec.TYPE_DOUBLE: {
                            double[] range = {spec.range[0], spec.range[1]};
                            double defaultVal = spec.value;

                            DoubleParameter p0 = new DoubleParameter(pName, defaultVal, range, new Random(this.classifierRandom.nextLong()));
                            candidatesParameters.get(0).add(p0);
                            setParameter(candidates[0], p0);
                            for (int i = 1; i < n; i++) {
                                double rVal = range[0] + (range[1] - range[0]) * classifierRandom.nextDouble();
                                DoubleParameter pi = new DoubleParameter(pName, rVal, range, new Random(classifierRandom.nextLong()));
                                candidatesParameters.get(i).add(pi);
                                setParameter(candidates[i], pi);
                            }
                            this.numParameters++;
                            break;
                        }
                    }
                }
            }

            this.surrogateHeader = createSurrogateHeader(this.numParameters + 10);
            this.surrogate = new BayesianLinearRegression();
            this.surrogate.setModelContext(this.surrogateHeader);
            this.surrogate.resetLearning();

        } catch (Exception e) {
            // Fail fast: a swallowed error here leaves candidates/surrogate null and
            // surfaces later as a confusing NPE far from the real cause.
            throw new RuntimeException(
                    "Failed to load tuner configuration from " + ConfigurationSpace.describeSource(configurationFileOption.getValue(), searchSpaceOption.getValue()), e);
        }
    }

    /**
     * Apply one drawn hyperparameter to a candidate through its MOA options.
     * The parameter object is the same one recorded in {@code candidatesParameters},
     * so the model and the surrogate's feature vector cannot drift apart.
     */
    private void setParameter(Classifier c, Parameter p) {
        this.configurator.applyLive(c, p);
    }

    // ---- BLR instance helpers ----

    private InstancesHeader createSurrogateHeader(int numFeatures) {
        ArrayList<Attribute> atts = new ArrayList<>();
        for (int i = 0; i < numFeatures; i++) atts.add(new Attribute("f" + i));
        atts.add(new Attribute("performance"));
        Instances dataset = new Instances("BLR", atts, 0);
        dataset.setClassIndex(numFeatures);
        return new InstancesHeader(dataset);
    }

    private Instance createSurrogateInstance(double[] features, double performance) {
        double[] vals = new double[features.length + 1];
        System.arraycopy(features, 0, vals, 0, features.length);
        vals[features.length] = performance;
        DenseInstance inst = new DenseInstance(1.0, vals);
        inst.setDataset(surrogateHeader);
        return inst;
    }

    // ---- Data window ----

    private void initDataWindow(int numFeatures) {
        this.windowFeatures = numFeatures;
        this.dataWindow = new double[periodicityOption.getValue()][numFeatures];
        this.windowHead = 0;
        this.windowCount = 0;
    }

    private void addToDataWindow(Instance inst) {
        int nf = inst.numAttributes() - 1;
        if (dataWindow == null) initDataWindow(nf);
        double[] row = new double[nf];
        int idx = 0;
        for (int i = 0; i < inst.numAttributes(); i++) {
            if (i != inst.classIndex()) row[idx++] = inst.value(i);
        }
        dataWindow[windowHead] = row;
        windowHead = (windowHead + 1) % dataWindow.length;
        if (windowCount < dataWindow.length) windowCount++;
    }

    private double[] extractStatFeatures() {
        if (windowCount == 0 || dataWindow == null) return new double[10];

        double[] flat = new double[windowCount * windowFeatures];
        int idx = 0;
        for (int i = 0; i < windowCount; i++) {
            int pos = (windowHead - windowCount + i + dataWindow.length) % dataWindow.length;
            System.arraycopy(dataWindow[pos], 0, flat, idx, windowFeatures);
            idx += windowFeatures;
        }
        Arrays.sort(flat);

        double sum = 0, sum2 = 0;
        for (double v : flat) { sum += v; sum2 += v * v; }
        double mean = sum / flat.length;
        double std = Math.sqrt(Math.max(0.0, sum2 / flat.length - mean * mean));

        double sum3 = 0, sum4 = 0;
        for (double v : flat) {
            double d = v - mean;
            double d2 = d * d;
            sum3 += d2 * d;
            sum4 += d2 * d2;
        }
        double skewness = (std > 1e-9) ? (sum3 / flat.length) / (std * std * std) : 0.0;
        double kurt = (std > 1e-9) ? (sum4 / flat.length) / (std * std * std * std) - 3.0 : 0.0;

        return new double[]{
                mean, std,
                percentile(flat, 50), flat[flat.length - 1] - flat[0],
                percentile(flat, 25), percentile(flat, 75),
                flat[0], flat[flat.length - 1],
                skewness, kurt
        };
    }

    private double percentile(double[] sorted, double p) {
        double index = (p / 100.0) * (sorted.length - 1);
        int lo = (int) index, hi = lo + 1;
        if (hi >= sorted.length) return sorted[sorted.length - 1];
        return sorted[lo] + (index - lo) * (sorted[hi] - sorted[lo]);
    }

    // ---- Parameter vector ----

    private double[] paramsToVector(ArrayList<Parameter> params) {
        double[] v = new double[params.size()];
        for (int i = 0; i < params.size(); i++) {
            Parameter p = params.get(i);
            switch (p.type) {
                case Parameter.TYPE_INT: v[i] = ((IntParameter) p).value; break;
                case Parameter.TYPE_DOUBLE: v[i] = ((DoubleParameter) p).value; break;
                case Parameter.TYPE_CATEGORICAL: v[i] = ((CategoricalParameter) p).active; break;
            }
        }
        return v;
    }

    private double[] combineParamsAndStats(ArrayList<Parameter> params, double[] sv) {
        double[] pv = paramsToVector(params);
        double[] combined = new double[pv.length + sv.length];
        System.arraycopy(pv, 0, combined, 0, pv.length);
        System.arraycopy(sv, 0, combined, pv.length, sv.length);
        return combined;
    }

    // ---- Acquisition functions ----

    private double computeAcquisition(double mu, double sigma, double bestF) {
        // MetricUtils.getRegressionScore already orients every metric so that
        // larger is better, so improvement is a plain difference here.
        double improvement = mu - bestF;
        switch (acquisitionFunctionOption.getChosenIndex()) {
            case 0: { // PI
                double z = improvement / (sigma + 1e-9);
                return normalCDF(z);
            }
            case 1: { // EI
                double z = improvement / (sigma + 1e-9);
                return improvement * normalCDF(z) + sigma * normalPDF(z);
            }
            default: { // UCB
                double kappa = Math.max(0.1, 2.0 * (1.0 - instanceCount / (10.0 * periodicityOption.getValue())));
                return mu + kappa * sigma;
            }
        }
    }

    private double normalCDF(double z) {
        return 0.5 * (1.0 + erf(z / Math.sqrt(2.0)));
    }

    private double normalPDF(double z) {
        return Math.exp(-0.5 * z * z) / Math.sqrt(2.0 * Math.PI);
    }

    private double erf(double x) {
        double t = 1.0 / (1.0 + 0.3275911 * Math.abs(x));
        double poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
        double r = 1.0 - poly * Math.exp(-x * x);
        return x >= 0 ? r : -r;
    }

    // ---- Training ----

    @Override
    public void trainOnInstanceImpl(Instance inst) {
        double[] incumbentVotes = driftDetector != null ? getVotesForInstance(inst) : null;
        initExecutor();
        addToDataWindow(inst);

        InstanceExample example = new InstanceExample(inst);
        Collection<TrainingRunnable> trainers = this.executor == null
                ? null : new ArrayList<TrainingRunnable>();
        for (int i = 0; i < candidates.length; i++) {
            evaluators[i].addResult(example, candidates[i].getVotesForInstance(inst));
            if (trainers == null) {
                candidates[i].trainOnInstance(inst);
            } else {
                // Every candidate owns its model, hence no synchronization here.
                trainers.add(new TrainingRunnable(candidates[i], inst));
            }
        }
        if (trainers != null) {
            try {
                this.executor.invokeAll(trainers);
            } catch (InterruptedException ex) {
                throw new RuntimeException("Could not call invokeAll() on training threads.");
            }
        }

        instanceCount++;
        if (instanceCount % periodicityOption.getValue() == 0) {
            updateModels();
        }

        if (incumbentVotes != null) checkDrift(inst, incumbentVotes);
    }

    @Override
    public void checkParameterChange() {
        updateModels();
    }

    @Override
    public void swapClassifiers(int bestPerforming) {}

    @Override
    public void deepCopyList(int bestPerforming) {}

    @Override
    public void changeStateParameter(Parameter parameter, int index) {}

    private void updateModels() {
        int n = numberOfCandidatesOption.getValue();
        double[] performances = new double[n];
        for (int i = 0; i < n; i++) {
            performances[i] = MetricUtils.getRegressionScore(
                    evaluators[i].getPerformanceMeasurements(), metricOption.getChosenIndex());
        }

        bestCandidateIndex = 0;
        for (int i = 1; i < n; i++) {
            if (performances[i] > performances[bestCandidateIndex]) bestCandidateIndex = i;
        }

        // Window statistics are identical for every candidate this cycle; compute once.
        double[] stats = extractStatFeatures();

        // Train BLR on all non-default candidates (skip index 0)
        for (int i = 1; i < n; i++) {
            double[] features = combineParamsAndStats(candidatesParameters.get(i), stats);
            surrogate.trainOnInstance(createSurrogateInstance(features, performances[i]));
        }

        // Sort worst-first; scores are already oriented so that larger is better.
        Integer[] sortedIdx = new Integer[n];
        for (int i = 0; i < n; i++) sortedIdx[i] = i;
        Arrays.sort(sortedIdx, (a, b) -> Double.compare(performances[a], performances[b]));

        // Replace worst half, never touching the best or the default (index 0).
        int replaceCount = n / 2;
        int replaced = 0;
        for (int k = 0; k < n && replaced < replaceCount; k++) {
            int idx = sortedIdx[k];
            if (idx == bestCandidateIndex || idx == 0) continue;
            replaceCandidate(idx, proposeNextParams(performances, stats));
            replaced++;
        }
    }

    private ArrayList<Parameter> proposeNextParams(double[] performances, double[] stats) {
        double bestPerf = -Double.MAX_VALUE;
        for (double p : performances) {
            if (p > bestPerf) bestPerf = p;
        }

        ArrayList<Parameter> best = null;
        double bestAcq = Double.NEGATIVE_INFINITY;

        for (int k = 0; k < 100; k++) {
            ArrayList<Parameter> candidate = randomConfig();
            double[] pv = paramsToVector(candidate);
            double[] features = new double[pv.length + stats.length];
            System.arraycopy(pv, 0, features, 0, pv.length);
            System.arraycopy(stats, 0, features, pv.length, stats.length);

            double[] muSigma = surrogate.predictWithVariance(createSurrogateInstance(features, 0.0));
            double acq = computeAcquisition(muSigma[0], muSigma[1], bestPerf);
            if (acq > bestAcq) {
                bestAcq = acq;
                best = candidate;
            }
        }
        return best != null ? best : randomConfig();
    }

    private ArrayList<Parameter> randomConfig() {
        ArrayList<Parameter> params = new ArrayList<>();
        for (Parameter p : candidatesParameters.get(0)) {
            Parameter clone;
            switch (p.type) {
                case Parameter.TYPE_INT: {
                    IntParameter ip = (IntParameter) p;
                    int rVal = classifierRandom.nextInt(ip.range[1] - ip.range[0] + 1) + ip.range[0];
                    clone = new IntParameter(ip.name, rVal, ip.range, new Random(classifierRandom.nextLong()));
                    break;
                }
                case Parameter.TYPE_DOUBLE: {
                    DoubleParameter dp = (DoubleParameter) p;
                    double rVal = dp.range[0] + (dp.range[1] - dp.range[0]) * classifierRandom.nextDouble();
                    clone = new DoubleParameter(dp.name, rVal, dp.range, new Random(classifierRandom.nextLong()));
                    break;
                }
                default: {
                    CategoricalParameter cp = (CategoricalParameter) p;
                    int rActive = classifierRandom.nextInt(cp.values.length);
                    clone = new CategoricalParameter(cp.name, cp.values, rActive, new Random(classifierRandom.nextLong()));
                    break;
                }
            }
            params.add(clone);
        }
        // Frozen dimensions take the incumbent (best candidate) value, so the
        // BLR is only ever queried/trained on real feature vectors - never on
        // placeholder values for the parameters we are not optimizing.
        HPOMethod.freezeToIncumbent(this.optimizeMask, params, candidatesParameters.get(bestCandidateIndex));
        return params;
    }

    private void replaceCandidate(int idx, ArrayList<Parameter> params) {
        try {
            Classifier c;
            if (!this.configurator.canApplyLive()) {
                // The space holds a hyperparameter the learner only reads when it
                // is built, so the candidate has to be rebuilt around it.
                c = this.configurator.instantiate(params, this.classifierRandom.nextInt());
            } else {
                c = candidates[bestCandidateIndex].copy();
                LearnerConfigurator.reseedCopy(c, this.classifierRandom.nextInt());
                this.configurator.applyLive(c, params);
            }
            candidates[idx] = c;
            candidatesParameters.set(idx, params);
            evaluators[idx] = new BasicRegressionPerformanceEvaluator();
        } catch (Exception e) {
            e.printStackTrace(System.out);
        }
    }

    // ---- MOA bookkeeping ----

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        ArrayList<Measurement> ms = new ArrayList<>();
        ms.add(new Measurement("driftsDetected", driftsDetected));
        for (Measurement m : this.candidates[bestCandidateIndex].getModelMeasurements()) {
            ms.add(m);
        }
        ms.add(new Measurement("bestCandidateIndex", bestCandidateIndex));
        for (Parameter p : candidatesParameters.get(bestCandidateIndex)) {
            switch (p.type) {
                case Parameter.TYPE_INT: ms.add(new Measurement(p.name, ((IntParameter) p).value)); break;
                case Parameter.TYPE_DOUBLE: ms.add(new Measurement(p.name, ((DoubleParameter) p).value)); break;
                case Parameter.TYPE_CATEGORICAL: ms.add(new Measurement(p.name, ((CategoricalParameter) p).active)); break;
            }
        }
        return ms.toArray(new Measurement[0]);
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {}

    @Override
    public long measureByteSize() {
        long size = SizeOf.sizeOf(this);
        for (Classifier c : candidates) size += c.measureByteSize();
        return size;
    }

    @Override
    public double getCandidateScore(int i) {
        return MetricUtils.getRegressionScore(evaluators[i].getPerformanceMeasurements(),
                metricOption.getChosenIndex());
    }

    @Override
    public double getClassifierScore() {
        return MetricUtils.getRegressionScore(
                evaluators[bestCandidateIndex].getPerformanceMeasurements(), metricOption.getChosenIndex());
    }

    @Override
    public int getNumberOfCandidates() { return numberOfCandidatesOption.getValue(); }

    @Override
    public long getStatesEvaluatedCount() { return instanceCount / periodicityOption.getValue(); }

    @Override
    public int getEvaluationInstancesCount() { return (int) (instanceCount % periodicityOption.getValue()); }

    @Override
    public int getPeriodicity() { return periodicityOption.getValue(); }

    @Override
    public Classifier getMainClassifier() { return candidates[bestCandidateIndex]; }

    @Override
    public ArrayList<Parameter> getReferenceParameters() {
        return (this.candidatesParameters != null && !this.candidatesParameters.isEmpty())
                ? this.candidatesParameters.get(0)
                : null;
    }

    @Override
    public String getConfigurationFile() { return this.configurationFileOption.getValue();}


    @Override
    public ArrayList<ArrayList<Parameter>> getCandidateParameters() {
        return this.candidatesParameters;
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
