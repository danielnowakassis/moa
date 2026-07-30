/*
 *    MetricUtils.java
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

import moa.core.Measurement;
import moa.evaluation.BasicClassificationPerformanceEvaluator;

/**
 * Shared metric selection for the AutoML methods. Centralises the list of
 * metrics exposed through each method's {@code metricOption}, read out of a
 * {@link BasicClassificationPerformanceEvaluator}'s measurements by name so it
 * is independent of the number of classes.
 *
 * <p>F1, precision and recall are aggregated here rather than taken from the
 * evaluator's own aggregate measurements, for two reasons. First, MOA's
 * {@code getF1Statistic()} returns the F1 <i>of the averaged</i> precision and
 * recall, which is not the macro F1 and is not what a search should rank
 * candidates by. Second, a class absent from an evaluation window leaves its
 * per-class estimator at {@code 0/0}, so MOA's aggregates collapse to
 * {@code NaN} and every candidate ties - fatal for a search that ranks by that
 * score. Both are handled here by averaging per-class values under an explicit
 * NaN policy, which keeps the correction inside this package instead of
 * changing an evaluator the rest of MOA depends on.
 *
 * <p>Methods using these metrics must switch on the per-class outputs of their
 * evaluators, which {@link #configure} does.
 *
 * @author Daniel Nowak Assis (daniel dot nowak-assis at lip6 dot fr)
 */
public final class MetricUtils {

    public static final String[] METRIC_NAMES = {
            "Accuracy", "Kappa", "KappaTemporal", "KappaM", "F1", "Precision", "Recall"
    };

    public static final String[] METRIC_DESCRIPTIONS = {
            "Percentage of correctly classified instances",
            "Kappa statistic (percent)",
            "Kappa Temporal statistic (percent)",
            "Kappa M statistic (percent)",
            "F1 score, macro averaged over classes (percent)",
            "Precision, macro averaged over observed classes (percent)",
            "Recall, macro averaged over observed classes (percent)"
    };

    public static final int ACCURACY = 0;
    public static final int KAPPA = 1;
    public static final int KAPPA_TEMPORAL = 2;
    public static final int KAPPA_M = 3;
    public static final int F1 = 4;
    public static final int PRECISION = 5;
    public static final int RECALL = 6;

    private static final String[] AGGREGATE_NAMES = {
            "classifications correct (percent)",
            "Kappa Statistic (percent)",
            "Kappa Temporal Statistic (percent)",
            "Kappa M Statistic (percent)",
            "F1 Score (percent)",
            "Precision (percent)",
            "Recall (percent)"
    };

    private static final String F1_PER_CLASS = "F1 Score for class ";
    private static final String PRECISION_PER_CLASS = "Precision for class ";
    private static final String RECALL_PER_CLASS = "Recall for class ";

    public static final String[] REGRESSION_METRIC_NAMES = {
            "R2", "AdjustedR2", "MAE", "RMSE", "RelativeMAE", "RelativeRMSE"
    };

    public static final String[] REGRESSION_METRIC_DESCRIPTIONS = {
            "Coefficient of determination",
            "Adjusted coefficient of determination",
            "Mean absolute error",
            "Root mean squared error",
            "Relative mean absolute error",
            "Relative root mean squared error"
    };

    private static final String[] REGRESSION_MEASUREMENT_NAMES = {
            "coefficient of determination",
            "adjusted coefficient of determination",
            "mean absolute error",
            "root mean squared error",
            "relative mean absolute error",
            "relative root mean squared error"
    };

    /** Whether a larger value of the metric at the same index means a better model. */
    private static final boolean[] REGRESSION_HIGHER_IS_BETTER = {
            true, true, false, false, false, false
    };

    private MetricUtils() {}

    /**
     * Switches on every output {@link #getScore} needs. Call this on each
     * evaluator an AutoML method creates, in place of setting
     * {@code precisionRecallOutputOption} alone.
     */
    public static void configure(BasicClassificationPerformanceEvaluator evaluator) {
        evaluator.precisionRecallOutputOption.setValue(true);
        evaluator.f1PerClassOption.setValue(true);
        evaluator.precisionPerClassOption.setValue(true);
        evaluator.recallPerClassOption.setValue(true);
        // A freshly constructed evaluator allocates its estimators only in
        // reset(), so getPerformanceMeasurements() would throw on one that has
        // not scored an instance yet. That happens whenever the search state is
        // read at an evaluation window boundary, just after candidates were
        // respawned with new evaluators.
        evaluator.reset();
    }

    public static double getScore(Measurement[] measurements, int metricChoice) {
        if (measurements == null || metricChoice < 0 || metricChoice >= AGGREGATE_NAMES.length) {
            return Double.NEGATIVE_INFINITY;
        }
        switch (metricChoice) {
            case F1:
                // An undefined class scores zero, so the average stays over all
                // classes and a candidate that ignores a class is penalised.
                return macroAverage(measurements, F1_PER_CLASS, AGGREGATE_NAMES[F1], true);
            case PRECISION:
                return macroAverage(measurements, PRECISION_PER_CLASS, AGGREGATE_NAMES[PRECISION], false);
            case RECALL:
                return macroAverage(measurements, RECALL_PER_CLASS, AGGREGATE_NAMES[RECALL], false);
            default:
                return lookup(measurements, AGGREGATE_NAMES[metricChoice]);
        }
    }

    /**
     * Averages the per-class entries whose name starts with {@code prefix}.
     * When {@code countUndefined} is set, classes with no observations count as
     * zero; otherwise they are left out of the average entirely. Falls back to
     * {@code aggregateName} if per-class outputs are switched off.
     */
    private static double macroAverage(Measurement[] measurements, String prefix,
                                       String aggregateName, boolean countUndefined) {
        double total = 0.0;
        int classes = 0;
        int defined = 0;
        for (Measurement measurement : measurements) {
            if (!measurement.getName().startsWith(prefix)) continue;
            classes++;
            double value = measurement.getValue();
            if (Double.isNaN(value)) continue;
            total += value;
            defined++;
        }
        if (classes == 0) return lookup(measurements, aggregateName);
        if (countUndefined) return total / classes;
        return defined == 0 ? 0.0 : total / defined;
    }

    /**
     * Regression counterpart of {@link #getScore}, always oriented so that
     * larger is better: error metrics are negated, so a search can rank
     * candidates with a plain {@code >} whichever metric is selected.
     *
     * <p>Measurements are looked up by name rather than by position, and the
     * same metric is used for candidates and for the incumbent. Comparing, say,
     * a candidate's adjusted R² against the incumbent's plain R² would make
     * swaps depend on the number of attributes rather than on model quality.
     */
    public static double getRegressionScore(Measurement[] measurements, int metricChoice) {
        if (measurements == null || metricChoice < 0 || metricChoice >= REGRESSION_MEASUREMENT_NAMES.length) {
            return Double.NEGATIVE_INFINITY;
        }
        double value = lookup(measurements, REGRESSION_MEASUREMENT_NAMES[metricChoice]);
        if (Double.isNaN(value)) return Double.NEGATIVE_INFINITY;
        return REGRESSION_HIGHER_IS_BETTER[metricChoice] ? value : -value;
    }

    private static double lookup(Measurement[] measurements, String name) {
        for (Measurement measurement : measurements) {
            if (name.equals(measurement.getName())) return measurement.getValue();
        }
        return Double.NEGATIVE_INFINITY;
    }
}
