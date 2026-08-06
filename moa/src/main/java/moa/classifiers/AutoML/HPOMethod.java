/*
 *    HPOMethod.java
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

import moa.classifiers.AutoML.Parameters.CategoricalParameter;
import moa.classifiers.AutoML.Parameters.DoubleParameter;
import moa.classifiers.AutoML.Parameters.IntParameter;
import moa.classifiers.AutoML.Parameters.Parameter;
import moa.classifiers.Classifier;

import java.util.ArrayList;

/**
 * Common surface of the streaming hyperparameter optimisation methods, so that
 * wrappers can drive and introspect any of them without knowing which search
 * strategy is underneath.
 *
 * @author Daniel Nowak Assis (daniel dot nowak-assis at lip6 dot fr)
 */
public interface HPOMethod {

    void setConfigurations();

    void checkParameterChange();

    void swapClassifiers(int bestPerforming);

    void deepCopyList(int bestPerforming);

    void changeStateParameter(Parameter parameter, int index);

    default void cleanThreads() {}

    // Accessors letting an introspecting wrapper read the runtime state of an
    // HPO method. Defaults return null/0 so implementers compile unchanged;
    // a wrapper should require its wrapped learner to override the ones it needs.

    default ArrayList<Parameter> getReferenceParameters() { return null; }

    default ArrayList<ArrayList<Parameter>> getCandidateParameters() { return null; }

    default double getCandidateScore(int i) { return Double.NaN; }

    default double getClassifierScore() { return Double.NaN; }

    default int getNumberOfCandidates() { return 0; }

    /**
     * Number of candidates actually trained and evaluated in the current
     * evaluation window. Methods that keep a fixed pool simply report the pool
     * size; methods with a "max capacity pool, active prefix" design report the
     * prefix length.
     */
    default int getActiveCandidates() { return getNumberOfCandidates(); }

    /**
     * Request that only {@code n} candidates be trained and evaluated from the
     * next evaluation window on. Implementations must clamp {@code n} to
     * {@code [1, poolCapacity]} and must only let the change take effect at a
     * window boundary, so that all scores compared inside one window come from
     * candidates that saw the same instances. Returns the granted value, which
     * may differ from {@code n}.
     *
     * <p>Pushed by an external controller that sizes the candidate budget. The
     * default is a no-op so that existing implementers compile unchanged.
     */
    default int setActiveCandidates(int n) { return getActiveCandidates(); }

    default long getStatesEvaluatedCount() { return 0; }

    default int getEvaluationInstancesCount() { return 0; }

    default int getPeriodicity() { return Integer.MAX_VALUE; }

    default Classifier getMainClassifier() { return null; }

    default String getConfigurationFile() { return null; }

    /**
     * Restrict optimisation to a subset of hyperparameters. {@code mask[k]} is
     * {@code true} when parameter k (in search space order; a categorical
     * counts as a single entry) should keep being optimised, and {@code false}
     * when it must be frozen to the current incumbent ("best configuration")
     * value. A {@code null} mask (the default) means optimise every parameter,
     * preserving the unrestricted behaviour.
     */
    default void setOptimizableParameters(boolean[] mask) {}

    /**
     * Overwrite the frozen dimensions of {@code params} (those with
     * {@code mask[k] == false}) with the corresponding value from
     * {@code incumbent}. Optimised dimensions are left untouched. Shared by all
     * HPO methods so that frozen hyperparameters carry the real best-configuration
     * value rather than a placeholder - important for surrogate-based methods
     * (e.g. the Bayesian Stream Tuner) whose feature vector must stay consistent.
     */
    static void freezeToIncumbent(boolean[] mask, ArrayList<Parameter> params,
                                  ArrayList<Parameter> incumbent) {
        if (mask == null || params == null || incumbent == null) return;
        int n = Math.min(params.size(), Math.min(incumbent.size(), mask.length));
        for (int k = 0; k < n; k++) {
            if (mask[k]) continue;
            Parameter dst = params.get(k);
            Parameter src = incumbent.get(k);
            if (dst.type != src.type) continue;
            switch (dst.type) {
                case Parameter.TYPE_INT: ((IntParameter) dst).value = ((IntParameter) src).value; break;
                case Parameter.TYPE_DOUBLE: ((DoubleParameter) dst).value = ((DoubleParameter) src).value; break;
                case Parameter.TYPE_CATEGORICAL: ((CategoricalParameter) dst).active = ((CategoricalParameter) src).active; break;
            }
        }
    }
}
