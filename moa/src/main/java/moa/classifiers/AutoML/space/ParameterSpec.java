/*
 *    ParameterSpec.java
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
package moa.classifiers.AutoML.space;

import moa.classifiers.AutoML.Parameters.Parameter;

import java.io.Serializable;
import java.util.Arrays;

/**
 * Static description of one tunable hyperparameter, as read from the search
 * space JSON. This is the immutable half of a hyperparameter: the mutable
 * per-candidate value lives in {@link moa.classifiers.AutoML.Parameters.Parameter}.
 *
 * <p>The {@code parameter} field is an <i>option path</i>. A bare name such as
 * {@code "gracePeriod"} refers to an option of the learner itself; a slash
 * separated path such as {@code "treeLearner/gracePeriod"} refers to an option
 * of a nested learner reached through a {@link moa.options.ClassOption}. The
 * leaf segment is always the MOA option name, i.e. the first argument given to
 * the {@code IntOption} / {@code FloatOption} / {@code ClassOption} constructor
 * in the learner - not the Java field name, which conventionally carries an
 * {@code Option} suffix.
 *
 * @author Daniel Nowak Assis (daniel dot nowak-assis at lip6 dot fr)
 */
public class ParameterSpec implements Serializable {

    private static final long serialVersionUID = 1L;

    /** Matches {@link moa.classifiers.AutoML.Parameters.IntParameter}. */
    public static final int TYPE_INT = Parameter.TYPE_INT;

    /** Matches {@link moa.classifiers.AutoML.Parameters.DoubleParameter}. */
    public static final int TYPE_DOUBLE = Parameter.TYPE_DOUBLE;

    /** Matches {@link moa.classifiers.AutoML.Parameters.CategoricalParameter}. */
    public static final int TYPE_CATEGORICAL = Parameter.TYPE_CATEGORICAL;

    /**
     * Option path, e.g. {@code "gracePeriod"} or {@code "treeLearner/gracePeriod"}.
     * Read from the {@code "parameter"} entry of the search space JSON.
     */
    public String name;

    /** One of {@link #TYPE_INT}, {@link #TYPE_DOUBLE}, {@link #TYPE_CATEGORICAL}. */
    public int type;

    /** Initial value for numeric parameters. */
    public double value;

    /** Inclusive {@code [low, high]} bounds for numeric parameters. */
    public double[] range;

    /** Optional step hint used by the local-search methods; {@code NaN} when absent. */
    public double step = Double.NaN;

    /** Candidate CLI strings for categorical parameters. */
    public String[] values;

    /** Index into {@link #values} used as the initial setting. */
    public int active;

    /**
     * Whether this hyperparameter can be pushed into an already-trained learner
     * ({@code "onChange": "live"}, the default) or requires the candidate to be
     * rebuilt from scratch ({@code "onChange": "reset"}).
     *
     * <p>Declare {@code reset} whenever the learner only consumes the option
     * once, to derive some structural quantity it will not recompute: MOA
     * examples are {@code AdaptiveRandomForest.mFeaturesPerTreeSize}, which is
     * turned into {@code subspaceSize} in {@code initEnsemble}, and
     * {@code kNN.limit}, which sizes the sliding window on first use. Writing
     * the option after that point is silently ignored by the learner, so a
     * {@code live} declaration there would freeze the search.
     */
    public boolean live = true;

    /** Leaf segment of {@link #name}, i.e. the MOA option name. */
    public String optionName() {
        int slash = this.name.lastIndexOf('/');
        return slash < 0 ? this.name : this.name.substring(slash + 1);
    }

    /** Path segments leading to the owning learner, empty when the option is on the root. */
    public String[] pathPrefix() {
        int slash = this.name.lastIndexOf('/');
        if (slash < 0) return new String[0];
        return this.name.substring(0, slash).split("/");
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("ParameterSpec [").append(this.name).append(", ");
        switch (this.type) {
            case TYPE_INT:
                sb.append("integer, value=").append((int) this.value)
                  .append(", range=").append(Arrays.toString(new int[]{(int) this.range[0], (int) this.range[1]}));
                break;
            case TYPE_DOUBLE:
                sb.append("double, value=").append(this.value)
                  .append(", range=").append(Arrays.toString(this.range));
                break;
            default:
                sb.append("categorical, active=").append(this.active)
                  .append(", values=").append(Arrays.toString(this.values));
        }
        sb.append(this.live ? ", live]" : ", reset]");
        return sb.toString();
    }
}
