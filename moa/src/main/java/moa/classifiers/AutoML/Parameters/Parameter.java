/*
 *    Parameter.java
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
package moa.classifiers.AutoML.Parameters;

import java.io.Serializable;

/**
 * One hyperparameter carrying a concrete value for one candidate. This is the
 * mutable half of a hyperparameter: its immutable description, as read from the
 * search space JSON, lives in {@link moa.classifiers.AutoML.space.ParameterSpec}.
 *
 * <p>{@link #name} is the option path of the hyperparameter, e.g.
 * {@code "gracePeriod"} or {@code "treeLearner/gracePeriod"}, matching the
 * {@code "parameter"} entry of the search space JSON.
 *
 * @author Daniel Nowak Assis (daniel dot nowak-assis at lip6 dot fr)
 */
public abstract class Parameter implements Serializable {

    private static final long serialVersionUID = 1L;

    /** Type tag of {@link IntParameter}. */
    public static final int TYPE_INT = 0;

    /** Type tag of {@link DoubleParameter}. */
    public static final int TYPE_DOUBLE = 1;

    /** Type tag of {@link CategoricalParameter}. */
    public static final int TYPE_CATEGORICAL = 2;

    /** One of {@link #TYPE_INT}, {@link #TYPE_DOUBLE}, {@link #TYPE_CATEGORICAL}. */
    public int type;

    /** Option path of this hyperparameter. */
    public String name;

    /** Draws a new value for this hyperparameter from its own range. */
    public abstract void changeParameter();
}
