/*
 *    CategoricalParameter.java
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
import java.util.Random;

/**
 * A categorical hyperparameter: one of {@link #values} is selected by the
 * {@link #active} index, which is what the search actually moves.
 *
 * @author Daniel Nowak Assis (daniel dot nowak-assis at lip6 dot fr)
 */
public class CategoricalParameter extends Parameter implements Serializable {

    private static final long serialVersionUID = 1L;

    public String[] values;
    public int active;
    public Random random;

    public CategoricalParameter(String name, String[] values, int active, Random random) {
        this.name = name;
        this.values = values;
        this.active = active;
        this.type = TYPE_CATEGORICAL;
        this.random = random;
    }

    @Override
    public void changeParameter() {
        this.active = this.random.nextInt(this.values.length);
    }

    @Override
    public String toString() {
        return "CategoricalParameter [name=" + this.name
                + ", active=" + this.values[this.active] + "]";
    }
}
