/*
 *    IntParameter.java
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
import java.util.Arrays;
import java.util.Random;

/**
 * An integer hyperparameter, uniformly drawn from an inclusive range.
 *
 * @author Daniel Nowak Assis (daniel dot nowak-assis at lip6 dot fr)
 */
public class IntParameter extends Parameter implements Serializable {

    private static final long serialVersionUID = 1L;

    public int value;
    public int[] range;
    public Random random;

    public IntParameter(String name, int value, int[] range, Random random) {
        this.name = name;
        this.value = value;
        this.range = range;
        this.type = TYPE_INT;
        this.random = random;
    }

    @Override
    public void changeParameter() {
        this.value = this.random.nextInt((this.range[1] - this.range[0] + 1)) + this.range[0];
        this.value = Math.min(this.range[1], Math.max(this.range[0], this.value));
    }

    @Override
    public String toString() {
        return "IntParameter [name=" + this.name + ", value=" + this.value
                + ", range=" + Arrays.toString(this.range) + "]";
    }
}
