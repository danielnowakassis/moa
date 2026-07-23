/*
 *    FeatureImportanceLearner.java
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
 *
 */
package moa.learners.featureanalysis;

import moa.classifiers.Classifier;

/**
 * Feature Importance Learner
 *
 * <p>This interface defines the methods to be implemented on a learner to allow it to produce
 * feature importances. Nothing here depends on the kind of prediction being made, so it covers
 * classifiers and regressors alike; a regressor additionally declares
 * {@link moa.classifiers.Regressor} so that the regression tasks and the regression tab of the
 * GUI pick it up.</p>
 *
 * <p>See details in:<br> Heitor Murilo Gomes, Rodrigo Fernandes de Mello, Bernhard Pfahringer,
 * Albert Bifet. Feature Scoring using Tree-Based Ensembles for Evolving Data Streams.
 * IEEE International Conference on Big Data (pp. 761-769), 2019</p>
 *
 * @author Heitor Murilo Gomes
 */
public interface FeatureImportanceLearner extends Classifier {

    /**
     * Obtain the current importance for each feature.
     *
     * @param normalize whether to rescale the scores before returning them
     * @return array containing the importance/score estimated for each feature
     */
    double[] getFeatureImportances(boolean normalize);

    /**
     * The output is an array where values indicate the original feature index and the order of
     * the array its ranking. The size of this array is expected to be less than or equal to the
     * complete set of features.
     *
     * @param k how many features to return
     * @param normalize whether to rank the normalized scores
     * @return the k features with the highest scores.
     */
    int[] getTopKFeatures(int k, boolean normalize);
}
