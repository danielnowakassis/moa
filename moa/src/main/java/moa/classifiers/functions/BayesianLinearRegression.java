/*
 *    BayesianLinearRegression.java
 *    Port of River's BayesianLinearRegression to MOA.
 *    Based on Bishop's Pattern Recognition and Machine Learning (2006), equations 3.50-3.59.
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
package moa.classifiers.functions;

import com.github.javacliparser.FloatOption;
import com.yahoo.labs.samoa.instances.Instance;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.Regressor;
import moa.core.Measurement;
import moa.core.StringUtils;


public class BayesianLinearRegression extends AbstractClassifier implements Regressor {

    private static final long serialVersionUID = 1L;

    @Override
    public String getPurposeString() {
        return "Bayesian linear regression. Does not require feature scaling. "
                + "Supports concept drift via the smoothing parameter. "
                + "Port of River's BayesianLinearRegression.";
    }

    public FloatOption alphaOption = new FloatOption("alpha", 'a',
            "Prior precision. Controls the strength of the Gaussian prior over the weights.",
            1.0, 1e-10, Double.MAX_VALUE);

    public FloatOption betaOption = new FloatOption("beta", 'b',
            "Noise precision (inverse of the assumed noise variance).",
            1.0, 1e-10, Double.MAX_VALUE);

    public FloatOption smoothingOption = new FloatOption("smoothing", 's',
            "Smoothing factor in (0, 1) for concept drift adaptation. "
                    + "Set to 0 (default) to disable smoothing and use the fast Sherman-Morrison update. "
                    + "A value such as 0.8 makes the model gradually forget older observations.",
            0.0, 0.0, 1.0);

    // Posterior precision matrix S_N  (numFeatures x numFeatures)
    private double[][] S;

    // Posterior covariance matrix S_N^{-1}
    private double[][] Sinv;

    // Posterior mean vector m_N
    private double[] m;

    // Number of input features (excluding class attribute)
    private int numFeatures;

    @Override
    public void resetLearningImpl() {
        S = null;
        Sinv = null;
        m = null;
        numFeatures = 0;
    }

    /** Initialises S, S_inv, and m for n features using the configured alpha. */
    private void initModel(int n) {
        double alpha = alphaOption.getValue();
        numFeatures = n;
        S = new double[n][n];
        Sinv = new double[n][n];
        m = new double[n];
        // Prior: S_0 = alpha * I,  S_0^{-1} = (1/alpha) * I
        for (int i = 0; i < n; i++) {
            S[i][i] = alpha;
            Sinv[i][i] = 1.0 / alpha;
        }
    }

    /** Extracts the input feature values from an instance, skipping the class attribute. */
    private double[] featureVector(Instance inst) {
        int n = inst.numAttributes() - 1;
        double[] x = new double[n];
        int idx = 0;
        for (int i = 0; i < inst.numAttributes(); i++) {
            if (i != inst.classIndex()) {
                double v = inst.value(i);
                x[idx++] = Double.isNaN(v) ? 0.0 : v;
            }
        }
        return x;
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {
        if (inst.classIsMissing()) return;

        double[] x = featureVector(inst);
        int n = x.length;
        if (n == 0) return;

        if (S == null) initModel(n);

        double y = inst.classValue();
        double beta = betaOption.getValue();
        double smoothing = smoothingOption.getValue();

        // beta * x
        double[] bx = scale(x, beta);

        // S * m (needed in both branches)
        double[] Sm = matVec(S, m);

        if (smoothing <= 0.0) {
            // ---- No smoothing: Sherman-Morrison rank-1 update ----
            //
            // S_N     = S_{N-1} + beta * x * x^T          (Bishop eq. 3.51)
            // S_N^-1  updated via Sherman-Morrison:
            //   S_N^-1 = S_{N-1}^-1
            //            - (S_{N-1}^-1 * bx)(x^T * S_{N-1}^-1)
            //              / (1 + x^T * S_{N-1}^-1 * bx)
            // m_N     = S_N^-1 * (S_{N-1} * m_{N-1} + beta * y * x)  (eq. 3.50)

            double[] Sinv_bx = matVec(Sinv, bx);          // S^-1 * (beta*x)
            double[] xT_Sinv = vecMat(x, Sinv);           // x^T * S^-1
            double denom = 1.0 + dot(x, Sinv_bx);

            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    Sinv[i][j] -= Sinv_bx[i] * xT_Sinv[j] / denom;

            double[] rhs = new double[n];
            for (int i = 0; i < n; i++)
                rhs[i] = Sm[i] + bx[i] * y;
            m = matVec(Sinv, rhs);

            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    S[i][j] += bx[i] * x[j];

        } else {
            // ---- With smoothing: full matrix inversion ----
            //
            // S_N     = smoothing * S_{N-1} + (1-smoothing) * beta * x * x^T
            // S_N^-1  = inv(S_N)
            // m_N     = S_N^-1 * (smoothing * S_{N-1} * m_{N-1}
            //                     + (1-smoothing) * beta * y * x)

            double[][] newS = new double[n][n];
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    newS[i][j] = smoothing * S[i][j] + (1.0 - smoothing) * bx[i] * x[j];

            double[][] newSinv = invertMatrix(newS);

            double[] rhs = new double[n];
            for (int i = 0; i < n; i++)
                rhs[i] = smoothing * Sm[i] + (1.0 - smoothing) * bx[i] * y;
            m = matVec(newSinv, rhs);

            S = newS;
            Sinv = newSinv;
        }
    }

    /**
     * Returns the predicted value as the posterior mean estimate (Bishop eq. 3.58):
     *   y_hat = m_N^T * x
     */
    @Override
    public double[] getVotesForInstance(Instance inst) {
        if (m == null) return new double[]{0.0};
        double[] x = featureVector(inst);
        return new double[]{dot(m, x)};
    }

    /**
     * Returns {mean, std} of the predictive distribution (Bishop eq. 3.58-3.59):
     *   mean = m_N^T * x
     *   var  = 1/beta + x^T * S_N^{-1} * x
     */
    public double[] predictWithVariance(Instance inst) {
        if (m == null) return new double[]{0.0, 1.0};
        double[] x = featureVector(inst);
        double mean = dot(m, x);
        double[] Sinv_x = matVec(Sinv, x);
        double variance = 1.0 / betaOption.getValue() + dot(x, Sinv_x);
        return new double[]{mean, Math.sqrt(Math.max(0.0, variance))};
    }

    // ---- Matrix / vector helpers ----

    /** Matrix-vector product: A * v */
    private double[] matVec(double[][] A, double[] v) {
        int n = v.length;
        double[] r = new double[n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                r[i] += A[i][j] * v[j];
        return r;
    }

    /** Row-vector-matrix product: v^T * A  (returns row vector) */
    private double[] vecMat(double[] v, double[][] A) {
        int n = v.length;
        double[] r = new double[n];
        for (int j = 0; j < n; j++)
            for (int i = 0; i < n; i++)
                r[j] += v[i] * A[i][j];
        return r;
    }

    /** Dot product of two vectors. */
    private double dot(double[] a, double[] b) {
        double s = 0.0;
        for (int i = 0; i < a.length; i++) s += a[i] * b[i];
        return s;
    }

    /** Returns a new vector: scalar * v. */
    private double[] scale(double[] v, double scalar) {
        double[] r = new double[v.length];
        for (int i = 0; i < v.length; i++) r[i] = scalar * v[i];
        return r;
    }

    /**
     * Inverts a square matrix using Gauss-Jordan elimination with partial pivoting.
     * Returns the identity matrix if the input is (near-)singular.
     */
    private double[][] invertMatrix(double[][] A) {
        int n = A.length;
        // Build augmented matrix [A | I]
        double[][] aug = new double[n][2 * n];
        for (int i = 0; i < n; i++) {
            System.arraycopy(A[i], 0, aug[i], 0, n);
            aug[i][n + i] = 1.0;
        }

        for (int col = 0; col < n; col++) {
            // Partial pivot
            int pivotRow = col;
            for (int row = col + 1; row < n; row++)
                if (Math.abs(aug[row][col]) > Math.abs(aug[pivotRow][col]))
                    pivotRow = row;
            double[] tmp = aug[col]; aug[col] = aug[pivotRow]; aug[pivotRow] = tmp;

            double pivotVal = aug[col][col];
            if (Math.abs(pivotVal) < 1e-15) {
                // Singular column — leave as-is (graceful degradation)
                continue;
            }

            // Scale pivot row
            double inv = 1.0 / pivotVal;
            for (int j = col; j < 2 * n; j++) aug[col][j] *= inv;

            // Eliminate column in all other rows
            for (int row = 0; row < n; row++) {
                if (row == col) continue;
                double factor = aug[row][col];
                if (factor == 0.0) continue;
                for (int j = col; j < 2 * n; j++)
                    aug[row][j] -= factor * aug[col][j];
            }
        }

        // Extract right half as the inverse
        double[][] inv = new double[n][n];
        for (int i = 0; i < n; i++)
            System.arraycopy(aug[i], n, inv[i], 0, n);
        return inv;
    }

    // ---- MOA bookkeeping ----

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        return null;
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {
        StringUtils.appendIndented(out, indent, "Bayesian Linear Regression");
        StringUtils.appendNewline(out);
        if (m == null) {
            StringUtils.appendIndented(out, indent, "  Model not yet trained.");
            StringUtils.appendNewline(out);
            return;
        }
        StringUtils.appendIndented(out, indent, String.format(
                "  alpha=%.4f  beta=%.4f  smoothing=%.4f%n",
                alphaOption.getValue(), betaOption.getValue(), smoothingOption.getValue()));

        StringUtils.appendIndented(out, indent, "  Posterior mean weights (m_N):");
        StringUtils.appendNewline(out);
        // Attempt to print feature names from model context when available
        com.yahoo.labs.samoa.instances.InstancesHeader header = getModelContext();
        for (int i = 0, fi = 0; header != null && i < header.numAttributes(); i++) {
            if (i == header.classIndex()) continue;
            StringUtils.appendIndented(out, indent,
                    String.format("    %s: %.6f%n", header.attribute(i).name(), m[fi++]));
        }
        if (header == null) {
            for (int i = 0; i < m.length; i++) {
                StringUtils.appendIndented(out, indent,
                        String.format("    w[%d]: %.6f%n", i, m[i]));
            }
        }
    }

    @Override
    public boolean isRandomizable() {
        return false;
    }
}