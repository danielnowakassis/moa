/*
 *    LearnerAccess.java
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

import com.yahoo.labs.samoa.instances.Instance;
import moa.classifiers.Classifier;
import moa.options.AbstractOptionHandler;
import moa.options.ClassOption;
import moa.options.OptionHandler;
import moa.options.OptionsHandler;
import moa.tasks.NullMonitor;

import java.lang.reflect.Array;
import java.lang.reflect.Field;

/**
 * Read-only access to learner internals that MOA does not expose publicly, for
 * the surrogate machinery of the adaptive and multi-fidelity methods.
 *
 * <p>The reference implementation obtained these by widening the visibility of
 * fields inside {@code AdaptiveRandomForest} and
 * {@code AdaptivePredictionInterval}. Doing it here instead keeps every learner
 * exactly as MOA ships it, at the cost of a little reflection confined to this
 * class - and, unlike a field widening, it degrades gracefully: a learner that
 * does not look like an ensemble simply reports no member predictions, and the
 * caller falls back to its interval-based uncertainty estimate.
 *
 * @author Daniel Nowak Assis (daniel dot nowak-assis at lip6 dot fr)
 */
public final class LearnerAccess {

    private static final Field CONFIG_FIELD = resolveConfigField();

    private LearnerAccess() {}

    /**
     * Per-member predictions of an ensemble learner, or {@code null} when
     * {@code learner} does not expose an {@code ensemble} of sub-learners.
     * Members that abstain are skipped, so the result may be shorter than the
     * ensemble; a result of fewer than two entries carries no spread and is
     * reported as {@code null}.
     */
    public static double[] memberPredictions(Object learner, Instance inst) {
        if (learner == null || inst == null) return null;
        Object ensemble = readField(learner, "ensemble");
        if (ensemble == null || !ensemble.getClass().isArray()) return null;

        int length = Array.getLength(ensemble);
        double[] predictions = new double[length];
        int count = 0;
        for (int i = 0; i < length; i++) {
            Classifier member = asClassifier(Array.get(ensemble, i));
            if (member == null) continue;
            double[] votes = member.getVotesForInstance(inst);
            if (votes == null || votes.length == 0 || Double.isNaN(votes[0])) continue;
            predictions[count++] = votes[0];
        }
        if (count < 2) return null;
        double[] trimmed = new double[count];
        System.arraycopy(predictions, 0, trimmed, 0, count);
        return trimmed;
    }

    /** Standard deviation of the member predictions, or {@code NaN} when there are none. */
    public static double ensembleStd(Object learner, Instance inst) {
        double[] predictions = memberPredictions(learner, inst);
        if (predictions == null) return Double.NaN;
        double mean = 0.0;
        for (double p : predictions) mean += p / predictions.length;
        double variance = 0.0;
        for (double p : predictions) {
            double d = p - mean;
            variance += d * d / predictions.length;
        }
        return Math.sqrt(Math.max(0.0, variance));
    }

    /**
     * Point an existing {@link ClassOption} at an object the caller already
     * owns, so a wrapper can hand its own model to a learner that would
     * otherwise build one from the option's CLI string.
     *
     * @return whether the option was found and set
     */
    public static boolean setNestedLearner(OptionHandler owner, String optionName, Object learner) {
        if (owner == null) return false;
        com.github.javacliparser.Option option = owner.getOptions().getOption(optionName);
        if (!(option instanceof ClassOption)) return false;
        ((ClassOption) option).setCurrentObject(learner);
        refreshClassOptions(owner);
        return true;
    }

    /**
     * Rebuild the memo of materialised class options, so a freshly assigned
     * {@link ClassOption} value is the one the learner goes on to use.
     */
    public static void refreshClassOptions(OptionHandler owner) {
        if (CONFIG_FIELD == null || !(owner instanceof AbstractOptionHandler)) return;
        try {
            Object config = CONFIG_FIELD.get(owner);
            if (config instanceof OptionsHandler) {
                ((OptionsHandler) config).prepareClassOptions(new NullMonitor(), null);
            }
        } catch (IllegalAccessException ignored) {
            // Nothing to refresh; the caller's option write simply will not be seen.
        }
    }

    /** The member itself when it is a classifier, otherwise the classifier it wraps. */
    private static Classifier asClassifier(Object member) {
        if (member instanceof Classifier) return (Classifier) member;
        Object inner = readField(member, "classifier");
        return inner instanceof Classifier ? (Classifier) inner : null;
    }

    private static Object readField(Object target, String name) {
        if (target == null) return null;
        for (Class<?> c = target.getClass(); c != null && c != Object.class; c = c.getSuperclass()) {
            try {
                Field field = c.getDeclaredField(name);
                field.setAccessible(true);
                return field.get(target);
            } catch (NoSuchFieldException ignored) {
                // keep walking up
            } catch (RuntimeException | IllegalAccessException e) {
                return null;
            }
        }
        return null;
    }

    private static Field resolveConfigField() {
        try {
            Field field = AbstractOptionHandler.class.getDeclaredField("config");
            field.setAccessible(true);
            return field;
        } catch (Throwable ignored) {
            return null;
        }
    }
}
