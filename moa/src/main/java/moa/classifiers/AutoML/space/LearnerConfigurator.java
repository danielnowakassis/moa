/*
 *    LearnerConfigurator.java
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

import com.github.javacliparser.FlagOption;
import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.github.javacliparser.MultiChoiceOption;
import com.github.javacliparser.Option;
import com.github.javacliparser.StringOption;
import moa.classifiers.AutoML.Parameters.CategoricalParameter;
import moa.classifiers.AutoML.Parameters.DoubleParameter;
import moa.classifiers.AutoML.Parameters.IntParameter;
import moa.classifiers.AutoML.Parameters.Parameter;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.options.AbstractOptionHandler;
import moa.options.ClassOption;
import moa.options.OptionHandler;
import moa.options.OptionsHandler;
import moa.tasks.NullMonitor;

import java.io.Serializable;
import java.lang.reflect.Array;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * The single point of contact between an AutoML method and the learner it
 * tunes. Every hyperparameter write in this package goes through here, and no
 * learner in MOA needs to be modified to support it.
 *
 * <p>Hyperparameters are addressed as MOA {@link Option}s rather than as plain
 * Java fields, so a value is set with e.g. {@code IntOption.setValue(250)} on
 * the very {@code IntOption} instance the learner already dereferences. Two
 * ways of getting a configuration into a learner are offered:
 *
 * <ul>
 *   <li>{@link #instantiate} - the <i>cold</i> path. Builds a fresh learner
 *       from the search space's CLI string, writes every option, and resets it.
 *       Correct for any learner and any hyperparameter, at the cost of the
 *       model state.</li>
 *   <li>{@link #applyLive} - the <i>warm</i> path. Pushes options into a
 *       learner that is already training, so a candidate can inherit the
 *       incumbent's model. Reaches nested learners by walking the object graph:
 *       on an {@code AdaptiveRandomForest} it writes {@code gracePeriod} on the
 *       {@code treeLearner} prototype <i>and</i> on all
 *       {@code ensemble[i].classifier} trees, which is what makes a warm-started
 *       forest actually honour a hyperparameter change.</li>
 * </ul>
 *
 * <p>The warm path only works for hyperparameters the learner re-reads as it
 * trains. Ones that are consumed once to derive structural state - see
 * {@link ParameterSpec#live} - must be declared {@code "onChange": "reset"} in
 * the search space and are routed to the cold path instead.
 *
 * @author Daniel Nowak Assis (daniel dot nowak-assis at lip6 dot fr)
 */
public class LearnerConfigurator implements Serializable {

    private static final long serialVersionUID = 1L;

    /**
     * Depth cap for the object-graph walk. Sub-learners sit very close to the
     * root - an {@code AdaptiveRandomForest} tree is at depth three, via
     * {@code ensemble} then {@code ARFBaseLearner} then {@code classifier} - so
     * this is generous while still bounding pathological graphs.
     */
    private static final int MAX_DEPTH = 8;

    /** Hard cap on visited objects, a backstop against unforeseen graph shapes. */
    private static final int MAX_VISITS = 50000;

    /**
     * Types the walk must not descend into: learner model internals, which can
     * hold hundreds of thousands of objects and never contain a nested learner.
     * Resolved by name so that a MOA build missing one of them still works.
     */
    private static final String[] OPAQUE_TYPE_NAMES = {
            "moa.classifiers.trees.HoeffdingTree$Node",
            "moa.classifiers.trees.FIMTDD$Node",
            "moa.classifiers.trees.EFDT$EFDTNode",
            "moa.classifiers.core.attributeclassobservers.AttributeClassObserver",
            "moa.core.DoubleVector",
            "moa.core.GaussianEstimator",
            "moa.core.GreenwaldKhannaQuantileSummary",
    };

    private static final List<Class<?>> OPAQUE_TYPES = resolveOpaqueTypes();

    /**
     * {@code AbstractOptionHandler.config} holds the {@link OptionsHandler} that
     * memoises materialised {@link ClassOption} objects. Replacing a
     * ClassOption's value leaves that memo stale, and MOA offers no public way
     * to invalidate it, so this one protected field of MOA core is read
     * reflectively. It is the only reflective access in this class, and it does
     * not touch any learner-specific member.
     */
    private static final Field CONFIG_FIELD = resolveConfigField();

    private final ConfigurationSpace space;

    /** Cached per-class field lists for the object-graph walk. */
    private transient Map<Class<?>, Field[]> fieldCache;

    public LearnerConfigurator(ConfigurationSpace space) {
        this.space = space;
    }

    public ConfigurationSpace getSpace() {
        return this.space;
    }

    /** Whether every hyperparameter in the space may be pushed into a live learner. */
    public boolean canApplyLive() {
        return this.space.allLive();
    }

    // ------------------------------------------------------------------
    // Construction
    // ------------------------------------------------------------------

    /** A fresh learner at the search space's base configuration, reset and ready to train. */
    public Classifier newLearner() {
        Classifier learner = rawLearner();
        learner.resetLearning();
        return learner;
    }

    /** As {@link #newLearner()}, with {@code seed} pushed into a randomizable learner. */
    public Classifier newLearner(int seed) {
        Classifier learner = rawLearner();
        seed(learner, seed);
        learner.resetLearning();
        return learner;
    }

    private Classifier rawLearner() {
        try {
            return (Classifier) ClassOption.cliStringToObject(this.space.algorithm, Classifier.class, null);
        } catch (Exception e) {
            throw new IllegalStateException("Could not create the learner to tune from \""
                    + this.space.algorithm + "\": " + e.getMessage(), e);
        }
    }

    /**
     * Seeds a candidate, so that two candidates carrying identical
     * hyperparameters still behave differently and the pool does not collapse
     * into copies of one model.
     *
     * <p>Only learners that declare {@link Classifier#isRandomizable()} are
     * seeded: {@code AbstractClassifier} creates its seed option, and reseeds
     * {@code classifierRandom} from it in {@code resetLearning}, only under that
     * flag, so seeding a non-randomizable learner would be silently discarded.
     * This must run <i>before</i> {@code resetLearning}, which is what turns the
     * seed into the learner's random stream.
     */
    private static void seed(Classifier learner, int seed) {
        if (learner.isRandomizable()) {
            learner.setRandomSeed(seed);
        }
    }

    /**
     * Reseeds a warm-started candidate in place, i.e. one obtained by copying
     * an incumbent rather than by building it from the search space.
     *
     * <p>{@link #seed} is not enough on such a candidate: {@code setRandomSeed}
     * only records the seed, and {@code AbstractClassifier} turns it into a
     * random stream in {@code resetLearning} - which is precisely what warm
     * starting avoids, since that would discard the learned state being reused.
     * The generator is therefore replaced directly.
     *
     * <p>Without this, every candidate copied from one incumbent in the same
     * evaluation window shares that incumbent's generator <i>and its position</i>
     * in the stream, because MOA's {@code copy()} is serialization based. Two
     * candidates carrying identical hyperparameters would then be exact
     * duplicates and make identical random decisions, wasting a pool slot.
     *
     * <p>Only randomizable {@link AbstractClassifier}s expose a generator, so
     * anything else is left untouched. The reseed applies to the learner itself,
     * not to randomizable learners nested inside its {@link ClassOption}s.
     */
    public static void reseedCopy(Classifier learner, int seed) {
        if (learner instanceof AbstractClassifier && learner.isRandomizable()) {
            AbstractClassifier randomizable = (AbstractClassifier) learner;
            randomizable.setRandomSeed(seed);
            randomizable.classifierRandom = new Random(seed);
        }
    }

    /**
     * Builds a new learner carrying {@code params}. Used whenever a candidate
     * must start from scratch, and as the fallback for hyperparameters that
     * cannot be changed on a running learner.
     */
    public Classifier instantiate(List<? extends Parameter> params) {
        Classifier learner = rawLearner();
        writeByPath(learner, params);
        learner.resetLearning();
        return learner;
    }

    /** As {@link #instantiate(List)}, with {@code seed} pushed into a randomizable learner. */
    public Classifier instantiate(List<? extends Parameter> params, int seed) {
        Classifier learner = rawLearner();
        writeByPath(learner, params);
        seed(learner, seed);
        learner.resetLearning();
        return learner;
    }

    /**
     * Writes {@code params} into {@code learner} by resolving each option path
     * from the root. Nested learners are reached through the live objects held
     * by their {@link ClassOption}s, so no CLI round trip is involved.
     */
    private void writeByPath(Classifier learner, List<? extends Parameter> params) {
        for (int i = 0; i < params.size(); i++) {
            Parameter param = params.get(i);
            ParameterSpec spec = specFor(param, i);
            OptionHandler owner = resolveOwner(learner, spec);
            Option option = owner.getOptions().getOption(spec.optionName());
            if (option == null) {
                throw new IllegalStateException("Learner " + owner.getClass().getName()
                        + " has no option named \"" + spec.optionName() + "\" (from parameter \""
                        + spec.name + "\").");
            }
            if (write(option, param) && option instanceof ClassOption) {
                invalidateClassOptionCache(owner);
            }
        }
    }

    /** Walks a path prefix such as {@code treeLearner/...} down to the learner that owns the option. */
    private OptionHandler resolveOwner(OptionHandler root, ParameterSpec spec) {
        OptionHandler current = root;
        for (String segment : spec.pathPrefix()) {
            Option option = current.getOptions().getOption(segment);
            if (!(option instanceof ClassOption)) {
                throw new IllegalStateException("Path segment \"" + segment + "\" of parameter \""
                        + spec.name + "\" is not a class option of "
                        + current.getClass().getName() + ".");
            }
            Object nested = ((ClassOption) option).getPreMaterializedObject();
            if (!(nested instanceof OptionHandler)) {
                throw new IllegalStateException("Path segment \"" + segment + "\" of parameter \""
                        + spec.name + "\" does not resolve to a configurable object.");
            }
            current = (OptionHandler) nested;
        }
        return current;
    }

    // ------------------------------------------------------------------
    // Live reconfiguration
    // ------------------------------------------------------------------

    /**
     * Pushes {@code params} into an already-training learner, returning the
     * number of options actually written.
     *
     * <p>Options are matched by leaf name across the whole reachable object
     * graph, not by path. That is deliberate: an ensemble's sub-learners are
     * private copies of the prototype held by its {@link ClassOption}, so a
     * path-scoped write would update the prototype and leave every tree already
     * in the ensemble at its old setting. Matching by name reaches both.
     *
     * <p>A return value of zero means the learner ignored the configuration
     * entirely, which is worth surfacing rather than silently tolerating.
     */
    public int applyLive(Object learner, List<? extends Parameter> params) {
        if (learner == null || params == null || params.isEmpty()) return 0;

        Map<String, Parameter> byName = new HashMap<>();
        for (int i = 0; i < params.size(); i++) {
            Parameter param = params.get(i);
            ParameterSpec spec = specFor(param, i);
            if (!spec.live) continue;
            byName.put(spec.optionName(), param);
        }
        if (byName.isEmpty()) return 0;

        int written = 0;
        IdentityHashMap<Object, Boolean> seen = new IdentityHashMap<>();
        ArrayDeque<Object> queue = new ArrayDeque<>();
        ArrayDeque<Integer> depths = new ArrayDeque<>();
        queue.add(learner);
        depths.add(0);
        int visits = 0;

        while (!queue.isEmpty() && visits < MAX_VISITS) {
            Object node = queue.poll();
            int depth = depths.poll();
            if (node == null || seen.put(node, Boolean.TRUE) != null) continue;
            visits++;

            if (node instanceof OptionHandler) {
                OptionHandler handler = (OptionHandler) node;
                boolean classOptionWritten = false;
                for (Option option : handler.getOptions().getOptionArray()) {
                    Parameter param = byName.get(option.getName());
                    if (param != null && write(option, param)) {
                        written++;
                        classOptionWritten |= option instanceof ClassOption;
                    }
                    // Descend through nested learners held by class options: this
                    // is how an ensemble's prototype gets reconfigured.
                    if (depth < MAX_DEPTH && option instanceof ClassOption) {
                        Object nested = ((ClassOption) option).getPreMaterializedObject();
                        if (shouldVisit(nested)) {
                            queue.add(nested);
                            depths.add(depth + 1);
                        }
                    }
                }
                if (classOptionWritten) invalidateClassOptionCache(handler);
            }

            if (depth >= MAX_DEPTH) continue;
            for (Object child : childrenOf(node)) {
                if (shouldVisit(child)) {
                    queue.add(child);
                    depths.add(depth + 1);
                }
            }
        }
        return written;
    }

    /** Single-parameter convenience used by the per-candidate mutation step. */
    public int applyLive(Object learner, Parameter param) {
        List<Parameter> one = new ArrayList<>(1);
        one.add(param);
        return applyLive(learner, one);
    }

    // ------------------------------------------------------------------
    // Option writing
    // ------------------------------------------------------------------

    /** Writes one parameter value onto one option, returning whether it applied. */
    private static boolean write(Option option, Parameter param) {
        switch (param.type) {
            case ParameterSpec.TYPE_INT: {
                int value = ((IntParameter) param).value;
                if (option instanceof IntOption) {
                    ((IntOption) option).setValue(value);
                    return true;
                }
                if (option instanceof FloatOption) {
                    ((FloatOption) option).setValue(value);
                    return true;
                }
                return false;
            }
            case ParameterSpec.TYPE_DOUBLE: {
                double value = ((DoubleParameter) param).value;
                if (option instanceof FloatOption) {
                    ((FloatOption) option).setValue(value);
                    return true;
                }
                if (option instanceof IntOption) {
                    ((IntOption) option).setValue((int) Math.round(value));
                    return true;
                }
                return false;
            }
            case ParameterSpec.TYPE_CATEGORICAL: {
                CategoricalParameter categorical = (CategoricalParameter) param;
                String value = categorical.values[categorical.active];
                if (option instanceof ClassOption) {
                    ((ClassOption) option).setValueViaCLIString(value);
                    return true;
                }
                if (option instanceof MultiChoiceOption) {
                    ((MultiChoiceOption) option).setChosenLabel(value);
                    return true;
                }
                if (option instanceof FlagOption) {
                    ((FlagOption) option).setValue(Boolean.parseBoolean(value));
                    return true;
                }
                if (option instanceof StringOption) {
                    ((StringOption) option).setValue(value);
                    return true;
                }
                return false;
            }
            default:
                return false;
        }
    }

    /**
     * Drops the memo of materialised class options so the learner picks up a
     * newly written {@link ClassOption} value. Only called after such a write.
     */
    private static void invalidateClassOptionCache(OptionHandler handler) {
        if (CONFIG_FIELD == null || !(handler instanceof AbstractOptionHandler)) return;
        try {
            Object config = CONFIG_FIELD.get(handler);
            if (config instanceof OptionsHandler) {
                ((OptionsHandler) config).prepareClassOptions(new NullMonitor(), null);
            }
        } catch (IllegalAccessException ignored) {
            // Without the memo refresh a categorical change would not be seen;
            // validate() reports this up front rather than failing silently here.
        }
    }

    // ------------------------------------------------------------------
    // Validation
    // ------------------------------------------------------------------

    /**
     * Checks the search space against the learner before any training starts:
     * every option path must resolve, every option must accept the declared
     * type, and both ends of every numeric range must be within the bounds MOA
     * declares for that option. Fails loudly here rather than thousands of
     * instances into a run.
     */
    public void validate() {
        Classifier probe = rawLearner();

        if (CONFIG_FIELD == null) {
            for (ParameterSpec spec : this.space.parameters) {
                if (spec.type == ParameterSpec.TYPE_CATEGORICAL) {
                    throw new IllegalStateException("Cannot tune categorical parameter \""
                            + spec.name + "\": this MOA build does not expose "
                            + "AbstractOptionHandler.config, so class option changes could not take effect.");
                }
            }
        }

        for (ParameterSpec spec : this.space.parameters) {
            OptionHandler owner = resolveOwner(probe, spec);
            Option option = owner.getOptions().getOption(spec.optionName());
            if (option == null) {
                throw new IllegalStateException("Parameter \"" + spec.name + "\": "
                        + owner.getClass().getName() + " has no option named \"" + spec.optionName()
                        + "\". Use the MOA option name, not the Java field name -"
                        + " for instance \"gracePeriod\", not \"gracePeriodOption\".");
            }
            switch (spec.type) {
                case ParameterSpec.TYPE_INT:
                case ParameterSpec.TYPE_DOUBLE:
                    if (!(option instanceof IntOption) && !(option instanceof FloatOption)) {
                        throw new IllegalStateException("Parameter \"" + spec.name
                                + "\" is declared numeric but option \"" + spec.optionName() + "\" of "
                                + owner.getClass().getName() + " is a " + option.getClass().getSimpleName() + ".");
                    }
                    checkBound(option, spec, spec.range[0]);
                    checkBound(option, spec, spec.range[1]);
                    break;
                case ParameterSpec.TYPE_CATEGORICAL:
                    if (!(option instanceof ClassOption) && !(option instanceof MultiChoiceOption)
                            && !(option instanceof FlagOption) && !(option instanceof StringOption)) {
                        throw new IllegalStateException("Parameter \"" + spec.name
                                + "\" is declared categorical but option \"" + spec.optionName() + "\" of "
                                + owner.getClass().getName() + " is a " + option.getClass().getSimpleName() + ".");
                    }
                    for (String value : spec.values) {
                        try {
                            CategoricalParameter probeParam =
                                    new CategoricalParameter(spec.optionName(), new String[]{value}, 0, null);
                            write(option, probeParam);
                        } catch (RuntimeException e) {
                            throw new IllegalStateException("Parameter \"" + spec.name
                                    + "\" cannot take value \"" + value + "\": " + e.getMessage(), e);
                        }
                    }
                    break;
                default:
                    break;
            }
        }
    }

    private static void checkBound(Option option, ParameterSpec spec, double bound) {
        try {
            if (option instanceof IntOption) {
                ((IntOption) option).setValue((int) Math.round(bound));
            } else {
                ((FloatOption) option).setValue(bound);
            }
        } catch (RuntimeException e) {
            throw new IllegalStateException("Parameter \"" + spec.name + "\" has a range reaching "
                    + bound + ", which the learner rejects: " + e.getMessage(), e);
        }
    }

    private ParameterSpec specFor(Parameter param, int index) {
        if (index < this.space.size()) {
            ParameterSpec spec = this.space.get(index);
            if (spec.name.equals(param.name) || spec.optionName().equals(param.name)) {
                return spec;
            }
        }
        for (ParameterSpec spec : this.space.parameters) {
            if (spec.name.equals(param.name) || spec.optionName().equals(param.name)) {
                return spec;
            }
        }
        throw new IllegalStateException("No search space entry for parameter \"" + param.name + "\".");
    }

    // ------------------------------------------------------------------
    // Object-graph walk
    // ------------------------------------------------------------------

    private boolean shouldVisit(Object object) {
        if (object == null) return false;
        Class<?> type = object.getClass();
        if (type.isArray()) return !type.getComponentType().isPrimitive();
        if (object instanceof Option) return false;
        if (object instanceof Collection || object instanceof Map) return true;
        if (object instanceof OptionHandler) return true;

        String name = type.getName();
        if (!name.startsWith("moa.")) return false;
        for (Class<?> opaque : OPAQUE_TYPES) {
            if (opaque.isInstance(object)) return false;
        }
        return true;
    }

    /** Object-valued members of {@code node} that the walk should consider. */
    private Iterable<Object> childrenOf(Object node) {
        List<Object> children = new ArrayList<>();
        Class<?> type = node.getClass();

        if (type.isArray()) {
            int length = Array.getLength(node);
            for (int i = 0; i < length; i++) children.add(Array.get(node, i));
            return children;
        }
        if (node instanceof Collection) {
            children.addAll((Collection<?>) node);
            return children;
        }
        if (node instanceof Map) {
            children.addAll(((Map<?, ?>) node).values());
            return children;
        }

        for (Field field : fieldsOf(type)) {
            try {
                children.add(field.get(node));
            } catch (IllegalAccessException | RuntimeException ignored) {
                // Inaccessible members simply do not take part in the walk.
            }
        }
        return children;
    }

    private Field[] fieldsOf(Class<?> type) {
        if (this.fieldCache == null) this.fieldCache = new HashMap<>();
        Field[] cached = this.fieldCache.get(type);
        if (cached != null) return cached;

        List<Field> fields = new ArrayList<>();
        for (Class<?> current = type; current != null && current != Object.class; current = current.getSuperclass()) {
            for (Field field : current.getDeclaredFields()) {
                if (Modifier.isStatic(field.getModifiers())) continue;
                if (field.getType().isPrimitive()) continue;
                try {
                    field.setAccessible(true);
                } catch (RuntimeException e) {
                    continue;
                }
                fields.add(field);
            }
        }
        cached = fields.toArray(new Field[0]);
        this.fieldCache.put(type, cached);
        return cached;
    }

    private static List<Class<?>> resolveOpaqueTypes() {
        List<Class<?>> types = new ArrayList<>();
        for (String name : OPAQUE_TYPE_NAMES) {
            try {
                types.add(Class.forName(name));
            } catch (Throwable ignored) {
                // Absent in this build; nothing to exclude.
            }
        }
        return types;
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
