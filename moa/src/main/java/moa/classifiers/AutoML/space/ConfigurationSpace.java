/*
 *    ConfigurationSpace.java
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

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.io.Serializable;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.List;

/**
 * The search space shared by every AutoML method: the CLI string of the learner
 * being tuned, plus the list of tunable hyperparameters.
 *
 * <p>Expected JSON, as produced by hand or by the CapyMOA wrappers:
 *
 * <pre>
 * {
 *   "algorithm": "moa.classifiers.trees.HoeffdingTree",
 *   "parameters": [
 *     {"parameter": "gracePeriod",    "type": "integer",     "value": 200,  "range": [50, 500], "step": 10},
 *     {"parameter": "splitConfidence","type": "double",      "value": 1E-7, "range": [1E-7, 0.05]},
 *     {"parameter": "splitCriterion", "type": "categorical", "active": 0,
 *      "values": ["moa.classifiers.core.splitcriteria.InfoGainSplitCriterion",
 *                 "moa.classifiers.core.splitcriteria.GiniSplitCriterion"]}
 *   ]
 * }
 * </pre>
 *
 * <p>{@code "parameter"} is an option path resolved against the learner's MOA
 * options - see {@link ParameterSpec}. {@code "type"} accepts {@code integer} /
 * {@code int}, {@code double} / {@code float} / {@code real}, and
 * {@code categorical} / {@code nominal}. The optional {@code "onChange"} key
 * takes {@code "live"} (default) or {@code "reset"}.
 *
 * @author Daniel Nowak Assis (daniel dot nowak-assis at lip6 dot fr)
 */
public class ConfigurationSpace implements Serializable {

    private static final long serialVersionUID = 1L;

    /** CLI string of the learner to tune, e.g. {@code "moa.classifiers.trees.HoeffdingTree -g 50"}. */
    public final String algorithm;

    /** Tunable hyperparameters, in declaration order. */
    public final List<ParameterSpec> parameters;

    public ConfigurationSpace(String algorithm, List<ParameterSpec> parameters) {
        this.algorithm = algorithm;
        this.parameters = parameters;
    }

    public static ConfigurationSpace fromFile(String path) throws IOException {
        if (path == null || path.trim().isEmpty()) {
            throw new IOException("No configuration file given. Set the -f option to a search space JSON file.");
        }
        File file = new File(path);
        if (!file.isFile()) {
            throw new IOException("Configuration file not found: " + file.getAbsolutePath());
        }
        try (Reader reader = new FileReader(file)) {
            return read(reader, file.getAbsolutePath());
        }
    }

    public static ConfigurationSpace fromString(String json) throws IOException {
        return read(new StringReader(json), "<string>");
    }

    private static ConfigurationSpace read(Reader reader, String origin) throws IOException {
        JsonElement rootElement = JsonParser.parseReader(reader);
        if (rootElement == null || !rootElement.isJsonObject()) {
            throw new IOException("Configuration in " + origin + " is not a JSON object.");
        }
        JsonObject root = rootElement.getAsJsonObject();

        if (!root.has("algorithm")) {
            throw new IOException("Configuration in " + origin + " has no \"algorithm\" entry.");
        }
        String algorithm = root.get("algorithm").getAsString();

        List<ParameterSpec> specs = new ArrayList<>();
        if (root.has("parameters")) {
            JsonArray array = root.getAsJsonArray("parameters");
            for (int i = 0; i < array.size(); i++) {
                specs.add(readParameter(array.get(i).getAsJsonObject(), origin, i));
            }
        }
        return new ConfigurationSpace(algorithm, specs);
    }

    private static ParameterSpec readParameter(JsonObject node, String origin, int index) throws IOException {
        ParameterSpec spec = new ParameterSpec();

        if (!node.has("parameter")) {
            throw new IOException("Parameter " + index + " in " + origin + " has no \"parameter\" name.");
        }
        spec.name = node.get("parameter").getAsString();

        String type = node.has("type") ? node.get("type").getAsString().toLowerCase() : "";
        switch (type) {
            case "integer":
            case "int":
                spec.type = ParameterSpec.TYPE_INT;
                break;
            case "double":
            case "float":
            case "real":
                spec.type = ParameterSpec.TYPE_DOUBLE;
                break;
            case "categorical":
            case "nominal":
                spec.type = ParameterSpec.TYPE_CATEGORICAL;
                break;
            default:
                throw new IOException("Parameter \"" + spec.name + "\" in " + origin
                        + " has unknown type \"" + type + "\".");
        }

        if (spec.type == ParameterSpec.TYPE_CATEGORICAL) {
            if (!node.has("values")) {
                throw new IOException("Categorical parameter \"" + spec.name + "\" in " + origin
                        + " has no \"values\" list.");
            }
            JsonArray values = node.getAsJsonArray("values");
            spec.values = new String[values.size()];
            for (int i = 0; i < values.size(); i++) {
                spec.values[i] = values.get(i).getAsString();
            }
            spec.active = node.has("active") ? node.get("active").getAsInt() : 0;
            if (spec.active < 0 || spec.active >= spec.values.length) {
                throw new IOException("Categorical parameter \"" + spec.name + "\" in " + origin
                        + " has \"active\" outside its \"values\" list.");
            }
        } else {
            if (!node.has("range")) {
                throw new IOException("Numeric parameter \"" + spec.name + "\" in " + origin
                        + " has no \"range\".");
            }
            JsonArray range = node.getAsJsonArray("range");
            if (range.size() != 2) {
                throw new IOException("Numeric parameter \"" + spec.name + "\" in " + origin
                        + " needs a \"range\" of exactly two entries.");
            }
            spec.range = new double[]{range.get(0).getAsDouble(), range.get(1).getAsDouble()};
            if (spec.range[0] > spec.range[1]) {
                throw new IOException("Numeric parameter \"" + spec.name + "\" in " + origin
                        + " has an inverted \"range\".");
            }
            spec.value = node.has("value") ? node.get("value").getAsDouble() : spec.range[0];
            if (node.has("step")) {
                spec.step = node.get("step").getAsDouble();
            }
        }

        if (node.has("onChange")) {
            String onChange = node.get("onChange").getAsString().toLowerCase();
            switch (onChange) {
                case "live":
                    spec.live = true;
                    break;
                case "reset":
                    spec.live = false;
                    break;
                default:
                    throw new IOException("Parameter \"" + spec.name + "\" in " + origin
                            + " has unknown \"onChange\" value \"" + onChange + "\"; expected live or reset.");
            }
        }

        return spec;
    }

    /** Number of {@code integer} and {@code double} parameters, i.e. the surrogate feature count. */
    public int numericalCount() {
        int count = 0;
        for (ParameterSpec spec : this.parameters) {
            if (spec.type != ParameterSpec.TYPE_CATEGORICAL) count++;
        }
        return count;
    }

    /** Whether every hyperparameter can be pushed into an already-trained learner. */
    public boolean allLive() {
        for (ParameterSpec spec : this.parameters) {
            if (!spec.live) return false;
        }
        return true;
    }

    public int size() {
        return this.parameters.size();
    }

    public ParameterSpec get(int index) {
        return this.parameters.get(index);
    }
}
