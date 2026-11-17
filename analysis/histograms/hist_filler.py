import numpy as np
import awkward as ak


def normalize(array: ak.Array):
    if array.ndim == 2:
        return ak.fill_none(ak.flatten(array), np.nan)
    else:
        return ak.fill_none(array, np.nan)


def get_flow_array(histogram, variable, variables_map):
    histogram_edges = histogram.axes[variable].edges
    epsilon = (histogram_edges[-1] - histogram_edges[-2]) / 2
    hist_max_bin_edge = histogram_edges[-1] - epsilon
    hist_min_bin_edge = histogram_edges[0]

    return np.maximum(
        np.minimum(normalize(variables_map[variable]), hist_max_bin_edge),
        hist_min_bin_edge,
    )


def get_weight_array(variable, variables_map, weights):
    variable_array = variables_map[variable]

    if variable_array.ndim == 2:
        weight_array = ak.broadcast_arrays(
            ak.Array(weights)[:, np.newaxis],
            variable_array,
        )[0]
        return ak.flatten(weight_array)

    return weights


def get_variable_array(
    histogram,
    histogram_config,
    variable,
    variables_map,
    flow,
):
    axis_type = histogram_config.axes[variable].type

    if axis_type in ["IntCategory", "Integer"]:
        variable_array = normalize(variables_map[variable])
        variable_array = ak.to_numpy(variable_array).astype(int)

    elif axis_type == "Boolean":
        variable_array = normalize(variables_map[variable])
        variable_array = ak.to_numpy(variable_array).astype(bool)

    elif flow:
        # Add underflow and overflow to the first and last bins.
        variable_array = get_flow_array(
            histogram=histogram,
            variable=variable,
            variables_map=variables_map,
        )

    else:
        variable_array = normalize(variables_map[variable])

    return variable_array


def fill_histogram(
    histograms,
    histogram_config,
    variables_map,
    category,
    weights,
    variation,
    flow=True,
):
    if histogram_config.layout == "individual":
        for variable in histograms:
            variable_array = get_variable_array(
                histogram=histograms[variable],
                histogram_config=histogram_config,
                variable=variable,
                variables_map=variables_map,
                flow=flow,
            )

            fill_args = {variable: variable_array}

            if len(histogram_config.categories) > 1:
                fill_args["category"] = category

            if histogram_config.add_syst_axis:
                fill_args["variation"] = variation

            if histogram_config.add_weight:
                fill_args["weight"] = get_weight_array(
                    variable=variable,
                    variables_map=variables_map,
                    weights=weights,
                )

            histograms[variable].fill(**fill_args)

    else:
        for key, variables in histogram_config.layout.items():
            fill_args = {}
            for variable in variables:
                fill_args[variable] = get_variable_array(
                    histogram=histograms[key],
                    histogram_config=histogram_config,
                    variable=variable,
                    variables_map=variables_map,
                    flow=flow,
                )
            if len(histogram_config.categories) > 1:
                fill_args["category"] = category
            if histogram_config.add_syst_axis:
                fill_args["variation"] = variation
            if histogram_config.add_weight:
                # All variables in a histogram must have compatible
                # event/object-level structures.
                reference_variable = variables[0]

                fill_args["weight"] = get_weight_array(
                    variable=reference_variable,
                    variables_map=variables_map,
                    weights=weights,
                )
            histograms[key].fill(**fill_args)


def fill_histograms(
    histograms,
    histogram_config,
    variables_map,
    category,
    flow,
    is_mc,
    weights_container,
    shift_name,
):
    if shift_name is None:
        variations = ["nominal"]

        if is_mc:
            variations += list(weights_container.variations)

        for variation in variations:
            if variation == "nominal":
                region_weight = weights_container.weight()
            else:
                region_weight = weights_container.weight(modifier=variation)

            fill_histogram(
                histograms=histograms,
                histogram_config=histogram_config,
                variables_map=variables_map,
                weights=region_weight,
                variation=variation,
                category=category,
                flow=flow,
            )

    else:
        region_weight = weights_container.weight()
        fill_histogram(
            histograms=histograms,
            histogram_config=histogram_config,
            variables_map=variables_map,
            weights=region_weight,
            variation=shift_name,
            category=category,
            flow=flow,
        )