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


def get_variable_array(histogram, histogram_config, variable, variables_map, flow):
    if histogram_config.axes[variable].type in ["IntCategory", "Integer"]:
        # cast to integer array
        variable_array = normalize(variables_map[variable])
        variable_array = ak.to_numpy(variable_array).astype(int)
    elif flow:
        # add underflow/overflow to first/last bin
        variable_array = get_flow_array(
            histogram=histogram,
            variable=variable,
            variables_map=variables_map,
        )
    else:
        variable_array = normalize(variables_map[variable])
    return variable_array


def fill_histogram(
    histograms, histogram_config, variables_map, category, weights, variation, flow=True
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
                fill_args.update({"category": category})
            if histogram_config.add_syst_axis:
                fill_args.update({"variation": variation})
            if histogram_config.add_weight:
                fill_args.update(
                    {
                        "weight": (
                            ak.flatten(ak.ones_like(variables_map[variable]) * weights)
                            if variables_map[variable].ndim == 2
                            else weights
                        ),
                    }
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
                fill_args.update({"category": category})
            if histogram_config.add_syst_axis:
                fill_args.update({"variation": variation})
            if histogram_config.add_weight:
                fill_args.update(
                    {
                        "weight": (
                            ak.flatten(ak.ones_like(variables_map[variable]) * weights)
                            if variables_map[variable].ndim == 2
                            else weights
                        ),
                    }
                )
            histograms[key].fill(**fill_args)


def fill_histograms(
    histograms,
    histogram_config,
    variables_map,
    category,
    shift_name,
    flow,
    is_mc,
    weights_container,
):
    if is_mc and (shift_name == "nominal"):
        variations = ["nominal"] + list(weights_container.variations)
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
                flow=True,
            )
    elif is_mc and (shift_name != "nominal"):
        region_weight = weights_container.weight()
        fill_histogram(
            histograms=histograms,
            histogram_config=histogram_config,
            variables_map=variables_map,
            weights=region_weight,
            variation=shift_name,
            category=category,
            flow=True,
        )
    elif not is_mc and (shift_name == "nominal"):
        region_weight = weights_container.weight()
        fill_histogram(
            histograms=histograms,
            histogram_config=histogram_config,
            variables_map=variables_map,
            weights=region_weight,
            variation=shift_name,
            category=category,
            flow=True,
        )
