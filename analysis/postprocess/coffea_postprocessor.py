import yaml
import glob
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from coffea.util import load, save
from coffea.processor import accumulate
from analysis.filesets.utils import get_dataset_config, get_process_maps
from analysis.postprocess.utils import print_header, get_variations_keys


def load_processed_histograms(
    year: str,
    output_dir: str,
    process_samples_map: dict,
):
    processed_histograms = {}
    for process in process_samples_map:
        processed_histograms.update(load(f"{output_dir}/{process}.coffea"))
    save(processed_histograms, f"{output_dir}/{year}_processed_histograms.coffea")
    return processed_histograms


def find_kin_and_axis(processed_histograms, name="jet_multiplicity"):
    for process, histogram_dict in processed_histograms.items():
        if process == "Data":
            continue
        for kin, hist in histogram_dict.items():
            for axis_name in hist.axes.name:
                if axis_name != "variation" and name in axis_name:
                    return kin, axis_name
    raise ValueError(f"No histogram with a '{name}' axis found.")


def get_results_report(processed_histograms, workflow_config, category, columns_to_drop, blind):
    kin, aux_var = find_kin_and_axis(processed_histograms)
    nominal = {}
    variations = {}
    mcstat_err = {}
    bin_error_up = {}
    bin_error_down = {}
    for process in processed_histograms:
        aux_hist = processed_histograms[process][kin]
        nominal_selector = {"variation": "nominal"}
        if "category" in aux_hist.axes.name:
            nominal_selector["category"] = category
        nominal_hist = aux_hist[nominal_selector].project(aux_var)
        nominal[process] = nominal_hist

        mcstat_err[process] = {}
        bin_error_up[process] = {}
        bin_error_down[process] = {}
        mcstat_err2 = nominal_hist.variances()
        mcstat_err[process] = np.sum(np.sqrt(mcstat_err2))
        err2_up = mcstat_err2
        err2_down = mcstat_err2

        if process == "Data":
            continue

        for variation in get_variations_keys(processed_histograms):
            if f"{variation}Up" not in aux_hist.axes["variation"]:
                continue
            selectorup = {"variation": f"{variation}Up"}
            selectordown = {"variation": f"{variation}Down"}
            if "category" in aux_hist.axes.name:
                selectorup["category"] = category
                selectordown["category"] = category
            var_up = aux_hist[selectorup].project(aux_var).values()
            var_down = aux_hist[selectordown].project(aux_var).values()
            # Compute the uncertainties corresponding to the up/down variations
            err_up = var_up - nominal_hist.values()
            err_down = var_down - nominal_hist.values()
            # Compute the flags to check which of the two variations (up and down) are pushing the nominal value up and down
            up_is_up = err_up > 0
            down_is_down = err_down < 0
            # Compute the flag to check if the uncertainty is one-sided, i.e. when both variations are up or down
            is_onesided = up_is_up ^ down_is_down
            # Sum in quadrature of the systematic uncertainties taking into account if the uncertainty is one- or double-sided
            err2_up_twosided = np.where(up_is_up, err_up**2, err_down**2)
            err2_down_twosided = np.where(up_is_up, err_down**2, err_up**2)
            err2_max = np.maximum(err2_up_twosided, err2_down_twosided)
            err2_up_onesided = np.where(is_onesided & up_is_up, err2_max, 0)
            err2_down_onesided = np.where(is_onesided & down_is_down, err2_max, 0)
            err2_up_combined = np.where(is_onesided, err2_up_onesided, err2_up_twosided)
            err2_down_combined = np.where(
                is_onesided, err2_down_onesided, err2_down_twosided
            )
            # Sum in quadrature of the systematic uncertainty corresponding to a MC sample
            err2_up += err2_up_combined
            err2_down += err2_down_combined

        bin_error_up[process] = np.sum(np.sqrt(err2_up))
        bin_error_down[process] = np.sum(np.sqrt(err2_down))

    mcs = []
    results = {}
    for process in nominal:
        results[process] = {}
        results[process]["events"] = np.sum(nominal[process].values())
        if process == "Data":
            results[process]["stat err"] = np.sqrt(np.sum(nominal[process].values()))
        else:
            if process not in columns_to_drop:
                mcs.append(process)
            results[process]["stat err"] = mcstat_err[process]
            results[process]["syst err up"] = bin_error_up[process]
            results[process]["syst err down"] = bin_error_down[process]
    df = pd.DataFrame(results)
    df["Total background"] = df.loc[["events"], mcs].sum(axis=1)
    df.loc["stat err", "Total background"] = np.sqrt(
        np.sum(df.loc["stat err", mcs] ** 2)
    )
    df.loc["syst err up", "Total background"] = np.sqrt(
        np.sum(df.loc["syst err up", mcs] ** 2)
    )
    df.loc["syst err down", "Total background"] = np.sqrt(
        np.sum(df.loc["syst err down", mcs] ** 2)
    )
    df = df.T
    if not blind:
        df.loc["Data/Total background"] = (
            df.loc["Data", ["events"]] / df.loc["Total background", ["events"]]
        )
    return df