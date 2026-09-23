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


def save_process_histograms_by_sample(
    grouped_outputs,
    sample,
    year: str,
    output_dir: str,
    categories,
):
    print_header(f"Processing {sample} outputs")

    metadata = {}
    histograms = {}
    grouped_metadata = {}
    grouped_histograms = {}

    logging.info("Accumulating histograms and metadata...")
    grouped_histograms = []
    grouped_metadata = {}
    for fname in grouped_outputs[sample]:
        output = load(fname)
        if output:
            # group histograms by sample
            grouped_histograms.append(output["histograms"])
            # group metadata by sample
            for meta_key in output["metadata"]:
                if meta_key in grouped_metadata:
                    grouped_metadata[meta_key].append(output["metadata"][meta_key])
                else:
                    grouped_metadata[meta_key] = [output["metadata"][meta_key]]

    histograms = accumulate(grouped_histograms)
    metadata = {}
    for meta_key in grouped_metadata:
        metadata[meta_key] = accumulate(grouped_metadata[meta_key])

    logging.info("Scaling lumi-xsec weights")
    lumi_file = Path.cwd() / "analysis" / "postprocess" / "luminosity.yaml"
    with open(lumi_file, "r") as f:
        luminosities = yaml.safe_load(f)

    weights = {}
    xsecs = {}
    sumw = {}
    weight = 1
    dataset_config = get_dataset_config(year)
    xsec = dataset_config[sample]["xsec"]
    sumw = metadata["sumw"]
    if dataset_config[sample]["era"] == "MC":
        weight = (luminosities[year] * xsec) / sumw

    logging.info(f"luminosity [1/pb]: {luminosities[year]}")
    logging.info(f"xsec [pb]: {xsec}")
    logging.info(f"sumw: {sumw}")
    logging.info(f"weight: {weight}")

    scaled_histograms = {}
    for variable in histograms:
        scaled_histograms[variable] = histograms[variable] * weight

    logging.info(f"saving histograms")
    save(scaled_histograms, f"{output_dir}/{sample}.coffea")

    scaled_cutflow = {}
    for category in categories:
        logging.info(f"saving cutflow for category {category}\n")
        category_dir = Path(f"{output_dir}/{category}")
        if not category_dir.exists():
            category_dir.mkdir(parents=True, exist_ok=True)
        scaled_cutflow[category] = {}
        if category in metadata:
            for cut, nevents in metadata[category]["cutflow"].items():
                scaled_cutflow[category][cut] = nevents * weight
        processed_cutflow = {sample: scaled_cutflow[category]}
        cutflow_file = Path(
            f"{output_dir}/{category}/cutflow_{category}_{sample}.coffea"
        )
        save(processed_cutflow, cutflow_file)


def save_process_histograms_by_process(
    process: str,
    year: str,
    output_dir: str,
    process_samples_map: dict,
    categories,
):
    print_header(f"Processing {process} outputs")
    # group and accumulate output files by sample
    extension = ".coffea"
    output_files = []
    for sample in process_samples_map[process]:
        output_files += glob.glob(f"{output_dir}/{sample}*{extension}", recursive=True)

    logging.info(f"saving {process} histograms")
    hist_to_accumulate = []
    for out in output_files:
        hist_to_accumulate.append(load(out))
    output_histograms = {process: accumulate(hist_to_accumulate)}
    save(output_histograms, f"{output_dir}/{process}.coffea")

    cutflow = {}
    for category in categories:
        logging.info(f"saving cutflow for category {category}\n")
        category_dir = Path(f"{output_dir}/{category}")
        cut_to_accumulate = []
        df = pd.DataFrame()
        for sample in process_samples_map[process]:
            cutflow_file = category_dir / f"cutflow_{category}_{sample}.coffea"
            if cutflow_file.exists():
                df_sample = pd.DataFrame(load(cutflow_file).values())
                df = pd.concat([df, df_sample])

        cutflow_df = pd.DataFrame(df.sum())
        cutflow_df.columns = [process]
        cutflow_file = Path(f"{output_dir}/{category}/cutflow_{category}_{process}.csv")
        cutflow_df.to_csv(cutflow_file)


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