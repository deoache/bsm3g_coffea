import os
import glob
import hist
import yaml
import pickle
import logging
import numpy as np
import pandas as pd
import dask.dataframe as dd
from pathlib import Path
from collections import defaultdict
from coffea.util import load, save
from coffea.processor import accumulate
from analysis.filesets.utils import get_dataset_config


def setup_logger(output_dir):
    """Set up the logger to log to a file in the specified output directory."""
    output_file_path = os.path.join(output_dir, "output.txt")
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.FileHandler(output_file_path), logging.StreamHandler()],
    )


def open_output(fname: str) -> dict:
    with open(fname, "rb") as f:
        output = pickle.load(f)
    return output


def print_header(text):
    logging.info("-" * 90)
    logging.info(text)
    logging.info("-" * 90)


def divide_by_binwidth(histogram):
    bin_width = histogram.axes.edges[0][1:] - histogram.axes.edges[0][:-1]
    return histogram / bin_width


def clear_output_directory(output_dir, ext):
    """Delete all result files in the output directory with extension 'ext'"""
    files = glob.glob(os.path.join(output_dir, f"*.{ext}"))
    for file in files:
        os.remove(file)


def combine_event_tables(df1, df2, blind):
    df1 = df1.sort_index()
    df2 = df2.sort_index()
    assert all(df1.index == df2.index), "index does not match!"
    combined = pd.DataFrame(index=df1.index)
    combined["events"] = df1["events"] + df2["events"]
    combined["stat err"] = np.sqrt(df1["stat err"] ** 2 + df2["stat err"] ** 2)
    combined["syst err up"] = np.sqrt(df1["syst err up"] ** 2 + df2["syst err up"] ** 2)
    combined["syst err down"] = np.sqrt(
        df1["syst err down"] ** 2 + df2["syst err down"] ** 2
    )
    total_bkg = combined.loc["Total background", "events"]
    if not blind:
        data = combined.loc["Data", "events"]
        combined.loc[
            "Data/Total background",
            ["events", "stat err", "syst err up", "syst err down"],
        ] = [
            data / total_bkg,
            np.nan,
            np.nan,
            np.nan,
        ]
    return combined


def combine_cutflows(df1, df2):
    if not df1.index.equals(df2.index):
        raise ValueError("Los índices (etiquetas de cortes) no coinciden.")
    combined = df1.add(df2, fill_value=0)
    return combined


def df_to_latex_asymmetric(df, table_title="Events"):
    output = rf"""\begin{{table}}[h!]
\centering
\begin{{tabular}}{{@{{}} l c @{{}}}}
\hline
 & \textbf{{{table_title}}} \\
\hline
"""

    for label, row in df.iterrows():
        events = row["events"]
        stat_err = row.get("stat err", None)
        syst_err_up = row.get("syst err up", None)
        syst_err_down = row.get("syst err down", None)

        events_f = f"{float(events):.2f}"
        stat_err_f = f"{float(stat_err):.2f}" if pd.notna(stat_err) else "nan"
        syst_err_up_f = f"{float(syst_err_up):.2f}" if pd.notna(syst_err_up) else "nan"
        syst_err_down_f = (
            f"{float(syst_err_down):.2f}" if pd.notna(syst_err_down) else "nan"
        )

        if label not in ["Data", "Total background", "Data/Total background"]:
            output += (
                f"{label} & $\\displaystyle {events_f} \\pm {stat_err_f}"
                f"^{{\\scriptstyle +{syst_err_up_f}}}_{{\\scriptstyle -{syst_err_down_f}}}$\\\\\n"
            )

    # Total background
    bg = df.loc["Total background"]
    output += (
        f"Total Background & $\\displaystyle {bg['events']:.2f} \\pm {bg['stat err']:.2f}"
        f"^{{\\scriptstyle +{bg['syst err up']:.2f}}}_{{\\scriptstyle -{bg['syst err down']:.2f}}}$ \\\\\n"
    )

    # Data
    output += f"Data & ${float(df.loc['Data']['events']):.0f}$ \\\\\n"

    output += r"\hline" + "\n"

    # Ratio
    ratio = df.loc["Data"]["events"] / df.loc["Total background"]["events"]
    output += f"Data/Total Background & ${ratio:.2f}$ \\\\\n"

    output += r"""\hline
\end{tabular}
\end{table}"""
    return output


def df_to_latex_average(df, table_title="Events"):
    output = rf"""\begin{{table}}[h!]
\centering
\begin{{tabular}}{{@{{}} l c @{{}}}}
\hline
 & \textbf{{{table_title}}} \\
\hline
"""

    for label, row in df.iterrows():
        events = row["events"]
        stat_err = row.get("stat err", None)
        syst_err_up = row.get("syst err up", None)
        syst_err_down = row.get("syst err down", None)

        events_f = f"{float(events):.2f}"
        stat_err_f = f"{float(stat_err):.2f}" if pd.notna(stat_err) else "nan"
        if pd.notna(syst_err_up) and pd.notna(syst_err_down):
            syst_avg = (syst_err_up + syst_err_down) / 2
            syst_err_f = f"{syst_avg:.2f}"
        else:
            syst_err_f = "nan"

        if label not in ["Data", "Total background", "Data/Total background"]:
            output += (
                f"{label} & ${events_f} \\pm {stat_err_f} \\pm {syst_err_f} \\ $\\\\\n"
            )

    # Total background
    bg = df.loc["Total background"]
    syst_avg_bg = (bg["syst err up"] + bg["syst err down"]) / 2
    output += (
        f"Total Background & ${bg['events']:.2f} \\pm {bg['stat err']:.2f} \\ "
        f"\\pm {syst_avg_bg:.2f} \\$ \\\\\n"
    )

    # Data
    output += f"Data & ${float(df.loc['Data']['events']):.0f}$ \\\\\n"

    output += r"\hline" + "\n"

    # Ratio
    ratio = df.loc["Data"]["events"] / df.loc["Total background"]["events"]
    output += f"Data/Total Background & ${ratio:.2f}$ \\\\\n"

    output += r"""\hline
\end{tabular}
\end{table}"""
    return output


def get_variations_keys(processed_histograms: dict):
    variations = {}
    for process, histogram_dict in processed_histograms.items():
        if process == "Data":
            continue
        for feature in histogram_dict:
            helper_histogram = histogram_dict[feature]
            variations = [
                var for var in helper_histogram.axes["variation"] if var != "nominal"
            ]
            break
        break
    variations = list(
        set([var.replace("Up", "").replace("Down", "") for var in variations])
    )
    return variations


def uncertainty_table(processed_histograms, workflow):
    to_accumulate = []
    for process in processed_histograms:
        if process != "Data":
            to_accumulate.append(processed_histograms[process])
    helper_histo = accumulate(to_accumulate)
    if workflow in ["2b1e", "1b1e1mu", "1b1e"]:
        var = "electron_met_mass"
    elif workflow in ["2b1mu", "1b1mu1e", "1b1mu"]:
        var = "muon_met_mass"
    helper_histo = helper_histo["mass"].project(var, "variation")

    # get histogram per variation
    variation_hists = {}
    for variation in helper_histo.axes["variation"]:
        if variation == "nominal":
            nominal = helper_histo[{"variation": variation}]
        else:
            variation_hists[variation] = helper_histo[{"variation": variation}]

    # get variations names
    variations_keys = []
    for variation in variation_hists:
        if variation == "nominal":
            continue
        # get variation key
        variation_key = variation.replace("Up", "").replace("Down", "")
        if variation_key not in variations_keys:
            variations_keys.append(variation_key)

    variation_impact = {}
    nom = nominal.values()
    for variation in variations_keys:
        # up/down yields by bin
        varup = variation_hists[f"{variation}Up"].values()
        vardown = variation_hists[f"{variation}Down"].values()
        # concatenate σxup−nominal, σxdown−nominal, and 0
        up_and_down = np.stack([varup - nom, vardown - nom, np.zeros_like(nom)], axis=0)
        # max(σxup−nominal, σxdown−nominal, 0.) / nominal
        max_up_and_down = np.max(up_and_down, axis=0) / (nom + 1e-5)
        # min(σxup−nominal,σ xdown−nominal,0.) / nominal
        min_up_and_down = np.min(up_and_down, axis=0) / (nom + 1e-5)
        # integate over all bins
        variation_impact[variation] = [
            np.sqrt(np.sum(max_up_and_down**2)),
            np.sqrt(np.sum(min_up_and_down**2)),
        ]

    syst_df = pd.DataFrame(variation_impact).T * 100
    syst_df = syst_df.rename({0: "Up", 1: "Down"}, axis=1)
    return syst_df


def build_systematic_summary(processed_histograms, workflow="1b1mu"):
    """Compute systematic uncertainties for all processes and build a summary table"""
    summary_dict = {}

    for process_name, process_hist in processed_histograms.items():
        if process_name == "Data":
            continue

        if workflow in ["2b1e", "1b1e1mu", "1b1e"]:
            mass_variable = "electron_met_mass"
        elif workflow in ["2b1mu", "1b1mu1e", "1b1mu"]:
            mass_variable = "muon_met_mass"
        else:
            raise ValueError(f"Unknown workflow: {workflow}")

        # Project the histogram onto the transverse mass variable and variation axis
        projected_hist = process_hist["mass"].project(mass_variable, "variation")

        # Separate nominal and systematic variations
        variation_histograms = {}
        for variation_label in projected_hist.axes["variation"]:
            if variation_label == "nominal":
                nominal_hist = projected_hist[{"variation": variation_label}]
            else:
                variation_histograms[variation_label] = projected_hist[
                    {"variation": variation_label}
                ]

        # Collect systematic names (without Up/Down)
        systematic_names = []
        for variation_label in variation_histograms:
            sys_name = variation_label.replace("Up", "").replace("Down", "")
            if sys_name not in systematic_names:
                systematic_names.append(sys_name)

        # Compute relative Up/Down uncertainties
        nominal_values = nominal_hist.values()
        systematic_scales = {}
        for sys_name in systematic_names:
            values_up = variation_histograms[f"{sys_name}Up"].values()
            values_down = variation_histograms[f"{sys_name}Down"].values()

            # Compute deviations relative to nominal
            diffs = np.stack(
                [
                    values_up - nominal_values,
                    values_down - nominal_values,
                    np.zeros_like(nominal_values),
                ],
                axis=0,
            )

            max_dev = np.max(diffs, axis=0) / (nominal_values + 1e-5)
            min_dev = np.min(diffs, axis=0) / (nominal_values + 1e-5)

            # Quadrature sum across bins
            unc_up = np.sqrt(np.sum(max_dev**2))
            unc_down = np.sqrt(np.sum(min_dev**2))

            # Scale factor = 1 + max(%)/100
            scale_factor = 1 + max(unc_up, unc_down)
            systematic_scales[sys_name] = scale_factor

        summary_dict[process_name] = pd.Series(systematic_scales)

    # Build final DataFrame with systematics as rows, processes as columns
    summary_table = pd.DataFrame(summary_dict).fillna(1.0)

    return summary_table


def merge_parquets(inpath, outpath, sample_name):
    parquets = dd.read_parquet(
        f"{inpath}/*.parquet", engine="pyarrow", calculate_divisions=False
    )
    df = parquets.compute()
    outpath = Path(outpath)
    if not outpath.exists():
        outpath.mkdir(parents=True, exist_ok=True)
    df.to_parquet(f"{outpath}/{sample_name}.parquet", engine="pyarrow", index=False)


def accumulate_metadata(grouped_outputs, sample):
    grouped_metadata = {}
    for fname in grouped_outputs[sample]:
        output = load(fname)
        if not output:
            continue
        for meta_key, meta_val in output["metadata"].items():
            grouped_metadata.setdefault(meta_key, []).append(meta_val)
    return {k: accumulate(v) for k, v in grouped_metadata.items()}


def accumulate_histograms(grouped_outputs, sample):
    grouped_histograms = []
    for fname in grouped_outputs[sample]:
        output = load(fname)
        if output:
            grouped_histograms.append(output["histograms"])
    return accumulate(grouped_histograms)


def get_lumi_weight(year, sample, metadata):
    lumi_file = Path.cwd() / "analysis" / "postprocess" / "luminosity.yaml"
    with open(lumi_file, "r") as f:
        luminosities = yaml.safe_load(f)

    dataset_config = get_dataset_config(year)
    xsec = dataset_config[sample]["xsec"]
    sumw = metadata["sumw"]
    weight = 1
    if dataset_config[sample]["era"] in ["mc", "signal"]:
        weight = (luminosities[year] * xsec) / sumw

    logging.info(f"luminosity [1/pb]: {luminosities[year]}")
    logging.info(f"xsec [pb]: {xsec}")
    logging.info(f"sumw: {sumw}")
    logging.info(f"weight: {weight}")

    return weight


def save_cutflows(metadata, categories, sample, weight, output_dir):
    for category in categories:
        logging.info(f"saving {sample} cutflow for category {category}")

        category_dir = Path(output_dir) / category
        if not category_dir.exists():
            category_dir.mkdir(parents=True, exist_ok=True)

        scaled_cutflow = {
            cut: nevents * weight
            for cut, nevents in metadata[category]["cutflow"].items()
        }
        processed_cutflow = {sample: scaled_cutflow}
        cutflow_file = category_dir / f"cutflow_{category}_{sample}.coffea"
        save(processed_cutflow, cutflow_file)


def get_process_dict(output_dir, year, categories):
    folders = glob.glob(str(output_dir / "*"))
    process_dict = defaultdict(list)

    dataset_config = get_dataset_config(year)

    for folder in folders:
        folder_path = Path(folder)
        name = folder_path.name
        for category in categories:
            if name == category:
                continue
            if category not in process_dict:
                process_dict[category] = {}

            folder_content = glob.glob(f"{folder_path}/*")

            # case where there are multiple partitions and the folder includes only .coffea files
            if any(".coffea" in f for f in folder_content) and not any(
                category in f for f in folder_content
            ):
                if name not in process_dict[category]:
                    process_dict[category][name] = []
            # case where there is only one partition and the main folder includes both the .coffea file and the category folder
            elif any(".coffea" in f for f in folder_content) and any(
                category in f for f in folder_content
            ):
                for f in folder_content:
                    if category in f:
                        if name not in process_dict[category]:
                            process_dict[category][name] = [str(folder_path)]
            # case where there's only the category folder
            else:
                actual_name = name.rsplit("_", 1)[0]
                if actual_name in process_dict[category]:
                    process_dict[category][actual_name].append(str(folder_path))

    return dict(process_dict)


def merge_parquets_by_sample(output_dir, year, categories):
    """Merge parquet files from subfolders into a single output path per sample and category."""
    print_header("Merging parquet outputs by sample")
    process_dict = get_process_dict(output_dir, year, categories)
    for category in categories:
        for name, subfolders in process_dict[category].items():
            outpath = f"{output_dir}/parquets_{name}/{category}"
            if len(subfolders) != 0:
                logging.info(
                    f"Merging {name} outputs into {len(subfolders)} "
                    f"{'partition' if len(subfolders) == 1 else 'partitions'}"
                )
                for i, subfolder in enumerate(subfolders, start=1):
                    inpath = f"{subfolder}/{category}"
                    merge_parquets(inpath, outpath, f"{name}_{i}")


def accumulate_and_save_cutflows(process, process_samples_map, output_dir, categories):
    """Accumulate cutflows from all samples in a process and save them per category."""
    for category in categories:
        category_dir = Path(f"{output_dir}/{category}")
        df_total = pd.DataFrame()

        for sample in process_samples_map[process]:
            cutflow_file = category_dir / f"cutflow_{category}_{sample}.coffea"
            if cutflow_file.exists():
                df_sample = pd.DataFrame(load(cutflow_file).values())
                df_total = pd.concat([df_total, df_sample])

        if not df_total.empty:
            cutflow_df = pd.DataFrame(df_total.sum())
            cutflow_df.columns = [process]
            cutflow_csv = category_dir / f"cutflow_{category}_{process}.csv"
            logging.info(f"Saving cutflow for process {process}, category {category}")
            cutflow_df.to_csv(cutflow_csv)


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


def get_results_report(
    processed_histograms, workflow_config, category, columns_to_drop, blind
):
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
