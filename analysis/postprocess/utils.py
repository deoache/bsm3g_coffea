import os
import glob
import hist
import pickle
import logging
import numpy as np
import pandas as pd
from coffea.processor import accumulate
import matplotlib.text as mtext
import logging
from pathlib import Path

def setup_logger(output_dir):
    afs_log_dir = Path.cwd() / "logs"
    afs_log_dir.mkdir(parents=True, exist_ok=True)
    output_file_path = afs_log_dir / "output.txt"

    # Handler para el archivo (con timestamp y [INFO])
    file_handler = logging.FileHandler(str(output_file_path), mode="w")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

    # Handler para la pantalla (solo el mensaje de texto limpio)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(message)s"))

    # Configuración global del logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = [file_handler, console_handler]

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

def apply_cms_text_patch():
    """Applies the patch to raise the vertical position of the 'Preliminary' text."""
    if not hasattr(mtext.Text, "_original_init"):
        mtext.Text._original_init = mtext.Text.__init__

        def _patched_text_init(self, x=0, y=0, text="", **kwargs):
            if hasattr(y, "shape") or isinstance(y, (list, tuple)):
                y = float(np.ravel(y)[0])
            if hasattr(x, "shape") or isinstance(x, (list, tuple)):
                x = float(np.ravel(x)[0])
            if isinstance(text, str) and "Preliminary" in text:
                y += 0.015
            mtext.Text._original_init(self, x, y, text, **kwargs)

        mtext.Text.__init__ = _patched_text_init
