import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.text as mtext
import mplhep as hep
from pathlib import Path
import yaml
import hist
from analysis.filesets.utils import get_process_maps
from analysis.workflows.config import WorkflowConfigBuilder


class CoffeaPlotter:
    def __init__(self, workflow, year, processed_histograms, output_dir="plots"):
        self.workflow = workflow
        self.year = year
        self.processed_histograms = processed_histograms
        self.output_dir = Path(output_dir)
        builder = WorkflowConfigBuilder(workflow=workflow)
        self.config = builder.build_workflow_config()

        path = Path.cwd() / "analysis" / "postprocess"
        with open(path / "style.yaml") as f:
            self.style = yaml.safe_load(f)
        with open(path / "color_map.yaml") as f:
            cmap = yaml.safe_load(f)

        procs, _, _ = get_process_maps(self.config, year)
        self.color_map = {
            p: c
            for p, c in cmap.items()
            if p in procs and not p.startswith("Wpb")
        }
        self.color_map["QCD"] = "#FFC0CB"
        self.signal_color_map = {
            p: c for p, c in cmap.items() if p.startswith("Wpb")
        }

    def collect_and_project(self, main_key, var_axis, category):
        h_info = {
            "mc": {"hists": [], "labels": [], "colors": [], "variations": {}},
            "signal": {"hists": [], "labels": [], "colors": []},
            "data": None,
        }

        all_variations = set()
        for proc, h_dict in self.processed_histograms.items():
            h = h_dict.get(main_key)
            if h is not None and "variation" in h.axes.name:
                for var in h.axes["variation"]:
                    if var != "nominal":
                        all_variations.add(var)

        for proc, h_dict in self.processed_histograms.items():
            h = h_dict.get(main_key)
            if h is None:
                continue

            sel = {}
            if "variation" in h.axes.name:
                sel["variation"] = "nominal"
            if "category" in h.axes.name:
                sel["category"] = category

            temp_h = h[sel]
            if var_axis in temp_h.axes.name:
                temp_h = temp_h.project(var_axis)

            widths = temp_h.axes[0].widths
            vals = temp_h.values() / widths
            vars_ = temp_h.variances() / (widths**2)

            h_norm = hist.Hist(temp_h.axes[0], storage=hist.storage.Weight())
            h_norm.view()[:] = np.stack([vals, vars_], axis=-1)

            if proc == "Data":
                h_info["data"] = h_norm
            elif proc.startswith("Wpb"):
                h_info["signal"]["hists"].append(h_norm)
                h_info["signal"]["labels"].append(proc)
                h_info["signal"]["colors"].append(
                    self.signal_color_map.get(proc, "black")
                )
            else:
                h_info["mc"]["hists"].append(h_norm)
                h_info["mc"]["labels"].append(proc)
                h_info["mc"]["colors"].append(
                    self.color_map.get(proc, "gray")
                )

        for var in all_variations:
            accum_var_h = None
            for proc, h_dict in self.processed_histograms.items():
                if proc == "Data" or proc.startswith("Wpb"):
                    continue

                h = h_dict.get(main_key)
                if (
                    h is None
                    or "variation" not in h.axes.name
                    or var not in h.axes["variation"]
                ):
                    continue

                sel = {"variation": var}
                if "category" in h.axes.name:
                    sel["category"] = category

                temp_h = h[sel]
                if var_axis in temp_h.axes.name:
                    temp_h = temp_h.project(var_axis)

                widths = temp_h.axes[0].widths
                vals = temp_h.values() / widths
                vars_ = temp_h.variances() / (widths**2)

                h_norm = hist.Hist(
                    temp_h.axes[0], storage=hist.storage.Weight()
                )
                h_norm.view()[:] = np.stack([vals, vars_], axis=-1)

                if accum_var_h is None:
                    accum_var_h = h_norm
                else:
                    accum_var_h += h_norm

            if accum_var_h is not None:
                h_info["mc"]["variations"][var] = accum_var_h

        return h_info

    def plot_uncert_band(self, histogram_info, ax):
        safe_bins = self.nominal_values > 1e-3

        if histogram_info["mc"]["variations"]:
            mcstat_err2 = self.nominal_variances
            err2_up = mcstat_err2.copy()
            err2_down = mcstat_err2.copy()

            unique_vars = set(
                k[:-2]
                for k in histogram_info["mc"]["variations"].keys()
                if k.endswith("Up") or k.endswith("Down")
            )

            for variation in unique_vars:
                up_key = f"{variation}Up"
                down_key = f"{variation}Down"

                if (
                    up_key in histogram_info["mc"]["variations"]
                    and down_key in histogram_info["mc"]["variations"]
                ):
                    var_up = histogram_info["mc"]["variations"][up_key].values()
                    var_down = histogram_info["mc"]["variations"][
                        down_key
                    ].values()

                    err_up = var_up - self.nominal_values
                    err_down = var_down - self.nominal_values
                    max_delta = self.nominal_values
                    err_up = np.clip(err_up, -max_delta, max_delta)
                    err_down = np.clip(err_down, -max_delta, max_delta)

                    up_is_up = err_up > 0
                    down_is_down = err_down < 0
                    is_onesided = up_is_up ^ down_is_down

                    err2_up_twosided = np.where(
                        up_is_up, err_up**2, err_down**2
                    )
                    err2_down_twosided = np.where(
                        up_is_up, err_down**2, err_up**2
                    )
                    err2_max = np.maximum(err2_up_twosided, err2_down_twosided)

                    err2_up_onesided = np.where(
                        is_onesided & up_is_up, err2_max, 0
                    )
                    err2_down_onesided = np.where(
                        is_onesided & down_is_down, err2_max, 0
                    )

                    err2_up_combined = np.where(
                        is_onesided, err2_up_onesided, err2_up_twosided
                    )
                    err2_down_combined = np.where(
                        is_onesided, err2_down_onesided, err2_down_twosided
                    )

                    err2_up += err2_up_combined
                    err2_down += err2_down_combined
        else:
            err2_up = self.nominal_variances
            err2_down = self.nominal_variances

        self.band_up = self.nominal_values + np.sqrt(err2_up)

        sigma_down = np.sqrt(err2_down)
        self.band_down = np.where(
            sigma_down < self.nominal_values,
            self.nominal_values - sigma_down,
            self.nominal_values * 0.05,
        )
        self.band_down = np.maximum(self.band_down, 1e-2)

        band_args = self.style.get(
            "uncert_band_kwargs",
            {
                "label": "Stat + Syst unc.",
                "hatch": "//////",
                "facecolor": "none",
                "edgecolor": "black",
                "alpha": 0.4,
            },
        ).copy()

        ax.bar(
            x=self.centers[safe_bins],
            height=(self.band_up - self.band_down)[safe_bins],
            width=self.widths[safe_bins],
            bottom=self.band_down[safe_bins],
            **band_args,
        )

    def plot_histograms(
        self, main_key, var_axis, category, ax, luminosities, log=True
    ):
        plt.rcParams.update(self.style.get("rcParams", {}))

        h_info = self.collect_and_project(main_key, var_axis, category)
        mc_hists = h_info["mc"]["hists"]
        total_mc = sum(mc_hists)

        self.nominal_values = total_mc.values()
        self.nominal_variances = total_mc.variances()
        self.centers = total_mc.axes[0].centers
        self.widths = total_mc.axes[0].widths

        mc_stk_args = self.style.get("mc_hist_kwargs", {}).copy()
        mc_stk_args.pop("stack", None)
        mc_stk_args.pop("edgecolor", None)

        hep.histplot(
            mc_hists,
            label=h_info["mc"]["labels"],
            color=h_info["mc"]["colors"],
            stack=True,
            ax=ax,
            edgecolor="black",
            **mc_stk_args,
        )
        self.plot_uncert_band(h_info, ax)
        if h_info["signal"]["hists"]:
            for h_sig, label, color in zip(
                h_info["signal"]["hists"],
                h_info["signal"]["labels"],
                h_info["signal"]["colors"],
            ):
                hep.histplot(
                    h_sig,
                    label=label,
                    color=color,
                    histtype="step",
                    ax=ax,
                    linewidth=1.5,
                )
                ax.plot(
                    h_sig.axes[0].centers,
                    h_sig.values(),
                    marker="+",
                    color=color,
                    linestyle="None",
                    markersize=10,
                )
        if h_info["data"]:
            hep.histplot(
                h_info["data"],
                label="Data",
                ax=ax,
                histtype="errorbar",
                color="black",
                marker="o",
                markersize=6,
            )

        edges = total_mc.axes[0].edges
        if log:
            ax.set_yscale("log")
            ax.set_ylim(1e-1, 1e6)

        ax.set_xlim(edges[0], edges[-1])
        ax.tick_params(direction="in", top=True, right=True)

        ax.legend(
            ncol=2,
            fontsize=10,
            loc="upper right",
            frameon=True,
            edgecolor="black",
            labelspacing=0.3,
        )

        lumi_val = luminosities[self.year] * 1e-3
        hep.cms.label(
            "Preliminary",
            data=False,
            lumi=f"{lumi_val:.1f}",
            year=self.year,
            ax=ax,
            fontsize=11,
        )

        ax.set_ylabel("Events / GeV", fontsize=14)
        xlabel_str = (
            r"$m_T(\mu_2, p_T^{\mathrm{miss}})$ [GeV]"
            if "mu" in var_axis
            else r"$m_T(e_2, p_T^{\mathrm{miss}})$ [GeV]"
        )
        ax.set_xlabel(xlabel_str, fontsize=14)
        ax.text(
            0.05,
            0.90,
            f"{category}",
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
        )

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


STYLE_DATA_ERR = dict(
    fmt="o", color="black", markersize=4, capsize=0, elinewidth=1.0, zorder=3
)
STYLE_TEXT_BOX = dict(
    boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="none"
)

# Inside notebooks/utils/plotting.py, in the plot_panel function:

def plot_panel(
    ax_main,
    ax_ratio,
    data_data,
    mc_data,
    fit_data,
    meta_info,
    desired_order,
    color_map,
):
    """Renders the main and ratio panels for the ttbar estimation."""
    centers, edges, bin_widths, data_p_gev, data_err_p_gev = data_data
    mc_p_gev, mc_err_p_gev, y_min = mc_data
    accpt, yerr_accpt, xerr, ratio_up, ratio_down = fit_data
    x_fine, y_fine, best_degree, workflow, xlabel_str, lumi_val, com_val, year = meta_info

    # --- MAIN PANEL ---
    ax_main.errorbar(
        centers,
        data_p_gev,
        xerr=bin_widths / 2.0,
        yerr=data_err_p_gev,
        label=r"Data $-$ Non-$t\bar{t}$ MC",
        **STYLE_DATA_ERR,
    )
    ax_main.stairs(
        values=mc_p_gev,
        edges=edges,
        fill=True,
        color=color_map["tt"],
        edgecolor="black",
        linewidth=0.8,
        label=r"$t\bar{t}$ MC",
        zorder=1,
    )
    ax_main.stairs(
        baseline=np.maximum(mc_p_gev - mc_err_p_gev, y_min),
        values=np.maximum(mc_p_gev + mc_err_p_gev, y_min),
        edges=edges,
        facecolor="lightgray",
        edgecolor="black",
        hatch="////",
        linewidth=0,
        alpha=0.8,
        label=r"Stat + Syst unc.",
        zorder=2,
    )

    ax_main.set_ylabel("Events / GeV", fontsize=12)
    ax_main.set_yscale("log")
    ax_main.set_ylim(bottom=y_min, top=np.max(data_p_gev) * 20)
    ax_main.tick_params(axis="x", which="both", labelbottom=False, direction="in")
    ax_main.tick_params(axis="y", which="both", labelsize=9, direction="in")

    handles, labels = ax_main.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax_main.legend(
        [by_label[l] for l in desired_order if l in by_label],
        [l for l in desired_order if l in by_label],
        loc="upper right",
        fontsize=9,
        frameon=True,
        edgecolor="black",
    )

    ax_main.text(
        0.05,
        0.88,
        f"{workflow}",
        transform=ax_main.transAxes,
        fontsize=11,
        va="top",
        ha="left",
        bbox=STYLE_TEXT_BOX,
    )

    # --- RATIO PANEL ---
    ax_ratio.fill_between(
        edges,
        ratio_up,
        ratio_down,
        step="pre",
        color="lightgray",
        alpha=0.6,
        zorder=1,
    )
    ax_ratio.axhline(
        1.0, color="k", linestyle=":", linewidth=0.8, alpha=0.8, zorder=2
    )
    ax_ratio.errorbar(
        centers,
        accpt,
        xerr=xerr,
        yerr=yerr_accpt,
        label=r"Data / MC $t\bar{t}$",
        **STYLE_DATA_ERR,
    )

    # Draw the curve only if x_fine and y_fine exist (are not None)
    if x_fine is not None and y_fine is not None:
        ax_ratio.plot(
            x_fine,
            y_fine,
            color="red",
            ls="-",
            lw=1.5,
            zorder=4,
            label=f"Fit (degree {best_degree})",
        )

    ax_ratio.set_ylabel(
        r"$r_{t\bar{t}}=\frac{\mathrm{Data}-\mathrm{non}\,t\bar{t}\,\mathrm{MC}}{t\bar{t}\,\mathrm{MC}}$",
        fontsize=10,
    )
    ax_ratio.set_xlabel(xlabel_str, fontsize=12)
    ax_ratio.set_xlim(edges[0], edges[-1])
    ax_ratio.set_ylim(0.5, 1.5)

    ax_ratio.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4, prune="upper"))
    ax_ratio.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=True,
        right=True,
        labelsize=9,
    )

    hep.cms.label(
        "Preliminary",
        data=True,
        lumi=lumi_val,
        com=com_val,
        year=year,
        ax=ax_main,
        fontsize=11,
    )


def plot_tf_panel(ax, tf_data, fit_data, meta_info):
    """Renders the Transfer Factor panel for the QCD estimation."""
    centers, values, xerr, ratio_uncert, edges = tf_data
    x_fine, y_fine, best_degree = fit_data
    xlabel_str, lumi_val, com_val, year = meta_info

    ax.errorbar(
        centers,
        values,
        xerr=xerr,
        yerr=ratio_uncert,
        fmt="o",
        color="black",
        markersize=4,
        capsize=0,
        elinewidth=1,
        label="Discrete TF",
    )

    ax.plot(
        x_fine,
        y_fine,
        color="red",
        linestyle="-",
        linewidth=2,
        label=f"Fit (degree {best_degree})",
    )
    ax.axhline(0, color="black", linestyle="--", linewidth=1)

    ax.set_ylabel("Transfer Factor", fontsize=14)
    ax.set_xlabel(xlabel_str, fontsize=14)
    ax.set_xlim(edges[0], edges[-1])

    lower_points = values - ratio_uncert[0]
    upper_points = values + ratio_uncert[1]
    max_y = np.max(upper_points) if len(upper_points) > 0 else 1.0
    min_y = np.min(lower_points) if len(lower_points) > 0 else 0.0
    y_floor = min_y - (max_y * 0.05) if min_y > 0 else -0.05 * max_y
    ax.set_ylim(y_floor, max_y * 1.3)

    ax.tick_params(direction="in", top=True, right=True)
    ax.legend(loc="upper right", fontsize=10, frameon=True, edgecolor="black")

    hep.cms.label(
        "Preliminary",
        data=True,
        lumi=lumi_val,
        com=com_val,
        year=year,
        ax=ax,
        fontsize=11,
    )

def plot_qcd_panel(ax, h_norm, centers, yerr_poisson, xlabel, lumi_val, com_val, year):
    """Renders the estimated QCD events histogram and its uncertainty."""
    h_norm.plot1d(ax=ax, histtype="fill", color="#FFC0CB", edgecolor="black", linewidth=1.2)
    ax.errorbar(centers, h_norm.values(), yerr=yerr_poisson, fmt='none', color='black', elinewidth=1.2, capsize=0)
    
    ax.set_ylabel("QCD Events / GeV", fontsize=14)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_yscale("log")
    edges = h_norm.axes[0].edges
    ax.set_xlim(edges[0], edges[-1])
    max_val = np.max(h_norm.values()) if len(h_norm.values()) > 0 and np.max(h_norm.values()) > 0 else 1.0
    ax.set_ylim(1e-2, max_val * 5)
    ax.tick_params(direction="in", top=True, right=True)

    hep.cms.label(
        "Preliminary",
        data=True,
        lumi=lumi_val,
        com=com_val,
        year=year,
        ax=ax,
        fontsize=11,
    )

def plot_sr_corrected_panel(
    ax,
    corr_per_gev,
    orig_per_gev,
    err_corr_per_gev,
    edges_sr,
    y_min,
    xlabel_str,
    lumi_val,
    com_val,
    year,
    sr_workflow,
    cr_workflow,
    color_map,
):
    """Renders the ttbar panel in the signal region (SR), corrected by the control region (CR)."""
    ax.stairs(
        values=corr_per_gev,
        edges=edges_sr,
        baseline=y_min,
        fill=True,
        color=color_map.get("tt", "#FFC0CB"),
        edgecolor="black",
        linewidth=0.8,
        label=r"$t\bar{t}$ MC ($\times r_{t\bar{t}}$)",
        zorder=1,
    )

    ax.stairs(
        baseline=np.maximum(corr_per_gev - err_corr_per_gev, y_min),
        values=np.maximum(corr_per_gev + err_corr_per_gev, y_min),
        edges=edges_sr,
        facecolor="lightgray",
        edgecolor="black",
        hatch="////",
        linewidth=0,
        alpha=0.8,
        label=r"Stat + Syst unc.",
        zorder=2,
    )

    ax.stairs(
        values=orig_per_gev,
        edges=edges_sr,
        baseline=y_min,
        fill=False,
        color="black",
        linestyle="--",
        linewidth=1.8,
        label=r"$t\bar{t}$ uncorrected",
        zorder=3,
    )

    ax.set_ylabel("Events / GeV", fontsize=12)
    ax.set_xlabel(xlabel_str, fontsize=12)
    ax.set_xlim(edges_sr[0], edges_sr[-1])
    ax.set_yscale("log")

    max_y = max(
        np.max(corr_per_gev) if len(corr_per_gev) > 0 else 1,
        np.max(orig_per_gev) if len(orig_per_gev) > 0 else 1,
    )
    ax.set_ylim(bottom=y_min, top=max_y * 20)
    ax.tick_params(axis="both", which="both", direction="in", labelsize=9)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    desired_sr_order = [
        r"$t\bar{t}$ uncorrected",
        r"$t\bar{t}$ MC ($\times r_{t\bar{t}}$)",
        r"Stat + Syst unc.",
    ]
    ax.legend(
        [by_label[l] for l in desired_sr_order if l in by_label],
        [l for l in desired_sr_order if l in by_label],
        loc="upper right",
        fontsize=9,
        frameon=True,
        edgecolor="black",
    )

    STYLE_TEXT_BOX = dict(
        boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="none"
    )
    ax.text(
        0.05,
        0.88,
        f"SR: {sr_workflow}\nShape-corrected via CR: {cr_workflow}",
        transform=ax.transAxes,
        fontsize=10,
        va="top",
        bbox=STYLE_TEXT_BOX,
    )

    hep.cms.label(
        "Preliminary",
        data=False,
        lumi=lumi_val,
        com=com_val,
        year=year,
        ax=ax,
        fontsize=11,
    )

def plot_syst_panel(
    ax_top,
    ax_bot,
    diff_nom,
    diff_recalc,
    delta_data,
    err_delta,
    edges,
    centers,
    bin_widths,
    workflow,
    year,
    lumi_val,
):
    """Renders top (shapes) and bottom (difference) panels for systematic nominal vs recalculated comparison."""
    diff_nom_per_gev = diff_nom / bin_widths
    diff_recalc_per_gev = diff_recalc / bin_widths

    ax_top.stairs(
        diff_nom_per_gev,
        edges=edges,
        color="black",
        linewidth=1.5,
        label=r"Nominal ($\mathrm{Data} - \mathrm{non}\text{-}t\bar{t}$)",
    )
    ax_top.stairs(
        diff_recalc_per_gev,
        edges=edges,
        color="crimson",
        linestyle="--",
        linewidth=1.5,
        label=r"Recalculated ($\mathrm{Data} - \mathrm{non}\text{-}t\bar{t}$)",
    )
    ax_top.set_ylabel("Events / GeV", fontsize=11)
    ax_top.set_yscale("log")
    ax_top.legend(
        loc="upper right", fontsize=9, frameon=True, edgecolor="black"
    )

    pos_all = np.concatenate(
        [
            diff_nom_per_gev[diff_nom_per_gev > 0],
            diff_recalc_per_gev[diff_recalc_per_gev > 0],
        ]
    )
    y_min = np.min(pos_all) * 0.1 if len(pos_all) > 0 else 1e-4
    y_max = np.maximum(np.max(diff_nom_per_gev), np.max(diff_recalc_per_gev))

    ax_top.set_ylim(bottom=y_min, top=y_max * 20)

    lumi_kwargs = {"lumi": lumi_val} if lumi_val else {}
    hep.cms.label(
        "Preliminary", data=True, year=year, ax=ax_top, fontsize=10, **lumi_kwargs
    )

    STYLE_TEXT_BOX = dict(
        boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="none"
    )
    ax_top.text(
        0.05,
        0.88,
        f"{workflow}",
        transform=ax_top.transAxes,
        fontsize=10,
        bbox=STYLE_TEXT_BOX,
    )

    # Bottom Panel: Absolute Difference (Nominal - Recalculated)
    ax_bot.errorbar(
        centers,
        delta_data / bin_widths,
        xerr=bin_widths / 2.0,
        yerr=err_delta / bin_widths,
        fmt="o",
        color="purple",
        markersize=4,
        capsize=0,
        label=r"$\Delta = \text{Nominal} - \text{Recalculated}$",
    )
    ax_bot.axhline(0, color="gray", linestyle=":", linewidth=1)
    ax_bot.set_ylabel(r"$\Delta$ Events / GeV", fontsize=9)
    ax_bot.set_xlabel(r"$m_{\mathrm{T}}$ [GeV]", fontsize=11)
    ax_bot.set_xlim(edges[0], edges[-1])

    ax_top.tick_params(axis="x", labelbottom=False, direction="in")
    ax_top.tick_params(axis="y", labelsize=8, direction="in")
    ax_bot.tick_params(axis="both", labelsize=8, direction="in")