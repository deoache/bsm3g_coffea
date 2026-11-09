import os
from pathlib import Path
import yaml
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
from coffea.util import load
from coffea.processor import accumulate
from analysis.filesets.utils import get_process_maps
from analysis.workflows.config import WorkflowConfigBuilder
from .plotting import plot_qcd_panel, CoffeaPlotter
import matplotlib.gridspec as gridspec
from .fitting import poly_func, find_optimal_tf_poly
from .plotting import plot_panel
from .plotting import plot_sr_corrected_panel
from .plotting import plot_syst_panel


def load_hists(year, workflow):
    user = os.environ["USER"]
    out = load(
        f"/eos/user/{user[0]}/{user}/bsm3g_coffea_outputs/{workflow}/{year}/{year}_processed_histograms.coffea"
    )
    config_builder = WorkflowConfigBuilder(workflow=workflow)
    workflow_config = config_builder.build_workflow_config()
    postprocess_dir = Path.cwd() / "analysis" / "postprocess"
    style_file = postprocess_dir / "style.yaml"

    with open(style_file, "r") as f:
        style = yaml.safe_load(f)

    hep.style.use(hep.style.CMS)
    plt.rcParams.update(style["rcParams"])
    return out


def get_total_mc_uncertainty(h_mc, target_axis="muon_met_mass"):
    h_nom = h_mc[{"variation": "nominal"}].project(target_axis)
    nom_vals = h_nom.values()
    stat_vars = (
        h_nom.variances() if h_nom.variances() is not None else nom_vals
    )

    var_axis = h_mc.axes["variation"]
    all_variations = [var_axis.value(i) for i in range(len(var_axis))]

    sources = set()
    for v in all_variations:
        if v == "nominal":
            continue
        if v.endswith("Up"):
            sources.add(v[:-2])
        elif v.endswith("Down"):
            sources.add(v[:-4])

    syst_vars = np.zeros_like(nom_vals)

    for source in sources:
        up_name = f"{source}Up"
        down_name = f"{source}Down"

        if up_name in all_variations:
            val_up = h_mc[{"variation": up_name}].project(target_axis).values()
            diff_up = np.abs(val_up - nom_vals)
        else:
            diff_up = np.zeros_like(nom_vals)

        if down_name in all_variations:
            val_down = (
                h_mc[{"variation": down_name}].project(target_axis).values()
            )
            diff_down = np.abs(val_down - nom_vals)
        else:
            diff_down = np.zeros_like(nom_vals)

        max_diff = np.maximum(diff_up, diff_down)
        syst_vars += max_diff**2

    total_vars = stat_vars + syst_vars
    return nom_vals, total_vars


def compute_tf_for_year(year_str, workflow):
    out_tf = load_hists(year_str, workflow)
    procs_tf = ["DYJetsToLL", "SingleTop", "tt", "Diboson"]

    mc_tf_all = accumulate(
        [out_tf[p]["transverse_mass"] for p in procs_tf if p in out_tf]
    )
    h_tf_base = out_tf["Data"]["transverse_mass"] + (mc_tf_all * -1)

    if "ele" in workflow:
        target_axis = "electron_met_mass_subleading"
        cat_num, cat_den = "1LnT1T", "1LnT1nT"
    else:
        target_axis = "muon_met_mass_subleading"
        cat_num, cat_den = "2T", "1T1LnT"

    h_Tight = h_tf_base[{"variation": "nominal", "category": cat_num}].project(
        target_axis
    )
    h_nTight = h_tf_base[{"variation": "nominal", "category": cat_den}].project(
        target_axis
    )

    return h_Tight, h_nTight

import hist
from coffea.processor import accumulate
from hist.intervals import poisson_interval
from .plotting import plot_qcd_panel

WORKFLOWS_QCD_CONFIG = {
    "qcd_tf_ele": {
        "qcd_workflow": "qcd_ele",
        "variable_qcd": "electron_met_mass",
        "xlabel": r"$m_T(e_2, p_T^{\mathrm{miss}})$ [GeV]",
        "cat_num": "1LnT1T",
        "cat_den": "1LnT1nT",
    },
    "qcd_tf_mu": {
        "qcd_workflow": "qcd_mu",
        "variable_qcd": "muon_met_mass",
        "xlabel": r"$m_T(\mu_2, p_T^{\mathrm{miss}})$ [GeV]",
        "cat_num": "2T",
        "cat_den": "1T1LnT",
    },
}

def run_qcd_estimation(tf_store, years, luminosities, plots_dir=None):
    if plots_dir is None:
        plots_dir = Path("notebooks/plots/qcd_estimation")
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    energy = {"2": "13", "3": "13.6"}
    procs_qcd = ["DYJetsToLL", "SingleTop", "tt", "Diboson", "WJetsToLNu"]
    procs_tf = ["DYJetsToLL", "SingleTop", "tt", "Diboson"]

    qcd_store = {}

    # Calculate grid layout dynamically based on the number of years
    n_years = len(years)
    ncols = 2
    nrows = (n_years + ncols - 1) // ncols

    for tf_workflow, config in WORKFLOWS_QCD_CONFIG.items():
        qcd_workflow = config["qcd_workflow"]
        variable_qcd = config["variable_qcd"]
        xlabel = config["xlabel"]
        cat_num = config["cat_num"]
        cat_den = config["cat_den"]

        fig, axes = plt.subplots(
            nrows, ncols, figsize=(14, 5 * nrows), sharex=False, sharey=False
        )
        axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
        qcd_store[qcd_workflow] = {}

        for idx, year in enumerate(years):
            run_key = "2" if year.startswith("201") else "3"
            ax_grid = axes_flat[idx]

            # 1. QCD control region
            out_qcd = load_hists(year, qcd_workflow)
            mc_qcd_all = accumulate(
                [out_qcd[p]["mass"] for p in procs_qcd if p in out_qcd]
            )
            h_qcd_base = out_qcd["Data"]["mass"] + (mc_qcd_all * -1)
            h_qcd_cr = h_qcd_base[{"variation": "nominal"}].project(variable_qcd)

            # 2. Discrete Transfer Factor
            out_tf = load_hists(year, tf_workflow)
            available_mcs_tf = [
                out_tf[p]["transverse_mass"]
                for p in procs_tf
                if p in out_tf and "transverse_mass" in out_tf[p]
            ]
            h_tf_base = out_tf["Data"]["transverse_mass"] + (
                accumulate(available_mcs_tf) * -1
            )
            axis_to_project = [
                ax.name
                for ax in h_tf_base.axes
                if ax.name not in ["variation", "category"]
            ][0]

            h_Tight = h_tf_base[
                {"variation": "nominal", "category": cat_num}
            ].project(axis_to_project)
            h_nTight = h_tf_base[
                {"variation": "nominal", "category": cat_den}
            ].project(axis_to_project)

            num_vals = h_Tight.values()
            den_vals = h_nTight.values() + 1e-32
            tf_discrete = num_vals / den_vals

            # 3. Hybrid TF
            tf_fit = tf_store[tf_workflow][year]
            tf_effective = np.where(tf_discrete < 0.0, tf_fit, tf_discrete)

            # 4. Scale the control region by the hybrid TF
            h_qcd = h_qcd_cr.copy()
            h_qcd.view()[:] = np.stack(
                [
                    h_qcd_cr.values() * tf_effective,
                    h_qcd_cr.variances() * (tf_effective**2),
                ],
                axis=-1,
            )

            qcd_store[qcd_workflow][year] = h_qcd.copy()

            # 5. Normalize by bin width and uncertainties
            widths = h_qcd.axes[0].widths
            edges = h_qcd.axes[0].edges
            centers = 0.5 * (edges[:-1] + edges[1:])

            var_num = (
                h_Tight.variances()
                if h_Tight.variances() is not None
                else np.abs(num_vals)
            )
            tf_variances = var_num * np.power(den_vals, -2)
            tf_uncert = np.abs(
                poisson_interval(np.maximum(tf_effective, 1e-5), tf_variances)
                - np.maximum(tf_effective, 1e-5)
            )

            qcd_cr_pos = np.maximum(h_qcd_cr.values(), 0.0)
            err_down_normalized = (tf_uncert[0] * qcd_cr_pos) / widths
            err_up_normalized = (tf_uncert[1] * qcd_cr_pos) / widths

            yerr_poisson = np.maximum(
                np.array([err_down_normalized, err_up_normalized]), 0.0
            )
            yerr_poisson = np.nan_to_num(
                yerr_poisson, nan=0.0, posinf=0.0, neginf=0.0
            )

            h_norm = hist.Hist(h_qcd.axes[0], storage=hist.storage.Weight())
            h_norm.view()[:] = np.stack(
                [h_qcd.values() / widths, h_qcd.variances() / (widths**2)],
                axis=-1,
            )

            lumi_val = float(round(luminosities[year] * 1e-3, 1))
            com_val = float(energy[run_key].split()[0])

            # Combined panel plot in grid
            plot_qcd_panel(
                ax_grid,
                h_norm,
                centers,
                yerr_poisson,
                xlabel,
                lumi_val,
                com_val,
                year,
            )

            # Individual panel plot
            fig_ind, ax_ind = plt.subplots(figsize=(8, 6))
            plot_qcd_panel(
                ax_ind,
                h_norm,
                centers,
                yerr_poisson,
                xlabel,
                lumi_val,
                com_val,
                year,
            )
            fig_ind.tight_layout()
            fig_ind.savefig(
                plots_dir / f"qcd_estimation_{qcd_workflow}_{year}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig_ind)

        # Remove any empty remaining axes (e.g., cell 6 on a 3x2 grid when processing 5 years)
        for j in range(len(years), len(axes_flat)):
            fig.delaxes(axes_flat[j])

        # Combined grid figure per workflow
        fig.tight_layout()
        fig.savefig(
            plots_dir / f"qcd_estimation_{qcd_workflow}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

    return qcd_store

SIGNAL_WORKFLOWS_CONFIG = {
    "1b1e": {
        "variable_axis": "electron_met_mass",
        "category_signal": "1b1e",
        "qcd_key": "qcd_ele",
    },
    "1b1mu": {
        "variable_axis": "muon_met_mass",
        "category_signal": "1b1mu",
        "qcd_key": "qcd_mu",
    },
}


def plot_qcd_injected_signals(qcd_store, years, luminosities, plots_dir=None):
    if plots_dir is None:
        plots_dir = Path("notebooks/plots/qcd_estimation")
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    main_key = "mass"

    n_years = len(years)
    ncols = 2
    nrows = (n_years + ncols - 1) // ncols

    for sig_workflow, config in SIGNAL_WORKFLOWS_CONFIG.items():
        variable_axis = config["variable_axis"]
        category_signal = config["category_signal"]
        qcd_key = config["qcd_key"]

        fig, axes = plt.subplots(
            nrows, ncols, figsize=(16, 6 * nrows), sharex=False, sharey=False
        )
        axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

        for idx, year in enumerate(years):
            ax_grid = axes_flat[idx]

            # 1. Load signal histograms
            out_signal = load_hists(year, sig_workflow)
            any_proc = next(p for p in out_signal.keys() if p != "Data")
            h_template_full = out_signal[any_proc][main_key]

            selector = {}
            if "variation" in h_template_full.axes.name:
                selector["variation"] = "nominal"
            if "category" in h_template_full.axes.name:
                selector["category"] = category_signal

            h_template_1d = h_template_full[selector]
            if variable_axis in h_template_1d.axes.name:
                h_template_1d = h_template_1d.project(variable_axis)

            h_qcd_inj = hist.Hist(
                *h_template_1d.axes, storage=hist.storage.Weight()
            )
            h_qcd_year = qcd_store[qcd_key][year]
            h_qcd_inj.values()[:] = h_qcd_year.values()[:]
            h_qcd_inj.variances()[:] = h_qcd_year.variances()[:]

            axes_to_add = []
            if "variation" in h_template_full.axes.name:
                axes_to_add.append(
                    hist.axis.StrCategory(["nominal"], name="variation")
                )
            if "category" in h_template_full.axes.name:
                axes_to_add.append(
                    hist.axis.StrCategory([category_signal], name="category")
                )

            if axes_to_add:
                h_final_qcd = hist.Hist(
                    *axes_to_add,
                    *h_qcd_inj.axes,
                    storage=hist.storage.Weight(),
                )
                h_final_qcd.view()[:] = h_qcd_inj.view()
            else:
                h_final_qcd = h_qcd_inj

            out_signal["QCD"] = {main_key: h_final_qcd}

            # 2. Draw panel in grid
            plotter_grid = CoffeaPlotter(
                workflow=sig_workflow,
                year=year,
                processed_histograms=out_signal,
                output_dir=str(plots_dir),
            )
            plotter_grid.plot_histograms(
                main_key,
                variable_axis,
                category_signal,
                ax=ax_grid,
                luminosities=luminosities,
                log=True,
            )

            # 3. Draw individual figure
            fig_ind, ax_ind = plt.subplots(figsize=(8, 7))
            plotter_ind = CoffeaPlotter(
                workflow=sig_workflow,
                year=year,
                processed_histograms=out_signal,
                output_dir=str(plots_dir),
            )
            plotter_ind.plot_histograms(
                main_key,
                variable_axis,
                category_signal,
                ax=ax_ind,
                luminosities=luminosities,
                log=True,
            )
            fig_ind.tight_layout()
            fig_ind.savefig(
                plots_dir / f"qcd_injected_{sig_workflow}_{year}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig_ind)

        # Clean unused axes
        for j in range(len(years), len(axes_flat)):
            fig.delaxes(axes_flat[j])

        fig.tight_layout()
        fig.savefig(
            plots_dir / f"qcd_injected_{sig_workflow}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

def run_ttbar_estimation(
    years, workflows, luminosities, color_map, plots_dir=None
):
    if plots_dir is None:
        plots_dir = Path("notebooks/plots/ttbar_estimation")
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    procs = ["DYJetsToLL", "SingleTop", "Diboson", "WJetsToLNu"]
    energy = {"2": "13", "3": "13.6"}
    desired_order = [
        r"Data $-$ Non-$t\bar{t}$ MC",
        r"$t\bar{t}$ MC",
        r"Stat + Syst unc.",
    ]

    n_years = len(years)
    ncols = 2
    nrows = (n_years + ncols - 1) // ncols

    for workflow in workflows:
        fig = plt.figure(figsize=(18, 6 * nrows))
        outer_grid = gridspec.GridSpec(
            nrows, ncols, figure=fig, wspace=0.15, hspace=0.25
        )
        channel = (
            "muon" if "mu" in workflow and workflow != "1b1e1mu" else "electron"
        )

        for idx, year in enumerate(years):
            run_key = "2" if year.startswith("201") else "3"
            out = load_hists(year, workflow)

            mc_non_tt = accumulate([out[p]["mass"] for p in procs if p in out])
            mc_tt, data = out["tt"]["mass"], out["Data"]["mass"]
            target_axis = f"{channel}_met_mass"

            edges = (
                mc_tt[{"variation": "nominal"}]
                .project(target_axis)
                .axes[0]
                .edges
            )
            centers = 0.5 * (edges[:-1] + edges[1:])
            bin_widths = np.diff(edges)
            xerr = bin_widths / 2.0

            h_data_nom = data[{"variation": "nominal"}].project(target_axis)
            data_vals = h_data_nom.values()
            var_D_stat = (
                h_data_nom.variances()
                if h_data_nom.variances() is not None
                else data_vals
            )

            mc_non_tt_vals, var_non_tt_total = get_total_mc_uncertainty(
                mc_non_tt, target_axis=target_axis
            )
            mc_tt_vals, var_tt_total = get_total_mc_uncertainty(
                mc_tt, target_axis=target_axis
            )

            D = data_vals - mc_non_tt_vals
            var_D = var_D_stat + var_non_tt_total
            data_errs_raw = np.sqrt(np.maximum(var_D, 0))

            M = mc_tt_vals + 1e-32
            accpt = D / M
            accpt_var = (var_D / (M**2)) + ((D**2 * var_tt_total) / (M**4))
            yerr_accpt = np.sqrt(np.maximum(accpt_var, 0))

            mc_vals_per_gev = mc_tt_vals / bin_widths
            mc_errs_per_gev = np.sqrt(np.maximum(var_tt_total, 0)) / bin_widths
            data_vals_per_gev = D / bin_widths
            data_errs_per_gev = data_errs_raw / bin_widths

            total_mc_pred = mc_tt_vals + mc_non_tt_vals + 1e-32
            rel_mc_uncert = (
                np.sqrt(np.maximum(var_tt_total + var_non_tt_total, 0))
                / total_mc_pred
            )
            ratio_up = np.nan_to_num(
                np.concatenate([[0], 1.0 + rel_mc_uncert]), nan=1.0
            )
            ratio_down = np.nan_to_num(
                np.concatenate([[0], 1.0 - rel_mc_uncert]), nan=1.0
            )

            pos_all = np.concatenate(
                [
                    mc_vals_per_gev[mc_vals_per_gev > 0],
                    data_vals_per_gev[data_vals_per_gev > 0],
                ]
            )
            y_min = np.min(pos_all) * 0.1 if len(pos_all) > 0 else 1e-4

            labels_map = {
                "muon": r"$m_T(\mu, p_T^{\mathrm{miss}})$ [GeV]",
                "electron": r"$m_T(e, p_T^{\mathrm{miss}})$ [GeV]",
            }
            xlabel_str = labels_map.get(channel, r"$m(e, \mu)$ [GeV]")

            lumi_val = float(round(luminosities[year] * 1e-3, 1))
            com_val = float(energy[run_key].split()[0])

            data_pack = (
                centers,
                edges,
                bin_widths,
                data_vals_per_gev,
                data_errs_per_gev,
            )
            mc_pack = (mc_vals_per_gev, mc_errs_per_gev, y_min)
            fit_pack = (accpt, yerr_accpt, xerr, ratio_up, ratio_down)
            
            meta_pack = (
                None,
                None,
                None,
                workflow,
                xlabel_str,
                lumi_val,
                com_val,
                year,
            )

            # Combined panel plot in grid
            inner_grid = gridspec.GridSpecFromSubplotSpec(
                2,
                1,
                subplot_spec=outer_grid[idx],
                height_ratios=[3.5, 1],
                hspace=0.08,
            )
            ax_main = fig.add_subplot(inner_grid[0])
            ax_ratio = fig.add_subplot(inner_grid[1], sharex=ax_main)

            plot_panel(
                ax_main,
                ax_ratio,
                data_pack,
                mc_pack,
                fit_pack,
                meta_pack,
                desired_order,
                color_map,
            )
            fig.align_ylabels([ax_main, ax_ratio])

            # Individual panel plot
            fig_ind, (ax_main_ind, ax_ratio_ind) = plt.subplots(
                2,
                1,
                figsize=(9, 7),
                sharex=True,
                gridspec_kw={"height_ratios": [3.5, 1], "hspace": 0.08},
            )
            plot_panel(
                ax_main_ind,
                ax_ratio_ind,
                data_pack,
                mc_pack,
                fit_pack,
                meta_pack,
                desired_order,
                color_map,
            )
            fig_ind.align_ylabels([ax_main_ind, ax_ratio_ind])

            fig_ind.savefig(
                plots_dir / f"ttbar_agreement_{workflow}_{year}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig_ind)

        fig.savefig(
            plots_dir / f"ttbar_agreement_{workflow}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

def run_sr_ttbar_correction(
    sr_to_cr, years, luminosities, color_map, plots_dir=None
):
    if plots_dir is None:
        plots_dir = Path("notebooks/plots/ttbar_estimation")
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    procs = ["DYJetsToLL", "SingleTop", "Diboson", "WJetsToLNu"]
    energy = {"2": "13", "3": "13.6"}

    n_years = len(years)
    ncols = 2
    nrows = (n_years + ncols - 1) // ncols

    for sr_workflow, cr_list in sr_to_cr.items():
        for cr_workflow in cr_list:
            fig = plt.figure(figsize=(18, 6 * nrows))
            outer_grid = gridspec.GridSpec(
                nrows, ncols, figure=fig, wspace=0.20, hspace=0.25
            )

            channel = "muon" if "mu" in sr_workflow else "electron"
            target_axis_sr = f"{channel}_met_mass"
            target_axis_cr = f"{channel}_met_mass"

            for idx, year in enumerate(years):
                run_key = "2" if year.startswith("201") else "3"

                out_cr = load_hists(year, cr_workflow)
                mc_non_tt_cr = accumulate(
                    [out_cr[p]["mass"] for p in procs if p in out_cr]
                )
                data_cr_nom = out_cr["Data"]["mass"][
                    {"variation": "nominal"}
                ].project(target_axis_cr)

                edges_cr = data_cr_nom.axes[0].edges
                centers_cr = 0.5 * (edges_cr[:-1] + edges_cr[1:])

                data_vals_cr = data_cr_nom.values()
                vals_non_tt_cr, var_non_tt_cr = get_total_mc_uncertainty(
                    mc_non_tt_cr, target_axis=target_axis_cr
                )
                vals_tt_cr, var_tt_cr = get_total_mc_uncertainty(
                    out_cr["tt"]["mass"], target_axis=target_axis_cr
                )

                D_cr = data_vals_cr - vals_non_tt_cr
                M_cr = vals_tt_cr

                valid_cr = (
                    (M_cr > 1e-5)
                    & ~np.isnan(D_cr)
                    & ~np.isinf(D_cr)
                    & (D_cr > 0)
                )

                centers_cr_clean = centers_cr[valid_cr]
                mu_bin_cr_clean = D_cr[valid_cr] / M_cr[valid_cr]

                var_D_cr = (
                    data_cr_nom.variances()
                    if data_cr_nom.variances() is not None
                    else data_vals_cr
                ) + var_non_tt_cr

                var_mu_cr = np.zeros_like(D_cr)
                var_mu_cr[valid_cr] = (
                    var_D_cr[valid_cr] / (M_cr[valid_cr] ** 2)
                ) + (
                    (D_cr[valid_cr] ** 2 * var_tt_cr[valid_cr])
                    / (M_cr[valid_cr] ** 4)
                )
                err_mu_cr_clean = np.sqrt(np.maximum(var_mu_cr[valid_cr], 0))

                out_sr = load_hists(year, sr_workflow)
                h_sr_tt = out_sr["tt"]["mass"]

                edges_sr = (
                    h_sr_tt[{"variation": "nominal"}]
                    .project(target_axis_sr)
                    .axes[0]
                    .edges
                )
                centers_sr = 0.5 * (edges_sr[:-1] + edges_sr[1:])
                bin_widths_sr = np.diff(edges_sr)

                mc_sr_vals, var_sr_total = get_total_mc_uncertainty(
                    h_sr_tt, target_axis=target_axis_sr
                )

                if len(centers_cr_clean) < 2:
                    sf_bin_by_bin = np.ones_like(centers_sr)
                    sf_err_bin_by_bin = np.zeros_like(centers_sr)
                else:
                    sf_bin_by_bin = np.interp(
                        centers_sr, centers_cr_clean, mu_bin_cr_clean
                    )
                    sf_err_bin_by_bin = np.interp(
                        centers_sr, centers_cr_clean, err_mu_cr_clean
                    )

                mc_sr_corr_vals = mc_sr_vals * sf_bin_by_bin

                var_sr_corr_total = (var_sr_total * (sf_bin_by_bin**2)) + (
                    (mc_sr_vals**2) * (sf_err_bin_by_bin**2)
                )
                err_sr_corr_total = np.sqrt(np.maximum(var_sr_corr_total, 0))

                orig_per_gev = mc_sr_vals / bin_widths_sr
                corr_per_gev = mc_sr_corr_vals / bin_widths_sr
                err_corr_per_gev = err_sr_corr_total / bin_widths_sr

                pos_vals = corr_per_gev[corr_per_gev > 0]
                y_min = np.min(pos_vals) * 0.1 if len(pos_vals) > 0 else 1e-4

                labels_map = {
                    "muon": r"$m_T(\mu, p_T^{\mathrm{miss}})$ [GeV]",
                    "electron": r"$m_T(e, p_T^{\mathrm{miss}})$ [GeV]",
                }
                xlabel_str = labels_map.get(channel, r"$m_T$ [GeV]")
                lumi_val = float(round(luminosities[year] * 1e-3, 1))
                com_val = float(energy[run_key].split()[0])

                # Combined grid panel
                ax_main = fig.add_subplot(outer_grid[idx])
                plot_sr_corrected_panel(
                    ax_main,
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
                )

                # Individual panel plot
                fig_ind, ax_m_ind = plt.subplots(figsize=(9, 6))
                plot_sr_corrected_panel(
                    ax_m_ind,
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
                )
                fig_ind.savefig(
                    plots_dir
                    / f"sr_ttbar_corrected_{sr_workflow}_via_{cr_workflow}_{year}.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close(fig_ind)

            fig.savefig(
                plots_dir
                / f"sr_ttbar_corrected_{sr_workflow}_via_{cr_workflow}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)

def run_ttbar_sr_poly_correction(
    sr_to_crs, years, luminosities, plots_dir=None
):
    """Fits a polynomial to the ttbar data/MC ratio in the CR, scales the ttbar histogram in the SR and produces the plots."""
    if plots_dir is None:
        plots_dir = Path("notebooks/plots/ttbar_estimation")
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    procs = ["DYJetsToLL", "SingleTop", "Diboson", "WJetsToLNu"]
    main_key = "mass"

    for sr_workflow, cr_workflows in sr_to_crs.items():
        channel = "muon" if "mu" in sr_workflow else "electron"
        variable_axis = f"{channel}_met_mass"
        category_signal = f"{sr_workflow} events"

        for cr_workflow in cr_workflows:
            for year in years:
                fig, ax = plt.subplots(figsize=(8, 7))

                out_cr = load_hists(year, cr_workflow)
                mc_non_tt_cr = accumulate(
                    [
                        out_cr[p][main_key]
                        for p in procs
                        if p in out_cr and p != "tt"
                    ]
                )
                data_cr_nom = out_cr["Data"][main_key][
                    {"variation": "nominal"}
                ].project(variable_axis)

                edges_cr = data_cr_nom.axes[0].edges
                centers_cr = 0.5 * (edges_cr[:-1] + edges_cr[1:])

                data_vals_cr = data_cr_nom.values()
                vals_non_tt_cr, var_non_tt_cr = get_total_mc_uncertainty(
                    mc_non_tt_cr, target_axis=variable_axis
                )
                vals_tt_cr, var_tt_cr = get_total_mc_uncertainty(
                    out_cr["tt"][main_key], target_axis=variable_axis
                )

                D_cr = data_vals_cr - vals_non_tt_cr
                M_cr = vals_tt_cr + 1e-32
                accpt_cr = D_cr / M_cr

                var_D_cr = (
                    data_cr_nom.variances()
                    if data_cr_nom.variances() is not None
                    else data_vals_cr
                ) + var_non_tt_cr
                accpt_var_cr = (var_D_cr / (M_cr**2)) + (
                    (D_cr**2 * var_tt_cr) / (M_cr**4)
                )
                yerr_accpt_cr = np.sqrt(np.maximum(accpt_var_cr, 0))

                valid_mask = (
                    (M_cr > 1e-5)
                    & (yerr_accpt_cr > 0)
                    & ~np.isnan(accpt_cr)
                    & ~np.isinf(accpt_cr)
                )

                best_degree, optimal_coefs, _ = find_optimal_tf_poly(
                    centers_cr[valid_mask],
                    accpt_cr[valid_mask],
                    yerr_accpt_cr[valid_mask],
                    max_degree=5,
                    alpha=0.05,
                )

                out_signal = load_hists(year, sr_workflow)
                h_tt_sr = out_signal["tt"][main_key]

                proj_tmp = (
                    h_tt_sr[{"variation": "nominal"}]
                    if "variation" in h_tt_sr.axes.name
                    else h_tt_sr
                )
                if "category" in proj_tmp.axes.name:
                    cat_val = (
                        category_signal
                        if category_signal in proj_tmp.axes["category"]
                        else 0
                    )
                    proj_tmp = proj_tmp[{"category": cat_val}]
                proj_tmp = proj_tmp.project(variable_axis)

                centers_sr = proj_tmp.axes[0].centers

                sf_bin = np.polyval(optimal_coefs, centers_sr)
                sf_bin = np.clip(sf_bin, 0.01, 5.0)

                axis_idx = h_tt_sr.axes.name.index(variable_axis)
                slicer = [slice(None)] * len(h_tt_sr.axes)

                reshape_pattern = [1] * len(h_tt_sr.axes)
                reshape_pattern[axis_idx] = len(sf_bin)
                sf_reshaped = sf_bin.reshape(reshape_pattern)

                view = h_tt_sr.view(flow=False)
                view["value"][tuple(slicer)] *= sf_reshaped
                view["variance"][tuple(slicer)] *= sf_reshaped**2

                plotter = CoffeaPlotter(
                    workflow=sr_workflow,
                    year=year,
                    processed_histograms=out_signal,
                    output_dir=str(plots_dir),
                )
                plotter.plot_histograms(
                    main_key,
                    variable_axis,
                    category_signal,
                    ax=ax,
                    luminosities=luminosities,
                    log=True,
                )

                save_path = (
                    plots_dir
                    / f"{sr_workflow}_{channel}_met_mass_ttbar_corrected_via_{cr_workflow}_{year}.png"
                )
                plt.tight_layout()
                plt.savefig(save_path, dpi=300)
                plt.close(fig)

def run_systematic_comparison(workflows, years, luminosities=None, output_dir=None):
    if output_dir is None:
        output_dir = Path("notebooks/plots/ttbar_estimation/syst")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    procs_non_tt = ["DYJetsToLL", "SingleTop", "Diboson", "WJetsToLNu"]
    config_mapping = {
        "2b1mu": ("muon_met_mass", "muon_met_mass_recalculated"),
        "2b1e": ("electron_met_mass", "electron_met_mass_recalculated"),
        "1b1mu1e": ("muon_met_mass", "muon_met_mass_recalculated"),
        "1b1e1mu": ("electron_met_mass", "electron_met_mass_recalculated"),
    }

    if luminosities is None:
        luminosities = {}

    n_years = len(years)
    ncols = 2
    nrows = (n_years + ncols - 1) // ncols

    for workflow in workflows:
        var_nom, var_recalc = config_mapping[workflow]

        fig_grid = plt.figure(figsize=(18, 6 * nrows))
        outer_grid = gridspec.GridSpec(
            nrows, ncols, figure=fig_grid, wspace=0.15, hspace=0.25
        )

        for idx, year in enumerate(years):
            lumi_val = (
                float(round(luminosities[year] * 1e-3, 1))
                if year in luminosities
                else None
            )

            file_path = Path(
                f"/eos/user/d/driosper/bsm3g_coffea_outputs/{workflow}/{year}/{year}_processed_histograms.coffea"
            )
            if not file_path.exists():
                print(f"Skipping {workflow} - {year} (file not found)")
                continue

            out = load(file_path)

            # 1. Extract Nominal distributions
            h_data_nom = out["Data"]["mass"]
            h_mc_non_tt_nom = accumulate(
                [out[p]["mass"] for p in procs_non_tt if p in out]
            )

            d_vals_nom = (
                h_data_nom[{"variation": "nominal"}].project(var_nom).values()
            )
            mc_non_tt_vals_nom = (
                h_mc_non_tt_nom[{"variation": "nominal"}]
                .project(var_nom)
                .values()
            )
            diff_nom = d_vals_nom - mc_non_tt_vals_nom

            v_d_nom = (
                h_data_nom[{"variation": "nominal"}].project(var_nom).variances()
            )
            v_mc_nom = (
                h_mc_non_tt_nom[{"variation": "nominal"}]
                .project(var_nom)
                .variances()
            )
            v_d_nom = v_d_nom if v_d_nom is not None else d_vals_nom
            v_mc_nom = v_mc_nom if v_mc_nom is not None else mc_non_tt_vals_nom
            var_tot_nom = v_d_nom + v_mc_nom

            # 2. Extract Recalculated distributions
            h_data_recalc = out["Data"]["mass_recalculated"]
            h_mc_non_tt_recalc = accumulate(
                [out[p]["mass_recalculated"] for p in procs_non_tt if p in out]
            )

            d_vals_recalc = (
                h_data_recalc[{"variation": "nominal"}]
                .project(var_recalc)
                .values()
            )
            mc_non_tt_vals_recalc = (
                h_mc_non_tt_recalc[{"variation": "nominal"}]
                .project(var_recalc)
                .values()
            )
            diff_recalc = d_vals_recalc - mc_non_tt_vals_recalc

            v_d_recalc = (
                h_data_recalc[{"variation": "nominal"}]
                .project(var_recalc)
                .variances()
            )
            v_mc_recalc = (
                h_mc_non_tt_recalc[{"variation": "nominal"}]
                .project(var_recalc)
                .variances()
            )
            v_d_recalc = (
                v_d_recalc if v_d_recalc is not None else d_vals_recalc
            )
            v_mc_recalc = (
                v_mc_recalc if v_mc_recalc is not None else mc_non_tt_vals_recalc
            )
            var_tot_recalc = v_d_recalc + v_mc_recalc

            # 3. Compute difference and statistical uncertainties
            delta_data = diff_nom - diff_recalc
            err_delta = np.sqrt(np.maximum(var_tot_nom + var_tot_recalc, 0))

            edges = (
                h_data_nom[{"variation": "nominal"}]
                .project(var_nom)
                .axes[0]
                .edges
            )
            centers = 0.5 * (edges[:-1] + edges[1:])
            bin_widths = np.diff(edges)

            # Combined grid panel
            inner_grid = gridspec.GridSpecFromSubplotSpec(
                2,
                1,
                subplot_spec=outer_grid[idx],
                height_ratios=[2.5, 1],
                hspace=0.08,
            )
            ax_main_grid = fig_grid.add_subplot(inner_grid[0])
            ax_bot_grid = fig_grid.add_subplot(
                inner_grid[1], sharex=ax_main_grid
            )

            plot_syst_panel(
                ax_main_grid,
                ax_bot_grid,
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
            )
            fig_grid.align_ylabels([ax_main_grid, ax_bot_grid])

            # Individual panel plot
            fig_ind, (ax_main_ind, ax_bot_ind) = plt.subplots(
                2,
                1,
                figsize=(9, 7),
                sharex=True,
                gridspec_kw={"height_ratios": [2.5, 1], "hspace": 0.08},
            )
            plot_syst_panel(
                ax_main_ind,
                ax_bot_ind,
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
            )
            fig_ind.align_ylabels([ax_main_ind, ax_bot_ind])
            fig_ind.savefig(
                output_dir / f"systematic_diff_{workflow}_{year}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig_ind)

        fig_grid.savefig(
            output_dir / f"systematic_diff_grid_{workflow}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig_grid)