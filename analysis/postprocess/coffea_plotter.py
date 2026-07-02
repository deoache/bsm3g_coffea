import yaml
import logging
import numpy as np
import mplhep as hep
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib import ticker
from matplotlib.lines import Line2D
from coffea.processor import accumulate
from hist.intervals import poisson_interval
from matplotlib.offsetbox import AnchoredText
from analysis.filesets.utils import get_process_maps
from analysis.histograms import VariableAxis, IntegerAxis
from analysis.workflows.config import WorkflowConfigBuilder
from analysis.postprocess.utils import (
    setup_logger,
    divide_by_binwidth,
    get_variations_keys,
)

np.seterr(invalid="ignore")
np.seterr(divide="ignore")


class CoffeaPlotter:
    def __init__(
        self,
        workflow: str,
        year: str,
        processed_histograms: dict,
        output_dir: str,
    ):
        self.workflow = workflow
        self.year = year
        self.processed_histograms = processed_histograms
        self.output_dir = output_dir

        # get histogram config
        config_builder = WorkflowConfigBuilder(workflow=workflow)
        workflow_config = config_builder.build_workflow_config()
        self.histogram_config = workflow_config.histogram_config
        self.datasets = workflow_config.datasets

        # load luminosities, style and colors
        postprocess_dir = Path.cwd() / "analysis" / "postprocess"
        style_file = postprocess_dir / "style.yaml"
        luminosity_file = postprocess_dir / "luminosity.yaml"
        color_map_file = postprocess_dir / "color_map.yaml"

        with open(style_file, "r") as f:
            self.style = yaml.safe_load(f)
        with open(f"{Path.cwd()}/analysis/postprocess/luminosity.yaml", "r") as f:
            self.luminosities = yaml.safe_load(f)

        # set processes -> color and processes -> sample name maps
        processes, self.process_name_map, _ = get_process_maps(workflow_config, year)
        with open(color_map_file, "r") as f:
            color_map = yaml.safe_load(f)
        self.color_map = {p: c for p, c in color_map.items() if p in processes}

    def get_histogram(
        self,
        variable,
        variation,
        category,
        histogram,
    ):
        """returns histogram by processes/variable/category"""
        # get variable histogram for nominal variation and category
        selector = {}
        if "variation" in histogram.axes.name:
            selector.update({"variation": variation})
        if "category" in histogram.axes.name:
            selector["category"] = category
        histogram = histogram[selector].project(variable)
        # if axis type is variable divide by bin width
        if isinstance(self.histogram_config.axes[variable], VariableAxis):
            histogram = divide_by_binwidth(histogram)
        return histogram

    def get_variations(
        self,
        variable,
        category,
        variation,
        histogram,
    ):
        """returns variation histogram by processes/variable/category"""
        # get variable histogram for nominal variation and category
        selectorup = {"variation": f"{variation}Up"}
        selectordown = {"variation": f"{variation}Down"}
        if "category" in histogram.axes.name:
            selectorup["category"] = category
            selectordown["category"] = category
        histogram_up = histogram[selectorup].project(variable)
        histogram_down = histogram[selectordown].project(variable)
        # if axis type is variable divide by bin width
        if isinstance(self.histogram_config.axes[variable], VariableAxis):
            histogram_up = divide_by_binwidth(histogram_up)
            histogram_down = divide_by_binwidth(histogram_down)

        return histogram_up, histogram_down

    def collect_histograms_for_plotting(self, variable, category, blind):
        histogram_info = {}
        if "mc" in self.datasets:
            histogram_info["mc"] = {"nominal": {}, "variations": {}}
        if "signal" in self.datasets:
            histogram_info["signal"] = {"nominal": {}}

        for process, histogram_dict in self.processed_histograms.items():
            axis_exists = False
            if variable in histogram_dict:
                aux_histogram = histogram_dict[variable]
                axis_exists = True
            else:
                for key in histogram_dict:
                    if variable in histogram_dict[key].axes.name:
                        aux_histogram = histogram_dict[key]
                        axis_exists = True
                        break

            if not axis_exists:
                raise ValueError(
                    f"'{variable}' axis not found in workflow's output histogram"
                )

            if process == "Data":
                histogram_info["data"] = self.get_histogram(
                    variable=variable,
                    category=category,
                    variation="nominal",
                    histogram=aux_histogram,
                )
            else:
                key = (
                    "signal"
                    if self.process_name_map[process].startswith("signal")
                    else "mc"
                )
                histogram_info[key]["nominal"][process] = self.get_histogram(
                    variable=variable,
                    category=category,
                    variation="nominal",
                    histogram=aux_histogram,
                )
                # save variations histograms
                if (key == "mc") and ("variation" in aux_histogram.axes.name):
                    for variation in get_variations_keys(self.processed_histograms):
                        var_cats = [v for v in aux_histogram.axes["variation"]]
                        if not f"{variation}Up" in var_cats:
                            continue
                        up, down = self.get_variations(
                            variable=variable,
                            category=category,
                            variation=variation,
                            histogram=aux_histogram,
                        )
                        if f"{variation}Up" in histogram_info[key]["variations"]:
                            histogram_info[key]["variations"][f"{variation}Up"] += up
                            histogram_info[key]["variations"][f"{variation}Down"] += down
                        else:
                            histogram_info[key]["variations"][f"{variation}Up"] = up
                            histogram_info[key]["variations"][f"{variation}Down"] = down

        return histogram_info

    def plot_uncert_band(self, histogram_info, ax):
        if histogram_info["mc"]["variations"]:
            # initialize up/down errors with statisticall error
            mcstat_err2 = self.nominal_variances
            err2_up = mcstat_err2
            err2_down = mcstat_err2
            for variation in get_variations_keys(self.processed_histograms):
                # Up/down variations for a single MC sample
                var_up = histogram_info["mc"]["variations"][f"{variation}Up"].values()
                var_down = histogram_info["mc"]["variations"][
                    f"{variation}Down"
                ].values()
                # Compute the uncertainties corresponding to the up/down variations
                err_up = var_up - self.nominal_values
                err_down = var_down - self.nominal_values
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
                err2_up_combined = np.where(
                    is_onesided, err2_up_onesided, err2_up_twosided
                )
                err2_down_combined = np.where(
                    is_onesided, err2_down_onesided, err2_down_twosided
                )
                # Sum in quadrature of the systematic uncertainty corresponding to a MC sample
                err2_up += err2_up_combined
                err2_down += err2_down_combined
        else:
            err2_up = np.sqrt(self.nominal_values)
            err2_down = np.sqrt(self.nominal_values)
            self.style["uncert_band_kwargs"]["label"] = "Stat unc"

        self.band_up = self.nominal_values + np.sqrt(err2_up)
        self.band_down = self.nominal_values - np.sqrt(err2_down)
        # plot stat + syst uncertainty band
        ax.bar(
            x=self.centers,
            height=self.band_up - self.band_down,
            width=self.widths,
            bottom=self.band_down,
            **self.style["uncert_band_kwargs"],
        )

    def plot_ratio(self, rax):
        # compute Data/MC ratio
        num = self.data_values
        den = self.nominal_values
        ratio = num / den
        # only the uncertainty of num (DATA) propagated
        num_variances = self.data_variances
        ratio_variance = num_variances * np.power(den, -2)
        ratio_uncert = np.abs(poisson_interval(ratio, ratio_variance) - ratio)
        # plot ratio and x-y errors
        xerr = self.edges[1:] - self.edges[:-1]
        rax.errorbar(
            self.centers,
            ratio,
            xerr=xerr / 2,
            yerr=ratio_uncert,
            **self.style["ratio_error_kwargs"],
        )
        # plot ratio uncertainty band
        ratio_up = np.concatenate([[0], self.band_up / den])
        ratio_down = np.concatenate([[0], self.band_down / den])
        ratio_up[np.isnan(ratio_up)] = 1.0
        ratio_down[np.isnan(ratio_down)] = 1.0
        ratio_uncertainty_band = rax.fill_between(
            self.edges,
            ratio_up,
            ratio_down,
            step="pre",
            **self.style["uncert_band_kwargs"],
        )
        # plot horizontal reference line at 1
        xmin, xmax = rax.get_xlim()
        rax.hlines(1, xmin, xmax, color="k", linestyle=":")

    def plot_histograms(
        self,
        variable: str,
        category: str,
        yratio_limits: str = None,
        log: bool = False,
        add_ratio: bool = True,
        extension: str = "png",
        blind: bool = False,
    ):
        setup_logger(self.output_dir)
        # set plot params
        hep.style.use(hep.style.CMS)
        plt.rcParams.update(self.style["rcParams"])
        # get nominal MC histograms
        histogram_info = self.collect_histograms_for_plotting(variable, category, blind)

        nominal_mc_hists = list(histogram_info["mc"]["nominal"].values())
        mc_colors, mc_labels = [], []
        for process in histogram_info["mc"]["nominal"]:
            mc_labels.append(process)
            mc_colors.append(self.color_map[process])

        mc_histogram = accumulate(nominal_mc_hists)
        self.nominal_values = mc_histogram.values()
        self.nominal_variances = mc_histogram.variances()
        self.edges = mc_histogram.axes.edges[0]
        self.centers = mc_histogram.axes.centers[0]
        self.widths = mc_histogram.axes.widths[0]

        # get variation histograms
        variation_histograms = histogram_info["mc"]["variations"]

        # get Data histogram
        if not blind:
            data_histogram = histogram_info["data"]
            self.data_values = data_histogram.values()
            self.data_variances = data_histogram.variances()

        # plot stacked MC and Data histograms
        if blind:
            add_ratio = False

        fig, (ax, rax) = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=(9, 10),
            tight_layout=True,
            gridspec_kw={"height_ratios": (4, 1)},
            sharex=True,
        )
        if not add_ratio:
            fig, ax = plt.subplots(
                nrows=1,
                ncols=1,
                figsize=(9, 9),
                tight_layout=True,
            )
        hep.histplot(
            nominal_mc_hists,
            label=mc_labels,
            color=mc_colors,
            flow="none",
            ax=ax,
            **self.style["mc_hist_kwargs"],
        )
        if ("data" in self.datasets) or (not blind):
            hep.histplot(
                data_histogram,
                label="Data",
                flow="none",
                ax=ax,
                **self.style["data_hist_kwargs"],
            )
        if "signal" in self.datasets:
            for signal_process, signal_histogram in histogram_info["signal"][
                "nominal"
            ].items():
                hep.histplot(
                    signal_histogram,
                    label=signal_process,
                    color=self.color_map[signal_process],
                    flow="none",
                    ax=ax,
                    **self.style["signal_hist_kwargs"],
                )
        # plot uncertainty band
        self.plot_uncert_band(histogram_info, ax)
        if add_ratio:
            # plot ratio
            self.plot_ratio(rax)
        # set limits
        hist_edges = np.array([[i, j] for i, j in zip(self.edges[:-1], self.edges[1:])])
        xlimits = np.min(hist_edges[self.nominal_values > 0]), np.max(
            hist_edges[self.nominal_values > 0]
        )
        ax.set_xlim(xlimits)
        if add_ratio:
            rax.set_xlim(xlimits)
            rax.set_ylim(yratio_limits)
        # set axes labels
        ylabel = "Events"
        if isinstance(self.histogram_config.axes[variable], VariableAxis):
            ylabel += " / GeV"

        formatter = ticker.ScalarFormatter()
        formatter.set_scientific(False)
        ax.yaxis.set_major_formatter(formatter)

        xlabel = self.histogram_config.axes[variable].label
        if self.workflow in ["zplusl_os", "zplusl_ss"]:
            if category == "electron":
                xlabel = xlabel.replace(r"\ell", r"e")
            elif category == "muon":
                xlabel = xlabel.replace(r"\ell", r"\mu")

        if add_ratio:
            ax.set(xlabel=None, ylabel=ylabel)
            rax.set(
                xlabel=xlabel,
                ylabel="Data / Pred",
                facecolor="white",
            )
        else:
            ax.set(xlabel=xlabel, ylabel=ylabel)
        text_map = {
            "ztoee": r"$ Z \rightarrow ee$ events",
            "ztomumu": r"$ Z \rightarrow \mu\mu$ events",
            "ztojets": f"{category} selection",
        }
        at = AnchoredText(
            text_map.get(self.workflow, f"{self.workflow} events") + "\n",
            loc="upper left",
            frameon=False,
        )
        ax.add_artist(at)
        # set log scale
        if log:
            ax.set_yscale("log")
            ax.set_ylim(top=np.max(mc_histogram.values()) * 100)
        else:
            ylim = ax.get_ylim()[1]
            ax.set_ylim(0, ylim + 0.2 * ylim)
        ax.legend(
            frameon=True,
            loc="upper right",
            fontsize=13,
        )
        if isinstance(self.histogram_config.axes[variable], IntegerAxis):
            start = self.histogram_config.axes[variable].start
            stop = self.histogram_config.axes[variable].stop
            categories = np.arange(start, stop)
            if len(categories) > 20:
                axis_to = rax if add_ratio else ax
                for i, label in enumerate(axis_to.get_xticklabels()):
                    if i % 5 != 0:  # Show only every 5th tick
                        label.set_visible(False)
        # add CMS info
        energy = {"run2": "13 TeV", "run3": "13.6 TeV"}
        run_key = (
            "run3"
            if (self.year.startswith("2022") or self.year.startswith("2023"))
            else "run2"
        )
        hep.cms.lumitext(
            f"{self.luminosities[self.year] * 1e-3:.1f} fb$^{{-1}}$ ({self.year}, {energy[run_key]})",
            ax=ax,
        )
        hep.cms.text("Preliminary", ax=ax)
        # save histograms
        output_path = Path(f"{self.output_dir}/{category}")
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)
        figname = f"{str(output_path)}/{self.workflow}_{category}_{variable}_{self.year}.{extension}"
        fig.savefig(figname)
        plt.close()
