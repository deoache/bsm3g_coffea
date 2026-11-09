import os
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import argparse
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import yaml
from coffea import processor, util
from coffea.lookup_tools.dense_lookup import dense_lookup
from hist.intervals import ratio_uncertainty
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredText

from analysis.workflows.config import WorkflowConfigBuilder

user = os.getenv("USER") or os.getlogin()

hep.style.use(hep.style.CMS)
with open("analysis/postprocess/luminosity.yaml", "r") as f:
    luminosities = yaml.safe_load(f)
with open("analysis/postprocess/style.yaml", "r") as f:
    style = yaml.safe_load(f)
plt.rcParams.update(style["rcParams"])


def get_eff_sf(eff_hist, wp, eta_region, jet_flavor, category):
    flavor_tags = {"light": 0, "c": 1, "b": 2}
    selector = {"jet_flavor": flavor_tags[jet_flavor]}
    if eta_region == "barrel":
        selector.update({"jet_abseta": slice(0.0j, 1.3j, sum)})
    else:
        selector.update({"jet_abseta": slice(1.3j, 2.5j, sum)})

    selector_numerator = {**selector, f"jet_pass_{wp}_wp": True}
    selector_denominator = {**selector, f"jet_pass_{wp}_wp": sum}

    if "category" in eff_hist[f"eff_{wp}"].axes.name:
        selector_numerator.update({"category": category})
        selector_denominator.update({"category": category})

    numerator = eff_hist[f"eff_{wp}"][selector_numerator]
    denominator = eff_hist[f"eff_{wp}"][selector_denominator]
    efficiency = numerator / denominator

    centers = efficiency.axes[0].centers
    edges = efficiency.axes[0].edges
    xerr = edges[1:] - edges[:-1]
    yerr = ratio_uncertainty(
        num=numerator.values(),
        denom=denominator.values(),
        uncertainty_type="efficiency",
    )

    return efficiency.values(), centers, xerr, yerr


def main():
    parser = argparse.ArgumentParser(
        description="Script de cálculo post-procesamiento con Coffea."
    )
    parser.add_argument(
        "--workflow",
        type=str,
        required=True,
        help="Nombre del workflow a ejecutar",
    )
    parser.add_argument(
        "--year",
        type=str,
        required=True,
        help="Año del dataset a calcular (ej. 2018, 2017)",
    )

    args = parser.parse_args()

    workflow = args.workflow
    year = args.year

    print(f"Ejecutando workflow: {workflow} para el año: {year}")

    run = "2" if year.startswith("201") else "3"
    energy = "13" if run == '2' else "13.6"
    tagger_map = {"2": "deepJet", "3": "ParticleNet" if year != "2024" else "UPartAK4"}

    config_builder = WorkflowConfigBuilder(workflow=workflow)
    workflow_config = config_builder.build_workflow_config()
    categories = list(workflow_config.event_selection["categories"].keys())

    # Load processed histograms from btag_eff workflow
    processed_histograms = util.load(
        f"/eos/user/{user[0]}/{user}/bsm3g_coffea_outputs/{workflow}/{year}/{year}_processed_histograms.coffea"
    )
    btag_eff_hist = processor.accumulate(processed_histograms.values())

    for category in categories:
        for wp in ["loose", "medium", "tight"]:

            numerator_selector = {f"jet_pass_{wp}_wp": True}
            denominator_selector = {f"jet_pass_{wp}_wp": sum}
            if len(categories) > 1:
                numerator_selector.update({"category": category})
                denominator_selector.update({"category": category})
                
            numerator = btag_eff_hist[f"eff_{wp}"][numerator_selector]
            denominator = btag_eff_hist[f"eff_{wp}"][denominator_selector]
            efficiency = numerator / denominator
            efflookup = dense_lookup(efficiency.values(), [ax.edges for ax in efficiency.axes])
            efflookup._axes[-1] = np.array([0.0, 4.0, 5.0, 6.0])
        
            output_path = (
                f"analysis/data/btag_efficiencies/{workflow}_{wp}_{category}_{year}.coffea"
            )
            util.save(efflookup, output_path)
        
            # plot b-tagging efficiencies
            eff_barrel_light, centers, xerr_barrel_light, yerr_barrel_light = get_eff_sf(
                btag_eff_hist, wp=wp, eta_region="barrel", jet_flavor="light", category=category
            )
            eff_barrel_c, centers, xerr_barrel_c, yerr_barrel_c = get_eff_sf(
                btag_eff_hist, wp=wp, eta_region="barrel", jet_flavor="c", category=category
        )
        eff_barrel_b, centers, xerr_barrel_b, yerr_barrel_b = get_eff_sf(
            btag_eff_hist, wp=wp, eta_region="barrel", jet_flavor="b", category=category
        )
    
        eff_endcap_light, centers, xerr_endcap_light, yerr_endcap_light = get_eff_sf(
            btag_eff_hist, wp=wp, eta_region="endcap", jet_flavor="light", category=category
        )
        eff_endcap_c, centers, xerr_endcap_c, yerr_endcap_c = get_eff_sf(
            btag_eff_hist, wp=wp, eta_region="endcap", jet_flavor="c", category=category
        )
        eff_endcap_b, centers, xerr_endcap_b, yerr_endcap_b = get_eff_sf(
            btag_eff_hist, wp=wp, eta_region="endcap", jet_flavor="b", category=category
        )
    
        fig, ax = plt.subplots(figsize=(10, 9))
        errorbar_container1 = ax.errorbar(
            centers,
            eff_barrel_light,
            xerr=xerr_barrel_light / 2,
            yerr=yerr_barrel_light,
            fmt="ro",
            elinewidth=1,
            linestyle="solid",
            linewidth=0,
            markersize=6,
        )
        errorbar_container2 = ax.errorbar(
            centers,
            eff_barrel_c,
            xerr=xerr_barrel_c / 2,
            yerr=yerr_barrel_c,
            fmt="bo",
            elinewidth=1,
            linestyle="solid",
            linewidth=0,
            markersize=6,
        )
        errorbar_container3 = ax.errorbar(
            centers,
            eff_barrel_b,
            xerr=xerr_barrel_b / 2,
            yerr=yerr_barrel_b,
            fmt="go",
            elinewidth=1,
            linestyle="solid",
            linewidth=0,
            markersize=6,
        )
    
        errorbar_container4 = ax.errorbar(
            centers,
            eff_endcap_light,
            xerr=xerr_endcap_light / 2,
            yerr=yerr_endcap_light,
            fmt="rs",
            elinewidth=1,
            linewidth=0,
            markersize=6,
        )
        errorbar_container5 = ax.errorbar(
            centers,
            eff_endcap_c,
            xerr=xerr_endcap_c / 2,
            yerr=yerr_endcap_c,
            fmt="bs",
            elinewidth=1,
            linewidth=0,
            markersize=6,
        )
        errorbar_container6 = ax.errorbar(
            centers,
            eff_endcap_b,
            xerr=xerr_endcap_b / 2,
            yerr=yerr_endcap_b,
            fmt="gs",
            elinewidth=1,
            linewidth=0,
            markersize=6,
        )
        for bar in errorbar_container4[2]:
            bar.set_linestyle(":")
            bar.set_linewidth(2)
        for bar in errorbar_container5[2]:
            bar.set_linestyle(":")
            bar.set_linewidth(2)
        for bar in errorbar_container6[2]:
            bar.set_linestyle(":")
            bar.set_linewidth(2)
    
        legend_elements = [
            Line2D(
                [0],
                [0],
                color="g",
                linewidth=2,
                linestyle="-",
                marker="o",
                markersize=6,
                label="B (barrel)",
            ),
            Line2D(
                [0],
                [0],
                color="b",
                linewidth=2,
                linestyle="-",
                marker="o",
                markersize=6,
                label="C (barrel)",
            ),
            Line2D(
                [0],
                [0],
                color="r",
                linewidth=2,
                linestyle="-",
                marker="o",
                markersize=6,
                label="Light (barrel)",
            ),
            Line2D(
                [0],
                [0],
                color="g",
                linewidth=2,
                linestyle=":",
                marker="s",
                markersize=6,
                label="B (endcap)",
            ),
            Line2D(
                [0],
                [0],
                color="b",
                linewidth=2,
                linestyle=":",
                marker="s",
                markersize=6,
                label="C (endcap)",
            ),
            Line2D(
                [0],
                [0],
                color="r",
                linewidth=2,
                linestyle=":",
                marker="s",
                markersize=6,
                label="Light (endcap)",
            ),
        ]
        ax.legend(
            handles=legend_elements,
            loc="upper right",
            handlelength=3,
            handleheight=3,
            fontsize=15,
            ncols=2,
        )
        ax.set(
            ylim=(0, 1.4),
            ylabel="B-tagging efficiency",
            xlabel="Jet $p_T$ [GeV]",
        )
        at = AnchoredText(
            f"{workflow.replace('btag_eff_', '').replace('_', ' ')} ({category})\n{tagger_map[run]}\n{wp.capitalize()} WP",
            loc="upper left",
            frameon=False,
            prop={'size': 16},
        )
        at.txt._text.set_multialignment('center')
        ax.add_artist(at)
        hep.cms.label(
            ax=ax,
            lumi=round(luminosities[year] * 1e-3, 1),
            com=energy,
            data=True,
            year=year
        )

        ax.text(
            0.1, 1.00,
            r"$\mathit{Simulation}$", 
            verticalalignment="bottom", 
            horizontalalignment="left", 
            transform=ax.transAxes, 
            fontsize=18
        )
        fig.savefig(f"notebooks/plots/btag_eff/{workflow}_{wp}_{category}_{year}.pdf")

if __name__ == "__main__":
    main()