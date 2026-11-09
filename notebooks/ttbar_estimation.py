"""
TTBar Estimation Pipeline Script.

Executes Data/MC agreement, SR corrections, systematic comparison, 
and generates LaTeX tables with PNG compilations.
"""

import argparse
import os
import sys
from pathlib import Path
import yaml
import mplhep as hep

# Automatically detect project root and insert it into sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
if (SCRIPT_DIR / "analysis").exists():
    REPO_ROOT = SCRIPT_DIR
elif (SCRIPT_DIR.parent / "analysis").exists():
    REPO_ROOT = SCRIPT_DIR.parent
else:
    REPO_ROOT = Path.cwd()

os.chdir(REPO_ROOT)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from notebooks.utils import (
    apply_cms_text_patch,
    run_ttbar_estimation,
    generate_ttbar_tables_and_pngs,
    run_sr_ttbar_correction,
    run_ttbar_sr_poly_correction,
    generate_ttbar_corrected_tables_and_pngs,
    run_systematic_comparison,
    generate_systematic_diff_tables_and_pngs,
)


ALL_YEARS = ["2022preEE", "2022postEE", "2023preBPix", "2023postBPix", "2024"]
WORKFLOWS = ["2b1mu", "2b1e", "1b1mu1e", "1b1e1mu"]
PROCS = ["DYJetsToLL", "SingleTop", "Diboson", "WJetsToLNu", "tt"]
SR_TO_CRS = {
    "1b1mu": ["2b1mu", "1b1mu1e"],
    "1b1e": ["2b1e", "1b1e1mu"],
}


def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run ttbar estimation pipeline, plots, and LaTeX table generation."
    )
    parser.add_argument(
        "--year",
        "-y",
        nargs="+",
        choices=ALL_YEARS,
        default=ALL_YEARS,
        help="Specify one or more years to process (default: all years).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    years = args.year

    print("=== Starting TTBar Estimation Pipeline ===")
    print(f"Working directory set to: {REPO_ROOT}")
    print(f"Target years: {', '.join(years)}")

    # Apply CMS style
    apply_cms_text_patch()
    hep.style.use("CMS")

    # Load configuration files
    postprocess_dir = REPO_ROOT / "analysis" / "postprocess"
    lumi_path = postprocess_dir / "luminosity.yaml"
    color_map_path = postprocess_dir / "color_map.yaml"

    luminosities = {}
    if lumi_path.exists():
        with open(lumi_path, "r") as f:
            luminosities = yaml.safe_load(f)
    else:
        print(f"Warning: Luminosity file not found at {lumi_path}")

    color_map = {}
    if color_map_path.exists():
        with open(color_map_path, "r") as f:
            color_map = yaml.safe_load(f)
    else:
        print(f"Warning: Color map file not found at {color_map_path}")

    # Output directories
    plots_dir = REPO_ROOT / "notebooks/plots/ttbar_estimation"
    syst_dir = plots_dir / "syst"
    plots_dir.mkdir(parents=True, exist_ok=True)
    syst_dir.mkdir(parents=True, exist_ok=True)

    # 1. TTBar Data/MC Agreement Plots
    print("\n[Step 1/7] Generating TTBar Data/MC agreement plots...")
    run_ttbar_estimation(years, WORKFLOWS, luminosities, color_map, plots_dir)

    # 2. TTBar Data/MC Agreement Tables & PNGs
    print("\n[Step 2/7] Generating TTBar Data/MC agreement LaTeX tables & PNGs...")
    generate_ttbar_tables_and_pngs(WORKFLOWS, years, PROCS, plots_dir)

    # 3. TTBar Correction in SR (Bin-by-bin)
    print("\n[Step 3/7] Running TTBar SR bin-by-bin correction plots...")
    run_sr_ttbar_correction(SR_TO_CRS, years, luminosities, color_map, plots_dir)

    # 4. TTBar Correction in SR (Polynomial TF)
    print("\n[Step 4/7] Running TTBar SR polynomial transfer factor correction plots...")
    run_ttbar_sr_poly_correction(SR_TO_CRS, years, luminosities, plots_dir)

    # 5. TTBar Corrected MC Yield Tables & PNGs
    print("\n[Step 5/7] Generating TTBar corrected MC yield LaTeX tables & PNGs...")
    generate_ttbar_corrected_tables_and_pngs(SR_TO_CRS, years, PROCS, plots_dir)

    # 6. Systematic Comparison (Nominal vs Recalculated)
    print("\n[Step 6/7] Running systematic comparison plots (Nominal vs Recalculated)...")
    run_systematic_comparison(WORKFLOWS, years, luminosities, syst_dir)

    # 7. Systematic Difference Tables & PNGs
    print("\n[Step 7/7] Generating systematic difference LaTeX tables & PNGs...")
    generate_systematic_diff_tables_and_pngs(WORKFLOWS, years, syst_dir)

    print("\n=== TTBar Estimation Pipeline Finished Successfully ===")


if __name__ == "__main__":
    main()
