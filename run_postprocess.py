import gc
import sys
import yaml
import glob
import json
import logging
import argparse
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from coffea.util import save, load
from coffea.processor import accumulate
from analysis.filesets.utils import get_workflow_key_process_map, get_process_sample_map
from analysis.workflows.config import WorkflowConfigBuilder
from analysis.postprocess.plotter import CoffeaPlotter
from analysis.postprocess.postprocessor import (
    save_histograms_by_sample,
    save_histograms_by_process,
)
from analysis.postprocess.utils import (
    print_header,
    setup_logger,
    clear_output_directory,
    df_to_latex_average,
    df_to_latex_asymmetric,
    combine_event_tables,
    combine_cutflows,
    uncertainty_table,
    build_systematic_summary,
    load_processed_histograms,
    get_results_report,
    merge_parquets_by_sample,
)


OUTPUT_DIR = Path.cwd() / "outputs"


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run postprocessing and plotting for specified workflow and year."
    )
    parser.add_argument(
        "-w",
        "--workflow",
        required=True,
        choices=[
            f.stem for f in (Path.cwd() / "analysis" / "workflows").glob("*.yaml")
        ],
        help="Workflow config to run",
    )
    parser.add_argument(
        "-y",
        "--year",
        required=True,
        choices=[
            "2016",
            "2016preVFP",
            "2016postVFP",
            "2017",
            "2018",
            "2022",
            "2023",
            "2022preEE",
            "2022postEE",
            "2023preBPix",
            "2023postBPix",
        ],
        help="Dataset year",
    )
    parser.add_argument(
        "--log", action="store_true", help="Enable log scale for y-axis"
    )
    parser.add_argument(
        "--yratio_limits",
        type=float,
        nargs=2,
        default=(0.5, 1.5),
        help="Set y-axis ratio limits (e.g., --yratio_limits 0 2)",
    )
    parser.add_argument(
        "--postprocess", action="store_true", help="Enable postprocessing"
    )
    parser.add_argument("--plot", action="store_true", help="Enable plotting")
    parser.add_argument(
        "--extension",
        type=str,
        default="pdf",
        choices=["pdf", "png"],
        help="File extension for plots",
    )
    parser.add_argument("--no_ratio", action="store_true", help="Enable postprocessing")
    parser.add_argument("--blind", action="store_true", help="Blind data")
    parser.add_argument(
        "--group_by",
        type=str,
        default="process",
        help="Axis to group by (e.g., 'process', or a JSON dict)",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="coffea",
        choices=["coffea", "parquet"],
        help="Format of output files",
    )
    parser.add_argument(
        "--skipmerging", action="store_true", help="Skip parquet outputs merging"
    )
    parser.add_argument(
        "--nocutflow", action="store_true", help="Enable postprocessing"
    )
    parser.add_argument(
        "--include_weights",
        type=str,
        default="all",
        help="weights to include when filling histograms, split by ','. Specifying `all` will run through all variables.",
    )
    parser.add_argument(
        "--exclude_weights",
        type=str,
        default="",
        help="weights to exclude when filling histograms, split by ','",
    )
    return parser.parse_args()


def check_output_dir(workflow: str, year: str) -> Path:
    """
    Verify that the output directory exists for the given workflow and year.
    - For years 2022 and 2023, both pre/post sub-year directories must exist
      before creating the parent directory.
    - Returns the valid Path if successful.
    - Raises FileNotFoundError if required directories are missing.
    """

    output_dir = OUTPUT_DIR / workflow / year

    if output_dir.exists():
        return output_dir

    # Years that require both pre and post subdirectories
    aux_map = {
        "2016": ["2016preVFP", "2016postVFP"],
        "2022": ["2022preEE", "2022postEE"],
        "2023": ["2023preBPix", "2023postBPix"],
    }

    if year in aux_map:
        pre_year, post_year = [OUTPUT_DIR / workflow / y for y in aux_map[year]]

        # Collect missing subdirectories
        missing = [str(p) for p in (pre_year, post_year) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing required directories for year {year}: {', '.join(missing)}"
            )

        # Create the parent directory if both pre and post exist
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    # Case for sub-years or any other invalid year
    raise FileNotFoundError(f"Could not find outputs at {output_dir}")


def get_sample_name(filename: str, year: str) -> str:
    """return sample name from filename"""
    sample_name = Path(filename).stem
    if sample_name.rsplit("_")[-1].isdigit():
        sample_name = "_".join(sample_name.rsplit("_")[:-1])
    return sample_name.replace(f"{year}_", "")


def load_year_histograms(workflow: str, year: str):
    """load and merge histograms from pre/post campaigns"""
    aux_map = {
        "2016": ["2016preVFP", "2016postVFP"],
        "2022": ["2022preEE", "2022postEE"],
        "2023": ["2023preBPix", "2023postBPix"],
    }
    pre_year, post_year = aux_map[year]
    base_path = OUTPUT_DIR / workflow
    pre_file = base_path / pre_year / f"{pre_year}_processed_histograms.coffea"
    post_file = base_path / post_year / f"{post_year}_processed_histograms.coffea"
    return accumulate([load(pre_file), load(post_file)])


def load_histogram_file(path: Path):
    return load(path) if path.exists() else None


def plot_variable(variable: str, group_by, histogram_config) -> bool:
    """decide whether to plot a given variable under group_by mode"""
    if isinstance(group_by, str) and group_by == "process":
        return True
    for hist_key, variables in histogram_config.layout.items():
        if variable in variables and group_by["name"] in variables:
            return group_by["name"] != variable
    return False


if __name__ == "__main__":
    args = parse_arguments()

    try:
        group_by = json.loads(args.group_by)
    except json.JSONDecodeError:
        group_by = args.group_by

    output_dir = check_output_dir(args.workflow, args.year)
    clear_output_directory(output_dir, "txt")
    setup_logger(output_dir)

    config_builder = WorkflowConfigBuilder(workflow=args.workflow)
    workflow_config = config_builder.build_workflow_config()
    histogram_config = workflow_config.histogram_config
    event_selection = workflow_config.event_selection
    categories = event_selection["categories"]
    processed_histograms = None

    if "data" not in workflow_config.datasets:
        args.blind = True

    if args.postprocess and (args.year not in ["2016", "2022", "2023"]):
        print_header(f"Running postprocess for {args.year}")
        print_header(f"Reading outputs from: {output_dir}")

        output_files = [
            f
            for f in glob.glob(f"{output_dir}/*/*.coffea", recursive=True)
            if not Path(f).stem.startswith("cutflow")
            and not Path(f).stem.startswith("processed")
        ]

        grouped_outputs = defaultdict(list)
        for output_file in output_files:
            sample_name = get_sample_name(output_file, args.year)
            grouped_outputs[sample_name].append(output_file)

        process_samples_map = get_process_sample_map(grouped_outputs.keys(), args.year)

        if args.output_format == "parquet":
            if not args.skipmerging:
                merge_parquets_by_sample(output_dir, args.year, categories)

        for sample in grouped_outputs:
            if sample in ["TWminusto4Q", "TbarWplusto4Q"]:
                continue
            save_histograms_by_sample(
                grouped_outputs,
                sample,
                args.year,
                output_dir,
                categories,
                workflow_config,
                args.nocutflow,
                args.output_format,
                args.skipmerging,
                args.include_weights,
                args.exclude_weights,
            )
            gc.collect()

        for process in process_samples_map:
            save_histograms_by_process(
                process,
                output_dir,
                process_samples_map,
                categories,
                args.nocutflow,
                args.output_format,
            )
            gc.collect()

        processed_histograms = load_processed_histograms(
            args.year,
            output_dir,
            process_samples_map,
        )

        for category in categories:
            logging.info(f"category: {category}")
            category_dir = Path(f"{output_dir}/{category}")

            if not args.nocutflow:
                print_header(f"Cutflow")
                cutflow_df = pd.DataFrame()
                for process in process_samples_map:
                    cutflow_file = category_dir / f"cutflow_{category}_{process}.csv"
                    cutflow_df = pd.concat(
                        [cutflow_df, pd.read_csv(cutflow_file, index_col=[0])], axis=1
                    )

                columns_to_drop = []
                key_process_map = get_workflow_key_process_map(
                    workflow_config, args.year
                )
                if "signal" in workflow_config.datasets:
                    signal_keys = [k for k in workflow_config.datasets["signal"]]
                    signals = [key_process_map[key] for key in signal_keys]
                    columns_to_drop += signals

                if not args.blind:
                    columns_to_drop += ["Data"]

                total_background = cutflow_df.drop(columns=columns_to_drop).sum(axis=1)
                cutflow_df["Total Background"] = total_background

                cutflow_index = event_selection["categories"][category]
                cutflow_df = cutflow_df.loc[cutflow_index]

                if not args.blind:
                    to_process = ["Data", "Total Background"]
                else:
                    to_process = ["Total Background"]
                cutflow_df = cutflow_df[
                    to_process
                    + [
                        process
                        for process in cutflow_df.columns
                        if process not in to_process
                    ]
                ]
                logging.info(
                    f'{cutflow_df.applymap(lambda x: f"{x:.3f}" if pd.notnull(x) else "")}\n'
                )
                cutflow_df.to_csv(f"{category_dir}/cutflow_{category}.csv")
                logging.info("\n")

            print_header(f"Results")
            results_df = get_results_report(
                processed_histograms,
                workflow_config,
                category,
                columns_to_drop,
                args.blind,
            )
            logging.info(
                results_df.applymap(lambda x: f"{x:.5f}" if pd.notnull(x) else "")
            )
            logging.info("\n")
            results_df.to_csv(f"{category_dir}/results_{category}.csv")

            if not args.blind:
                latex_table_asymmetric = df_to_latex_asymmetric(results_df)
                with open(
                    category_dir / f"results_{category}_asymmetric.txt", "w"
                ) as f:
                    f.write(latex_table_asymmetric)
                latex_table_average = df_to_latex_average(results_df)
                with open(category_dir / f"results_{category}_average.txt", "w") as f:
                    f.write(latex_table_average)

        if args.workflow in ["1b1mu", "1b1e", "2b1e", "2b1mu", "1b1mu1e", "1b1e1mu"]:
            print_header(f"Systematic uncertainty impact")
            syst_df = uncertainty_table(processed_histograms, args.workflow)
            syst_df.to_csv(f"{output_dir}/uncertainty_table.csv")
            logging.info(syst_df)
            logging.info("\n")

            print_header(f"Systematic uncertainty impact by process")
            summary_table = build_systematic_summary(
                processed_histograms, args.workflow
            )
            summary_table.to_csv(f"{output_dir}/uncertainty_table_by_process.csv")
            logging.info(summary_table)
            logging.info("\n")

    if args.year in ["2016", "2022", "2023"]:
        if args.postprocess:
            print_header(f"Running postprocess for {args.year}")
            # load and accumulate processed 2016preVFP and 2016postVFP histograms
            processed_histograms = load_year_histograms(args.workflow, args.year)
            save(
                processed_histograms,
                f"{output_dir}/{args.year}_processed_histograms.coffea",
            )
            identifier_map = {"2016": "VFP", "2022": "EE", "2023": "BPix"}
            identifier = identifier_map[args.year]

            if args.workflow in [
                "2b1e",
                "2b1mu",
                "1b1mu1e",
                "1b1e1mu",
                "1b1e",
                "1b1mu",
            ]:
                print_header(f"Systematic uncertainty impact")
                syst_df = uncertainty_table(processed_histograms, args.workflow)
                syst_df.to_csv(
                    f"{OUTPUT_DIR / args.workflow / args.year}/uncertainty_table.csv"
                )
                logging.info(syst_df)
                logging.info("\n")

            for category in categories:
                logging.info(f"category: {category}")
                # load and combine results tables
                results_pre = pd.read_csv(
                    OUTPUT_DIR
                    / args.workflow
                    / f"{args.year}pre{identifier}"
                    / category
                    / f"results_{category}.csv",
                    index_col=0,
                )
                results_post = pd.read_csv(
                    OUTPUT_DIR
                    / args.workflow
                    / f"{args.year}post{identifier}"
                    / category
                    / f"results_{category}.csv",
                    index_col=0,
                )
                combined_results = combine_event_tables(
                    results_pre, results_post, args.blind
                )

                print_header(f"Results")
                logging.info(
                    combined_results.applymap(
                        lambda x: f"{x:.5f}" if pd.notnull(x) else ""
                    )
                )
                logging.info("\n")

                category_dir = OUTPUT_DIR / args.workflow / args.year / category
                if not category_dir.exists():
                    category_dir.mkdir(parents=True, exist_ok=True)
                combined_results.to_csv(category_dir / f"results_{category}.csv")

                if not args.blind:
                    # save latex table
                    latex_table_asymmetric = df_to_latex_asymmetric(combined_results)
                    with open(
                        category_dir / f"results_{category}_asymmetric.txt", "w"
                    ) as f:
                        f.write(latex_table_asymmetric)
                    latex_table_average = df_to_latex_average(combined_results)
                    with open(
                        category_dir / f"results_{category}_average.txt", "w"
                    ) as f:
                        f.write(latex_table_average)

                # load and combine cutflow tables
                if not args.nocutflow:
                    print_header(f"Cutflow")
                    cutflow_pre = pd.read_csv(
                        OUTPUT_DIR
                        / args.workflow
                        / f"{args.year}pre{identifier}"
                        / category
                        / f"cutflow_{category}.csv",
                        index_col=0,
                    )
                    cutflow_post = pd.read_csv(
                        OUTPUT_DIR
                        / args.workflow
                        / f"{args.year}post{identifier}"
                        / category
                        / f"cutflow_{category}.csv",
                        index_col=0,
                    )
                    combined_cutflow = combine_cutflows(cutflow_pre, cutflow_post)
                    combined_cutflow.to_csv(category_dir / f"cutflow_{category}.csv")
                    logging.info(
                        combined_cutflow.applymap(
                            lambda x: f"{x:.2f}" if pd.notnull(x) else ""
                        )
                    )

    if args.plot:
        subprocess.run("python3 analysis/postprocess/color_map.py", shell=True)

        if not args.postprocess and args.year not in ["2016", "2022", "2023"]:
            postprocess_file = output_dir / f"{args.year}_processed_histograms.coffea"
            processed_histograms = load_histogram_file(postprocess_file)
            if processed_histograms is None:
                cmd = f"python3 run_postprocess.py -w {args.workflow} -y {args.year} --postprocess"
                raise ValueError(
                    f"Postprocess file not found. Please run:\n  '{cmd}' first"
                )

        print_header(f"Running plotter for {args.year}")
        plotter = CoffeaPlotter(
            workflow=args.workflow,
            processed_histograms=processed_histograms,
            year=args.year,
            output_dir=output_dir,
            group_by=group_by,
        )

        for category in workflow_config.event_selection["categories"]:
            logging.info(f"Plotting histograms for category: {category}")
            for variable in workflow_config.histogram_config.variables:
                if plot_variable(variable, group_by, workflow_config.histogram_config):
                    logging.info(variable)
                    plotter.plot_histograms(
                        variable=variable,
                        category=category,
                        yratio_limits=args.yratio_limits,
                        log=args.log,
                        extension=args.extension,
                        add_ratio=not args.no_ratio,
                        blind=args.blind,
                    )
            subprocess.run(
                f"tar -zcvf {output_dir}/{category}/{args.workflow}_{args.year}_plots.tar.gz {output_dir}/{category}/*.{args.extension}",
                shell=True,
            )
