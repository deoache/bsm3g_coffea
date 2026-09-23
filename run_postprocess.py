import gc
import sys
import yaml
import glob
import logging
import argparse
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from coffea.util import save, load
from coffea.processor import accumulate
from analysis.utils import make_output_directory
from analysis.filesets.utils import get_dataset_config, get_process_maps
from analysis.workflows.config import WorkflowConfigBuilder
from analysis.postprocess.coffea_plotter import CoffeaPlotter
from analysis.postprocess.coffea_postprocessor import (
    save_process_histograms_by_process,
    save_process_histograms_by_sample,
    load_processed_histograms,
    get_results_report,
)
from analysis.postprocess.utils import (
    print_header,
    setup_logger,
    clear_output_directory,
    combine_event_tables,
    combine_cutflows,
    df_to_latex_average,
    df_to_latex_asymmetric,
    uncertainty_table,
    build_systematic_summary,
)


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
            "2024",
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
    parser.add_argument(
        "--output_format",
        type=str,
        default="coffea",
        choices=["coffea", "root"],
        help="Format of output histograms",
    )
    parser.add_argument(
        "-l",
        "--label",
        type=str,
        default="",
        help="label for the output directory",
    )
    parser.add_argument("--blind", action="store_true", help="Blind data")
    parser.add_argument(
        "--eos", action="store_true", help="Enable reading outputs from /eos"
    )
    return parser.parse_args()


def load_year_histograms(year: str, output_dir: Path):
    """load and merge histograms from pre/post campaigns"""
    aux_map = {
        "2016": ["2016preVFP", "2016postVFP"],
        "2022": ["2022preEE", "2022postEE"],
        "2023": ["2023preBPix", "2023postBPix"],
    }
    pre_year, post_year = aux_map[year]

    pre_file = output_dir.parent / pre_year / f"{pre_year}_processed_histograms.coffea"
    post_file = (
        output_dir.parent / post_year / f"{post_year}_processed_histograms.coffea"
    )
    return accumulate([load(pre_file), load(post_file)])


if __name__ == "__main__":
    args = parse_arguments()

    output_dir = make_output_directory(args)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    clear_output_directory(str(output_dir), "txt")
    setup_logger(output_dir)

    config_builder = WorkflowConfigBuilder(workflow=args.workflow)
    workflow_config = config_builder.build_workflow_config()
    event_selection = workflow_config.event_selection
    categories = workflow_config.event_selection["categories"]
    processed_histograms = None

    if "data" not in workflow_config.datasets:
        args.blind = True

    _, process_name_map, key_process_map = get_process_maps(workflow_config, args.year)

    if args.postprocess and (args.year not in ["2016", "2022", "2023"]):
        print_header(f"Reading outputs from: {output_dir}")

        output_files = [
            i
            for i in glob.glob(f"{output_dir}/*/*.coffea", recursive=True)
            if not i.split("/")[-1].startswith("cutflow")
        ]
        process_samples_map = defaultdict(list)

        samples_in_out = [
            str(p).split("/")[-1] for p in output_dir.iterdir() if p.is_dir()
        ]

        dataset_keys = [
            key
            for dataset_keys in workflow_config.datasets.values()
            for key in dataset_keys
        ]

        for sample, config in get_dataset_config(args.year).items():
            if sample in samples_in_out:
                if config["key"] in dataset_keys:
                    process_samples_map[config["process"]].append(sample)

        # group output file paths by sample name
        grouped_outputs = {}
        for output_file in output_files:
            sample_name = output_file.split("/")[-1].split(".coffea")[0]
            if sample_name.rsplit("_")[-1].isdigit():
                sample_name = "_".join(sample_name.rsplit("_")[:-1])
            sample_name = sample_name.replace(f"{args.year}_", "")
            if sample_name in grouped_outputs:
                grouped_outputs[sample_name].append(output_file)
            else:
                grouped_outputs[sample_name] = [output_file]

        for sample in grouped_outputs:
            save_process_histograms_by_sample(
                year=args.year,
                output_dir=output_dir,
                sample=sample,
                grouped_outputs=grouped_outputs,
                categories=categories,
            )
            gc.collect()

        for process in process_samples_map:
            save_process_histograms_by_process(
                year=args.year,
                output_dir=output_dir,
                process_samples_map=process_samples_map,
                process=process,
                categories=categories,
            )
            gc.collect()

        processed_histograms = load_processed_histograms(
            year=args.year,
            output_dir=output_dir,
            process_samples_map=process_samples_map,
        )

        for category in categories:
            logging.info(f"category: {category}")
            category_dir = Path(f"{output_dir}/{category}")

            print_header(f"Cutflow")
            cutflow_df = pd.DataFrame()
            for process in process_samples_map:
                cutflow_file = category_dir / f"cutflow_{category}_{process}.csv"
                cutflow_df = pd.concat(
                    [cutflow_df, pd.read_csv(cutflow_file, index_col=[0])], axis=1
                )
            columns_to_drop = []
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

            if not "eff" in args.workflow:
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
                    with open(
                        category_dir / f"results_{category}_average.txt", "w"
                    ) as f:
                        f.write(latex_table_average)

    if args.year in ["2016", "2022", "2023"]:
        if args.postprocess:
            processed_histograms = load_year_histograms(args.year, output_dir)
            save(
                processed_histograms,
                f"{output_dir}/{args.year}_processed_histograms.coffea",
            )
        else:
            postprocess_file = output_dir / f"{args.year}_processed_histograms.coffea"
            if not postprocess_file.exists():
                cmd = f"python3 run_postprocess.py -w {args.workflow} -y {args.year} --postprocess"
                raise ValueError(
                    f"Postprocess file not found. Please run:\n  '{cmd}' first"
                )

            processed_histograms = load(str(postprocess_file))

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
            syst_df.to_csv(f"{output_dir}/uncertainty_table.csv")
            logging.info(syst_df)
            logging.info("\n")

        for category in categories:
            logging.info(f"category: {category}")
            category_dir = output_dir / category
            if not category_dir.exists():
                category_dir.mkdir(parents=True, exist_ok=True)

            # load and combine cutflow tables
            print_header(f"Cutflow")
            cutflow_pre = pd.read_csv(
                output_dir.parent
                / f"{args.year}pre{identifier}"
                / category
                / f"cutflow_{category}.csv",
                index_col=0,
            )
            cutflow_post = pd.read_csv(
                output_dir.parent
                / f"{args.year}post{identifier}"
                / category
                / f"cutflow_{category}.csv",
                index_col=0,
            )
            combined_cutflow = combine_cutflows(cutflow_pre, cutflow_post)
            combined_cutflow.to_csv(category_dir / f"cutflow_{category}.csv")
            logging.info(
                combined_cutflow.applymap(lambda x: f"{x:.2f}" if pd.notnull(x) else "")
            )
            if not "eff" in args.workflow:
                # load and combine results tables
                results_pre = pd.read_csv(
                    output_dir.parent
                    / f"{args.year}pre{identifier}"
                    / category
                    / f"results_{category}.csv",
                    index_col=0,
                )
                results_post = pd.read_csv(
                    output_dir.parent
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

    if args.postprocess:
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

    if args.plot:
        subprocess.run("python3 analysis/postprocess/color_map.py", shell=True)

        if not args.postprocess and args.year not in ["2016", "2022", "2023"]:
            postprocess_file = output_dir / f"{args.year}_processed_histograms.coffea"
            if not postprocess_file.exists():
                cmd = f"python3 run_postprocess.py -w {args.workflow} -y {args.year} --postprocess"
                raise ValueError(
                    f"Postprocess file not found. Please run:\n  '{cmd}' first"
                )
            processed_histograms = load(str(postprocess_file))

        print_header("Plots")
        plotter = CoffeaPlotter(
            workflow=args.workflow,
            processed_histograms=processed_histograms,
            year=args.year,
            output_dir=output_dir,
        )

        for category in workflow_config.event_selection["categories"]:
            logging.info(f"Plotting histograms for category: {category}")
            for variable in workflow_config.histogram_config.variables:
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