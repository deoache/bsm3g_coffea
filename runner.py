import json
import argparse
import subprocess
from pathlib import Path
from analysis.filesets.utils import get_datasets_to_run_over, fileset_checker


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-w",
        "--workflow",
        dest="workflow",
        type=str,
        choices=[
            f.stem for f in (Path.cwd() / "analysis" / "workflows").glob("*.yaml")
        ],
        help="workflow to run",
    )
    parser.add_argument(
        "-y",
        "--year",
        dest="year",
        type=str,
        choices=[
            "2016preVFP",
            "2016postVFP",
            "2017",
            "2018",
            "2022preEE",
            "2022postEE",
            "2023preBPix",
            "2023postBPix",
        ],
        help="dataset year",
    )
    parser.add_argument(
        "--nfiles",
        dest="nfiles",
        type=int,
        default=10,
        help="number of root files to include in each dataset partition (default 10)",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Enable Condor job submission. If not provided, it just builds condor files",
    )
    parser.add_argument(
        "--eos",
        action="store_true",
        help="Enable saving outputs to /eos",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="coffea",
        choices=["coffea", "parquet"],
        help="format of output histogram",
    )
    args = parser.parse_args()

    # submit (or prepare) a job for each dataset using the given arguments
    datasets_to_run = get_datasets_to_run_over(args.workflow, args.year)
    fileset_checker(datasets_to_run, args.year)
    cmd = ["python3", "submit_condor.py"]
    for dataset in datasets_to_run:
        cmd_args = [
            "--workflow",
            args.workflow,
            "--year",
            args.year,
            "--dataset",
            dataset,
            "--nfiles",
            str(args.nfiles),
            "--output_format",
            args.output_format,
        ]
        if args.submit:
            cmd_args.append("--submit")
        if args.eos:
            cmd_args.append("--eos")
        subprocess.run(cmd + cmd_args)
