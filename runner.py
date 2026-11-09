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
            "2024",
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
        "-e",
        "--executor",
        dest="executor",
        type=str,
        default="futures",
        help="executor to run the processor",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="coffea",
        choices=["coffea"],
        help="format of output histogram",
    )
    parser.add_argument(
        "-l",
        "--label",
        type=str,
        default="",
        help="label for the output directory",
    )
    parser.add_argument(
        "-m",
        "--memory",
        type=int,
        default=4000,
        help="Requested memory (in MB) for the condor job",
    )
    parser.add_argument(
        "-c",
        "--cpus",
        type=int,
        default=4,
        help="Requested cpus for the condor job",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="change the number of workers to process the analysis",
    )
    parser.add_argument(
        "--global",
        dest="use_global",
        action="store_true",
        help="send xrd-cms-global partition filesets",
    )
    parser.add_argument(
        "--skipbadfiles",
        action="store_true",
        help="skip root files with errors",
    )
    # Mutually exclusive group for dataset discovery strategy
    discovery_group = parser.add_mutually_exclusive_group()
    discovery_group.add_argument(
        "--das",
        dest="discovery",
        action="store_const",
        const="das",
        help="Use dasgoclient for dataset discovery (Default).",
    )
    discovery_group.add_argument(
        "--rucio",
        dest="discovery",
        action="store_const",
        const="rucio",
        help="Use DataDiscoveryCLI (Rucio) for dataset discovery.",
    )
    parser.set_defaults(discovery="rucio")
    args = parser.parse_args()

    # submit (or prepare) a job for each dataset using the given arguments
    datasets_to_run = get_datasets_to_run_over(args.workflow, args.year)
    fileset_checker(samples=datasets_to_run, year=args.year, discovery=args.discovery)
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
            "--memory",
            str(args.memory),
            "--cpus",
            str(args.cpus),
            "--workers",
            str(args.workers),
            "--executor",
            str(args.executor)
        ]
        if args.submit:
            cmd_args.append("--submit")
        if args.eos:
            cmd_args.append("--eos")
        if args.label:
            cmd_args.append("--label")
            cmd_args.append(args.label)
        if args.use_global:
            cmd_args.append("--global")
        if args.skipbadfiles:
            cmd_args.append("--skipbadfiles")
        if args.discovery == "rucio":
            cmd_args.append("--rucio")
        else:
            cmd_args.append("--das")
        subprocess.run(cmd + cmd_args)
