import json
import argparse
from pathlib import Path
from coffea import processor
from coffea.util import save
from coffea.nanoevents import NanoAODSchema
from analysis.utils import write_root
from analysis.processors.base import BaseProcessor


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
        help="workflow config to submit",
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
        "-d",
        "--dataset",
        dest="dataset",
        type=str,
        help="dataset",
    )
    parser.add_argument(
        "--partition_json",
        dest="partition_json",
        type=str,
        help="json with partition dataset",
    )
    parser.add_argument(
        "--output_path",
        dest="output_path",
        type=str,
        help="output path",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="coffea",
        choices=["coffea", "parquet"],
        help="format of output histogram",
    )
    parser.add_argument(
        "--eos",
        action="store_true",
        help="Enable saving outputs to /eos",
    )
    parser.add_argument(
        "--user",
        type=str,
        help="User name",
    )
    args = parser.parse_args()

    # set output location (used when --output_format parquet)
    if args.eos:
        output_location = f"root://eosuser.cern.ch//eos/user/{args.user[0]}/{args.user}/higgscharm/outputs/"
    else:
        output_location = (
            f"/afs/cern.ch/user/{args.user[0]}/{args.user}/public/higgscharm/outputs/"
        )

    # load partition_fileset,
    with open(args.partition_json) as f:
        partition_fileset = json.load(f)

    # run processor
    out = processor.run_uproot_job(
        partition_fileset,
        treename="Events",
        processor_instance=BaseProcessor(
            workflow=args.workflow,
            year=args.year,
            output_format=args.output_format,
            output_location=output_location,
        ),
        executor=processor.futures_executor,
        executor_args={"schema": NanoAODSchema, "workers": 4},
    )

    # save output
    savepath = f"{args.output_path}/{args.dataset}"
    save(out, f"{savepath}.coffea")
