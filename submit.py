import json
import argparse
from pathlib import Path
from coffea import processor
from coffea.util import save
from coffea.nanoevents import NanoAODSchema
from analysis.utils import write_root
from analysis.processors.base import BaseProcessor
from coffea.dataset_tools import preprocess


def main(args):
    with open(args.partition_json) as f:
        partition_fileset = json.load(f)
    # Configure the Runner using coffea 2024+ API

    preprocessed_partition_fileset, _ = preprocess(
        partition_fileset,
        align_clusters = True,
        #step_size = 100_000,
        skip_bad_files = args.skipbadfiles,
    )

    futures_run = processor.Runner(
        executor=processor.FuturesExecutor(workers=args.workers, compression=None, retries=1),
        schema=NanoAODSchema,
        #chunksize=100000,
        savemetrics=True,
        #xrootdtimeout=120,
    )
    # Execute processing job using the Runner instance
    out, metrics = futures_run(
        preprocessed_partition_fileset,
        treename="Events",
        processor_instance=BaseProcessor(
            workflow=args.workflow, year=args.year, mode="virtual"
        ),
    )
    del metrics["columns"]
    out["metrics"] = metrics
    savepath = f"{args.output_path}/{args.dataset}"
    if args.output_format == "coffea":
        save(out, f"{savepath}.coffea")
    elif args.output_format == "root":
        write_root(out, savepath, args)


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
            "2024",
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
        choices=["coffea", "root"],
        help="format of output histogram",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default="4",
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
    parser.add_argument(
        "-e",
        "--executor",
        dest="executor",
        type=str,
        default="futures",
        help="executor to run the processor",
    )
    args = parser.parse_args()
    main(args)
