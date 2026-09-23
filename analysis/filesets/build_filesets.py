import os
import yaml
import json
import argparse
from pathlib import Path
from coffea.dataset_tools.dataset_query import DataDiscoveryCLI


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
    )
    parser.add_argument(
        "--samples",
        nargs="*",
        type=str,
        help="(Optional) List of samples to use. If omitted, all available samples will be used",
    )
    args = parser.parse_args()

    # open dataset configs
    filesets_dir = Path.cwd() / "analysis" / "filesets"
    run_key = (
        "Run3"
        if args.year.startswith("202")
        else "Run2"
    )
    nano_version = "nanov15" if args.year == "2024" else ("nanov9" if run_key == "Run2" else "nanov12")
    datasets_dir = filesets_dir / f"{args.year}_{nano_version}.yaml"
    with open(datasets_dir, "r") as f:
        dataset_configs = yaml.safe_load(f)

    if args.samples:
        samples_to_use = args.samples
    else:
        samples_to_use = list(dataset_configs.keys())

    # read dataset queries
    das_queries = {}
    for sample in samples_to_use:
        das_queries[sample] = dataset_configs[sample]["query"]

    # create a dataset_definition dict for each yeare
    dataset_definition = {}
    for dataset_key, query in das_queries.items():
        dataset_definition[f"/{query}"] = {"short_name": dataset_key}
    # the dataset definition is passed to a DataDiscoveryCLI
    ddc = DataDiscoveryCLI()
    # set the allow sites to look for replicas
    sites_file = filesets_dir / f"{args.year}_sites.yaml"
    with open(sites_file, "r") as f:
        sites = yaml.safe_load(f)["white"]
    ddc.do_allowlist_sites(sites)
    # query rucio and get replicas
    ddc.load_dataset_definition(
        dataset_definition,
        query_results_strategy="all",
        replicas_strategy="round-robin",
    )
    ddc.do_save(f"dataset_discovery_{args.year}.json")

    with open(f"dataset_discovery_{args.year}.json", "r") as f:
        dataset_discovery = json.load(f)
    
    new_dataset = {}

    for i, (dataset_name, dataset_info) in enumerate(dataset_discovery.items(), start=1):
        short_name = dataset_info["metadata"]["short_name"]
    
        key_name = f"{short_name}"
    
        new_dataset[key_name] = {
            "files": dataset_info[
                "files"
            ],
            "metadata": {"short_name": short_name},
        }
    
    os.remove(f"dataset_discovery_{args.year}.json")
    fileset_file = filesets_dir / f"fileset_{args.year}_NANO_lxplus.json"
    
    with open(fileset_file, "w") as json_file:
        json.dump(new_dataset, json_file, indent=4, sort_keys=True)