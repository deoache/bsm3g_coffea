import yaml
import json
import argparse
import subprocess
from pathlib import Path


def get_signal_fileset(query):
    query = f"file dataset=/{query}"
    farray = subprocess.run(
        ["dasgoclient", f"-query={query} instance=prod/phys03"],
        stdout=subprocess.PIPE,
        universal_newlines=True,
    )
    stdout = farray.stdout
    stdout_array = stdout.split("\n")
    stdout_array = stdout_array[:-1]
    stdout_array[-1] = stdout_array[-1].replace(",", "")
    return stdout_array


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
        "--site",
        dest="site",
        default="root://xrootd-vanderbilt.sites.opensciencegrid.org:1094",
        type=str,
        help="site from which to read the signal samples",
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

    # load already generated fileset
    fileset_file = filesets_dir / f"fileset_{args.year}_NANO_lxplus.json"
    with open(fileset_file, "r") as f:
        new_dataset = json.load(f)

    # query and add signal samples to fileset
    for sample in list(dataset_configs.keys()):
        if sample.lower().startswith("signal"):
            query = dataset_configs[sample]["query"]
            files = get_signal_fileset(query)
            files = [f"{args.site}/{f}" for f in files]
            print(f"Adding sample {sample}")
            new_dataset[sample] = files

    with open(fileset_file, "w") as json_file:
        json.dump(new_dataset, json_file, indent=4, sort_keys=True)