import yaml
import argparse
from pathlib import Path
from xrootd_sites import xroot_to_site


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
        ],
    )
    args = parser.parse_args()
    data = {"black": [], "white": sorted(set(xroot_to_site.values()))}
    sites_file = Path.cwd() / "analysis" / "filesets" / f"{args.year}_sites.yaml"
    with open(sites_file, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
