import argparse
import subprocess
from pathlib import Path

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
    parser.add_argument(
        "--image",
        dest="image",
        type=str,
        default="/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask-almalinux9:2025.10.1-py3.10",
    )
    parser.add_argument(
        "--site",
        dest="site",
        default="root://xrootd-vanderbilt.sites.opensciencegrid.org:1094",
        type=str,
        help="site from which to read the signal samples",
    )
    parser.add_argument(
        "--skip_site",
        action="store_true",
        help="Skip white/black sites initialization",
    )
    args = parser.parse_args()

    try:
        subprocess.run("voms-proxy-info -exists", shell=True, check=True)
    except subprocess.CalledProcessError:
        raise Exception(
            "VOMS proxy expired or non-existing: please run 'voms-proxy-init --voms cms'"
        )

    if not args.skip_site:
        # initialize white/black sites
        cmd = f"python3 analysis/filesets/build_sites.py --year {args.year}"
        subprocess.run(cmd, shell=True)

    # keep container Python isolated from host user-site packages (~/.local),
    # otherwise dask/distributed versions can be mixed
    samples_str = " ".join(args.samples) if args.samples else ""
    cmd = (
        f"singularity exec "
        f"--env PYTHONNOUSERSITE=1 "
        f"-B /afs "
        f"-B /cvmfs "
        f"-B analysis/filesets/rucio_utils.py:/usr/local/lib/python3.10/site-packages/coffea/dataset_tools/rucio_utils.py "
        f"{args.image} "
        f"python3 analysis/filesets/build_filesets.py --year {args.year} --samples {samples_str}"
    )
    subprocess.run(cmd, shell=True)

    # add signal samples
    signal_cmd = f"python3 analysis/filesets/build_signal_filesets.py --year {args.year} --site {args.site}"
    subprocess.run(signal_cmd, shell=True)
