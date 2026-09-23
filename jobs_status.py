"""Check job outputs, identify missing results, and optionally resubmit jobs or update input filesets based on xrootd site issues"""

import sys
import select
import yaml
import json
import argparse
import logging
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from analysis.utils import make_output_directory
from analysis.filesets.xrootd_sites import xroot_to_site
from analysis.filesets.utils import divide_list, modify_site_list, extract_xrootd_errors


def timed_input(prompt, timeout=3, default="y"):
    """Request user input with a timeout. Falls back to default if timed out or empty."""
    print(f"{prompt} [{default.upper()}/n] (auto-selected in {timeout}s): ", end="", flush=True)
    
    # Check for available input on stdin within the timeout limit
    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    if ready:
        response = sys.stdin.readline().strip()
        return response.lower() if response else default.lower()
    else:
        print(f"\n[Timeout reached] Using default option: '{default.upper()}'")
        return default.lower()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-w",
        "--workflow",
        type=str,
        required=True,
        choices=[
            f.stem for f in (Path.cwd() / "analysis" / "workflows").glob("*.yaml")
        ],
        help="workflow config to test",
    )
    parser.add_argument(
        "-y",
        "--year",
        type=str,
        required=True,
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
        "--eos", action="store_true", help="Enable reading outputs from /eos"
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="coffea",
        choices=["coffea", "root"],
        help="Format of output histograms",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="change the number of workers to process the analysis",
    )
    parser.add_argument(
        "--hours_ago",
        type=int,
        default=3,
        help="use .err files that have been modified less than 'hours_ago' hours ago",
    )
    parser.add_argument(
        "-l",
        "--label",
        type=str,
        default="",
        help="label for the output directory",
    )
    parser.add_argument("--reset", action="store_true", help="Reset previous outputs and run runner.py")
    parser.add_argument(
        "-m",
        "--memory",
        type=str,
        default="4000",
        help="Requested memory (in MB) for the condor job",
    )
    parser.add_argument(
        "--global",
        action="store_true",
        help="send xrd-cms-global partition filesets",
    )
    return parser.parse_args()


def get_jobs_info(job_dir, output_dir, log_dir, output_format, hours_ago=3):
    """
    Collect expected and completed job numbers per dataset, and gather error logs.

    Parameters:
    -----------
        job_dir (Path): Directory containing dataset job folders.
        output_dir (Path): Directory with output files.
        log_dir (Path): Directory containing log files.
        output_format (str): File format of the output (e.g., 'coffea' or 'root').

    Returns:
    --------
        tuple:
            - jobnum (dict): Expected job numbers per dataset.
            - jobnum_done (dict): Successfully completed job numbers per dataset.
            - error_file (list): List of .err log files.
    """
    jobnum, jobnum_done = {}, {}
    error_file = []

    for dataset_dir in job_dir.iterdir():
        if not dataset_dir.is_dir():
            continue

        dataset = dataset_dir.name
        jobnum_path = dataset_dir / "jobnum.txt"
        if not jobnum_path.exists():
            raise FileNotFoundError(
                f"Missing jobnum.txt for dataset '{dataset}'. Expected at: {jobnum_path}"
            )

        jobnum[dataset] = jobnum_path.read_text().splitlines()
        output_files = list((output_dir / dataset).glob(f"*.{output_format}"))
        jobnum_done[dataset] = [f.stem.replace(f"{dataset}_", "") for f in output_files]

        # look for error files created within the last 3 hours
        x_hours_ago = datetime.now() - timedelta(hours=hours_ago)
        for err_file in (log_dir / dataset).glob("*.err"):
            if datetime.fromtimestamp(err_file.stat().st_mtime) > x_hours_ago:
                error_file.append(err_file)

    return jobnum, jobnum_done, error_file


def print_job_status(jobnum, jobnum_done):
    """
    Print a summary of expected, finished, and missing jobs. Show YAML list of datasets with missing jobs.

    Parameters:
    -----------
        jobnum (dict): Expected job numbers per dataset.
        jobnum_done (dict): Completed job numbers per dataset.

    Returns:
    --------
        tuple:
            - jobnum_missing (dict): Missing job numbers per dataset.
            - datasets_with_missing (list): Datasets that have missing jobs.
    """
    jobnum_missing = {d: set(jobnum[d]) - set(jobnum_done.get(d, [])) for d in jobnum}
    n_expected = sum(len(v) for v in jobnum.values())
    n_done = sum(len(v) for v in jobnum_done.values())
    n_missing = sum(len(v) for v in jobnum_missing.values())

    logging.info("Jobs status:")
    logging.info(f"Expected: {n_expected}")
    logging.info(f"Finished: {n_done}")
    logging.info(f"Missing: {n_missing}\n")

    datasets_with_missing = [d for d in jobnum_missing if jobnum_missing[d]]
    if n_missing:
        print(
            f"Datasets with missing jobs ({len(datasets_with_missing)}/{len(jobnum)}):"
        )
        print(
            yaml.dump(
                datasets_with_missing,
                default_flow_style=False,
                sort_keys=False,
                indent=2,
            )
        )

    return jobnum_missing, datasets_with_missing


def analyze_xrootd_errors(error_file):
    """
    Analyze error logs and extract problematic xrootd sites.

    Parameters:
    -----------
        error_file (list): List of error log files.

    Returns:
    --------
        list: Sites with detected xrootd errors.
    """
    xrootd_errs = extract_xrootd_errors(error_file)
    if not xrootd_errs:
        return []

    site_errs = [xroot_to_site[err] for err in xrootd_errs if err in xroot_to_site]
    for err in xrootd_errs:
        if err not in xroot_to_site:
            logging.warning(f"Could not identify the site for xrootd error {err}")

    print("Sites with xrootd OS errors:")
    print(yaml.dump(site_errs, default_flow_style=False, sort_keys=False, indent=2))
    return site_errs


def update_input_filesets(
    site_errs, year, fileset_dir, job_dir, datasets_with_missing_jobs, use_global
):
    """
    Blacklist failing xrootd sites and update filesets for affected datasets.

    Parameters:
    -----------
        site_errs (list): List of sites to blacklist.
        year (str): Dataset year.
        fileset_dir (Path): Directory containing JSON filesets.
        job_dir (Path): Directory with Condor job files.
        datasets_with_missing_jobs (list): Datasets to update.
    """
    for site in xroot_to_site.values():
        modify_site_list(year, site, "white")

    for site in site_errs:
        modify_site_list(year, site, "black")

    samples_str = (
        " ".join(datasets_with_missing_jobs) if datasets_with_missing_jobs else ""
    )
    subprocess.run(["python3", "fetch.py", "--year", year, "--samples", samples_str, "--skip_site"])

    fileset_path = fileset_dir / f"fileset_{year}_NANO_lxplus.json"
    all_filesets = json.loads(fileset_path.read_text())

    for dataset in datasets_with_missing_jobs:
        if dataset not in all_filesets:
            logging.warning(f"Dataset {dataset} not found in fileset JSON")
            continue

        root_files = all_filesets[dataset]['files']
        args_json = job_dir / dataset / "arguments.json"
        if not args_json.exists():
            logging.error(f"Missing arguments.json for dataset {dataset}")
            continue

        nfiles = json.loads(args_json.read_text())["nfiles"]
        root_files_list = divide_list(list(root_files), nfiles)

        partition_dataset = {
            str(i + 1): {
                (f"{dataset}_{i+1}" if len(root_files_list) > 1 else dataset): {
                    "files": {root_file: "Events" for root_file in chunk},
                    "metadata": all_filesets[dataset]['metadata']
                }
            }
            for i, chunk in enumerate(root_files_list)
        }

        partition_file = job_dir / dataset / "partitions.json"
        with open(partition_file, "w") as json_file:
            json.dump(partition_dataset, json_file, indent=4)
        
        if use_global:
            subprocess.run(f"sed -i -E 's#root://.*/store/#root://cms-xrd-global.cern.ch//store/#g' {partition_file}",shell=True)


def resubmit_jobs(job_dir, jobnum_missing, datasets_with_missing_jobs, workflow, year, use_global):
    """
    Prepare and resubmit jobs for datasets with missing jobs.

    Parameters:
    -----------
        job_dir (Path): Directory with Condor job files.
        jobnum_missing (dict): Missing job numbers per dataset.
        datasets_with_missing_jobs (list): List of affected datasets.
        workflow (str): Workflow name.
        year (str): Dataset year.
    """
    to_resubmit = []
    for dataset in datasets_with_missing_jobs:
        partition_file = job_dir / dataset / "partitions.json" 
        if use_global:
            subprocess.run(f"sed -i -E 's#root://.*/store/#root://cms-xrd-global.cern.ch//store/#g' {partition_file}",shell=True)
        missing_file = job_dir / dataset / "missing.txt"
        with open(missing_file, "w") as f:
            print(*sorted(jobnum_missing[dataset]), sep="\n", file=f)

        condor_file = job_dir / dataset / f"{workflow}_{dataset}.sub"
        condor_backup = condor_file.with_name(condor_file.stem + "_all.sub")
        subprocess.run(["cp", str(condor_file), str(condor_backup)])

        submit_text = condor_file.read_text().replace("jobnum.txt", "missing.txt")
        condor_file.write_text(submit_text)
        logging.info(f"Condor file updated: {workflow}/{year}/{dataset}")
        to_resubmit.append(str(condor_file))

    for submit_file in to_resubmit:
        subprocess.run(["condor_submit", submit_file])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    if args.label:
        workflow_dir = f"{args.workflow}/{args.label}"
    else:
        workflow_dir = args.workflow

    if args.reset:
        subprocess.run(f"rm -rf condor/{workflow_dir}/{args.year}", shell=True)
        subprocess.run(f"rm -rf condor/logs/{workflow_dir}/{args.year}", shell=True)
        subprocess.run(f"rm -rf analysis/filesets/{args.year}_sites.yaml", shell=True)
        subprocess.run(
            f"rm -rf analysis/filesets/fileset_{args.year}_NANO_lxplus.json", shell=True
        )
        reset_cmd = f"python3 runner.py -w {args.workflow} -y {args.year} -m {args.memory} --workers {args.workers}"
        if args.label:
            reset_cmd += f" -l {args.label}"
        if args.eos:
            reset_cmd += " --eos"
        subprocess.run(reset_cmd, shell=True)

    output_dir = make_output_directory(args)
    logging.info(f"Reading outputs from: {output_dir}\n")

    base_dir = Path.cwd()
    condor_dir = base_dir / "condor"
    job_dir = condor_dir / workflow_dir / args.year
    log_dir = condor_dir / "logs" / workflow_dir / args.year
    fileset_dir = base_dir / "analysis" / "filesets"

    jobnum, jobnum_done, error_file = get_jobs_info(
        job_dir, output_dir, log_dir, args.output_format, args.hours_ago
    )

    jobnum_missing, datasets_with_missing_jobs = print_job_status(jobnum, jobnum_done)

    if jobnum_missing and datasets_with_missing_jobs:
        #site_errs = analyze_xrootd_errors(error_file)

        #if site_errs and timed_input("Update input filesets?", timeout=3, default="y") in [
        #    "y",
        #    "yes",
        #]:
        #    update_input_filesets(
        #        site_errs, args.year, fileset_dir, job_dir, datasets_with_missing_jobs, getattr(args,"global") 
        #    )

        if timed_input("Update and resubmit jobs?", timeout=0, default="y") in ["y", "yes"]:
            resubmit_jobs(
                job_dir,
                jobnum_missing,
                datasets_with_missing_jobs,
                args.workflow,
                args.year,
                getattr(args,"global")
            )
