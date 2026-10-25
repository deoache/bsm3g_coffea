"""
This script inspects Condor job outputs, identifies missing jobs, analyzes recent xrootd-related failures, and optionally builds input filesets with problematic sites blacklisted and resubmits only the missing jobs
"""

import sys
import yaml
import json
import select
import argparse
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

from analysis.utils import make_output_directory
from analysis.filesets.xrootd_sites import xroot_to_site
from analysis.filesets.utils import (
    divide_list,
    modify_site_list,
    analyze_xrootd_error_logs,
    get_missing_job_partitions,
)


def timed_input(prompt, timeout=10, default="y"):
    """Request user input with a timeout. Falls back to default if timed out or empty."""
    print(
        f"{prompt} [{default.upper()}/n] (auto-selected in {timeout}s): ",
        end="",
        flush=True,
    )

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
        dest="workflow",
        required=True,
        type=str,
        choices=[
            f.stem for f in (Path.cwd() / "analysis" / "workflows").glob("*.yaml")
        ],
        help="Workflow name (must correspond to a YAML file in analysis/workflows)",
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
        help="Dataset year",
    )
    parser.add_argument(
        "--eos",
        action="store_true",
        help="Read job outputs from EOS instead of local storage",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="coffea",
        choices=["coffea", "parquet"],
        help="Output file format of the produced histograms",
    )
    parser.add_argument(
        "--hours_ago",
        type=int,
        default=8,
        help="Only consider .err files modified within the last N hours",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete all job outputs, logs, and filesets for this workflow/year and rerun",
    )
    parser.add_argument(
        "--single_fetch",
        action="store_true",
        help="Run fetch.py only once by merging all failing sites and datasets into a single group",
    )
    parser.add_argument(
        "--global",
        action="store_true",
        help="send xrd-cms-global partition filesets",
    )
    parser.add_argument(
        "-m",
        "--memory",
        type=str,
        default="4000",
        help="Requested memory (in MB) for the condor job",
    )
    parser.add_argument(
        "-l",
        "--label",
        type=str,
        default="",
        help="output directory label",
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
        "-v",
        "--verbose",
        action="store_true",
        help="Show detailed XRootD error diagnostics.",
    )
    return parser.parse_args()


def get_jobs_info(job_dir, output_dir, log_dir, output_format, hours_ago=3):
    """
    Inspect the job directory structure and determine:
      1. Which jobs were expected to run (from jobnum.txt)
      2. Which jobs successfully produced output files
      3. Which jobs produced recent error logs

    error logs are collected *per dataset*, which allows later
    analysis and remediation to remain dataset-local rather than global.

    Parameters
    ----------
    job_dir : Path
        Directory containing one subdirectory per dataset with Condor configs.
    output_dir : Path
        Directory containing produced output files.
    log_dir : Path
        Directory containing Condor log files (.err, .out, .log).
    output_format : str
        File extension of produced outputs (e.g. "coffea", "parquet").
    hours_ago : int
        Only .err files modified within this many hours are considered relevant.

    Returns
    -------
    tuple
        jobnum : dict
            dataset -> list of expected job numbers (strings)
        jobnum_done : dict
            dataset -> list of job numbers that produced output
        error_files : dict
            dataset -> list of Path objects pointing to recent .err files
    """
    jobnum, jobnum_done = {}, {}
    error_files = {}

    for dataset_dir in job_dir.iterdir():
        if not dataset_dir.is_dir():
            continue

        dataset = dataset_dir.name
        jobnum_path = dataset_dir / "jobnum.txt"
        if not jobnum_path.exists():
            raise FileNotFoundError(
                f"Missing jobnum.txt for dataset '{dataset}'. "
                f"Expected at: {jobnum_path}"
            )

        # Read expected job numbers
        jobnum[dataset] = jobnum_path.read_text().splitlines()

        # Discover completed jobs by looking for output files
        output_files = list((output_dir / dataset).glob(f"*.{output_format}"))

        prefix = f"{dataset}_"

        jobnum_done[dataset] = [
            f.stem.removeprefix(prefix)
            for f in output_files
            if f.stem.startswith(prefix)
        ]

        # Collect recent error logs for this dataset
        x_hours_ago = datetime.now() - timedelta(hours=hours_ago)
        dataset_errs = []
        for err_file in (log_dir / dataset).glob("*.err"):
            if datetime.fromtimestamp(err_file.stat().st_mtime) > x_hours_ago:
                dataset_errs.append(err_file)

        if dataset_errs:
            error_files[dataset] = dataset_errs

    return jobnum, jobnum_done, error_files


def print_job_status(jobnum, jobnum_done, show_missing_datasets=False):
    """
    Print the overall Condor job summary.

    Parameters
    ----------
    jobnum : dict
        Dataset -> list of expected job numbers.
    jobnum_done : dict
        Dataset -> list of job numbers with completed outputs.
    show_missing_datasets : bool
        Whether to print the basic list of datasets with missing jobs.

    Returns
    -------
    tuple
        jobnum_missing : dict
            Dataset -> set of missing job numbers.
        datasets_with_missing_jobs : list
            Datasets with at least one missing job.
    """
    jobnum_missing = {}
    datasets_with_missing_jobs = []

    for dataset, expected_jobs in jobnum.items():
        expected = set(expected_jobs)
        completed = set(jobnum_done.get(dataset, []))
        missing = expected - completed

        jobnum_missing[dataset] = missing

        if missing:
            datasets_with_missing_jobs.append(dataset)

    total_expected = sum(len(set(jobs)) for jobs in jobnum.values())
    total_finished = sum(
        len(set(jobnum.get(dataset, [])) & set(jobnum_done.get(dataset, [])))
        for dataset in jobnum
    )
    total_missing = sum(len(missing) for missing in jobnum_missing.values())

    print("-" * 60)
    print("JOB STATUS SUMMARY")
    print("-" * 60)
    print(f"Total jobs expected : {total_expected}")
    print(f"Total jobs finished : {total_finished}")
    print(f"Total jobs missing  : {total_missing}")
    print("-" * 60)

    if show_missing_datasets and datasets_with_missing_jobs:
        print(
            f"Datasets with missing jobs "
            f"({len(datasets_with_missing_jobs)}/{len(jobnum)}):"
        )
        print("-" * 60)

        for dataset in sorted(datasets_with_missing_jobs):
            print(f"- {dataset}")

    return jobnum_missing, datasets_with_missing_jobs


def update_executor_for_dataset(job_dir, dataset, executor="iterative"):
    """Update the executor in a dataset's arguments.json."""
    args_json = job_dir / dataset / "arguments.json"

    if not args_json.exists():
        print(f"[{dataset}] Missing arguments.json — cannot update executor")
        return

    arguments = json.loads(args_json.read_text())

    if arguments.get("executor") == executor:
        print(f"[{dataset}] Executor is already '{executor}'")
        return

    arguments["executor"] = executor
    args_json.write_text(json.dumps(arguments, indent=4) + "\n")

    print(f"[{dataset}] Updated executor to '{executor}'")


def analyze_xrootd_errors_by_dataset(
    error_files,
    job_dir,
    dataset,
    verbose=False,
):
    """
    Analyze XRootD errors and map endpoints to CMS site names.

    First uses endpoints extracted from error logs. If none of these
    endpoints can be mapped to a known CMS site, retries using the
    endpoints associated with the currently missing partitions.

    Parameters
    ----------
    error_files : iterable
        Paths to Condor error logs. May be empty.
    job_dir : str or pathlib.Path
        Root directory containing dataset job directories.
    dataset : str
        Dataset name.
    verbose : bool, optional
        Print detailed diagnostics.

    Returns
    -------
    dict
        Error types, endpoints, missing partitions, and mapped site names.
    """
    errors = analyze_xrootd_error_logs(
        error_files,
        job_dir=job_dir,
        dataset=dataset,
        verbose=verbose,
    )

    def map_sites(endpoints):
        return {
            xroot_to_site[endpoint]
            for endpoint in endpoints
            if endpoint in xroot_to_site
        }

    # First try the endpoints extracted from the logs.
    log_endpoints = set(errors["endpoints"])
    log_sites = map_sites(log_endpoints)

    errors["sites"] = log_sites

    # If no site is identified, explicitly retry using missing partitions.
    if not log_sites:
        dataset_dir = Path(job_dir) / dataset
        partitions_path = dataset_dir / "partitions.json"

        fallback_endpoints = set()

        if partitions_path.is_file():
            partitions = json.loads(partitions_path.read_text())

            missing_partitions = get_missing_job_partitions(
                Path(job_dir),
                dataset,
            )

            endpoint_pattern = re.compile(
                r"root://[a-zA-Z0-9._-]+(?::\d+)?",
                flags=re.IGNORECASE,
            )

            for partition_id in missing_partitions:
                partition_data = partitions.get(str(partition_id), {})

                for sample_data in partition_data.values():
                    for url in sample_data.get("files", {}):
                        match = endpoint_pattern.match(url)

                        if match:
                            fallback_endpoints.add(match.group(0))

        fallback_sites = map_sites(fallback_endpoints)

        if fallback_sites:
            errors["endpoints"] = fallback_endpoints
            errors["sites"] = fallback_sites

        if verbose:
            print(f"[{dataset}] Fallback endpoints: " f"{sorted(fallback_endpoints)}")
            print(f"[{dataset}] Fallback sites: " f"{sorted(fallback_sites)}")

    if verbose:
        unmapped_endpoints = {
            endpoint
            for endpoint in errors["endpoints"]
            if endpoint not in xroot_to_site
        }

        for endpoint in sorted(unmapped_endpoints):
            print(
                f"[{dataset}] Could not map XRootD endpoint "
                f"'{endpoint}' to a site name"
            )

    return errors


def group_datasets_by_sites(dataset_sites):
    """
    Group datasets by identical sets of failing sites.

    Many datasets often fail on exactly the same sites.
    Rather than building input filesets separately
    for each dataset, we group datasets by their failing-site
    signature and build filesets once per group.

    Example:
        Input:
            {
              "A": {"T2_US_Florida", "T2_FR_GRIF"},
              "B": {"T2_US_Florida", "T2_FR_GRIF"},
              "C": {"T2_IT_Pisa"},
            }

        Output:
            {
              frozenset({"T2_US_Florida", "T2_FR_GRIF"}): ["A", "B"],
              frozenset({"T2_IT_Pisa"}): ["C"],
            }

    Parameters
    ----------
    dataset_sites : dict
        dataset -> set of failing sites

    Returns
    -------
    dict
        frozenset(failing sites) -> list of datasets
    """
    groups = defaultdict(list)
    for dataset, sites in dataset_sites.items():
        groups[frozenset(sites)].append(dataset)
    return groups


def update_input_filesets_for_group(
    sites_to_blacklist,
    year,
    fileset_dir,
    job_dir,
    datasets,
):
    """
    Build input filesets for a group of datasets that share the same
    failing sites.

    The procedure is:
      1. Reset all sites to "white" (allowed)
      2. Blacklist only the sites known to be problematic for this group
      3. Run fetch.py once for all datasets in the group
      4. Build partition files (partitions.json) for each dataset

    This minimizes expensive calls to fetch.py while still preserving
    dataset-level correctness.

    Parameters
    ----------
    sites_to_blacklist : iterable
        Collection of site names to blacklist for this group
    year : str
        Dataset year
    fileset_dir : Path
        Directory containing JSON filesets
    job_dir : Path
        Directory containing Condor job files
    datasets : list
        List of dataset names to build
    """
    print("------------------------------------------------------------")
    print("BUILDING FILESETS FOR DATASET GROUP")
    print("------------------------------------------------------------")
    print(f"Datasets           : {datasets}")
    print(f"Blacklisted sites  : {sorted(sites_to_blacklist)}")

    # First, reset all sites to "white" to ensure no cross-group contamination.
    # This guarantees that each group is created with exactly its own
    # blacklist, and nothing else.
    for site in xroot_to_site.values():
        modify_site_list(year, site, "white")

    # Apply the blacklist for this specific group.
    for site in sites_to_blacklist:
        modify_site_list(year, site, "black")

    # Run fetch.py once for the entire dataset group.
    print("Running fetch.py for this group...")
    subprocess.run(
        [
            "python3",
            "fetch.py",
            "--year",
            year,
            "--samples",
            *datasets,
        ],
        check=True,
    )

    # Load the created filesets
    fileset_path = fileset_dir / f"fileset_{year}_NANO_lxplus.json"
    all_filesets = json.loads(fileset_path.read_text())

    # Build partitions.json for each dataset in the group
    for dataset in datasets:
        if dataset not in all_filesets:
            print(f"[{dataset}] Not found in fileset JSON — skipping")
            continue

        root_files = all_filesets[dataset]
        args_json = job_dir / dataset / "arguments.json"
        if not args_json.exists():
            print(f"[{dataset}] Missing arguments.json — cannot repartition")
            continue

        nfiles = json.loads(args_json.read_text())["nfiles"]
        metadata = all_filesets[dataset].get("metadata", {})
        root_files_list = divide_list(list(root_files["files"].keys()), nfiles)

        partition_dataset = {
            i
            + 1: {
                f"{dataset}_{i+1}" if len(root_files_list) > 1 else dataset: {
                    "files": {filename: "Events" for filename in chunk},
                    "metadata": metadata,
                }
            }
            for i, chunk in enumerate(root_files_list)
        }
        partition_file = job_dir / dataset / "partitions.json"
        with open(partition_file, "w") as json_file:
            json.dump(partition_dataset, json_file, indent=4)

    print("Fileset for this group completed.\n")


def update_partitions_to_global(job_dir, datasets):
    """Replace XRootD endpoints with the CMS global redirector."""
    global_redirector = "root://cms-xrd-global.cern.ch//store/"

    for dataset in datasets:
        partition_file = job_dir / dataset / "partitions.json"

        if not partition_file.exists():
            print(f"[{dataset}] Missing partitions.json — skipping")
            continue

        partitions = json.loads(partition_file.read_text())

        for partition in partitions.values():
            for sample in partition.values():
                sample["files"] = {
                    (
                        global_redirector + filename.split("/store/", 1)[1]
                        if "/store/" in filename
                        else filename
                    ): tree
                    for filename, tree in sample["files"].items()
                }

        partition_file.write_text(json.dumps(partitions, indent=4))
        print(f"[{dataset}] Updated partitions to CMS global redirector")


def resubmit_jobs(job_dir, jobnum_missing, datasets_with_missing_jobs, workflow, year):
    """
    Prepare and resubmit only the jobs that are missing output files.

    For each affected dataset:
      1. Write a missing.txt file listing the missing job numbers
      2. Patch the Condor submission file to use missing.txt instead of jobnum.txt
      3. Submit the modified job description to Condor

    A backup of the original submission file (*_all.sub) is created once
    per dataset to avoid destructive overwrites.

    Parameters
    ----------
    job_dir : Path
        Directory containing Condor job files
    jobnum_missing : dict
        dataset -> set of missing job numbers
    datasets_with_missing_jobs : list
        List of dataset names with missing jobs
    workflow : str
        Workflow name
    year : str
        Dataset year
    """
    print("------------------------------------------------------------")
    print("RESUBMITTING MISSING JOBS")
    print("------------------------------------------------------------")

    to_resubmit = []
    for dataset in datasets_with_missing_jobs:
        missing_jobs = sorted(jobnum_missing[dataset])
        missing_file = job_dir / dataset / "missing.txt"
        with open(missing_file, "w") as f:
            print(*missing_jobs, sep="\n", file=f)

        condor_file = job_dir / dataset / f"{workflow}_{dataset}.sub"
        condor_backup = condor_file.with_name(condor_file.stem + "_all.sub")
        if not condor_backup.exists():
            subprocess.run(["cp", str(condor_file), str(condor_backup)], check=True)

        submit_text = condor_file.read_text().replace("jobnum.txt", "missing.txt")
        condor_file.write_text(submit_text)
        to_resubmit.append(str(condor_file))

    for submit_file in to_resubmit:
        print(f"Submitting {submit_file}")
        subprocess.run(["condor_submit", submit_file], check=True)

    print("Job resubmission completed.\n")


def main():
    args = parse_args()

    # Directory setup
    base_dir = Path.cwd()
    output_dir = Path(make_output_directory(args))
    condor_dir = base_dir / "condor"
    workflow_dir = f"{args.workflow}/{args.label}" if args.label else args.workflow
    job_dir = condor_dir / workflow_dir / args.year
    log_dir = condor_dir / "logs" / workflow_dir / args.year
    fileset_dir = base_dir / "analysis" / "filesets"

    # Optional full reset
    if args.reset:
        print("-" * 60)
        print("RESET REQUESTED")
        print("-" * 60)

        subprocess.run(["rm", "-rf", job_dir], check=False)
        subprocess.run(["rm", "-rf", log_dir], check=False)
        subprocess.run(
            ["rm", "-rf", fileset_dir / f"{args.year}_sites.yaml"],
            check=False,
        )
        subprocess.run(
            [
                "rm",
                "-rf",
                fileset_dir / f"fileset_{args.year}_NANO_lxplus.json",
            ],
            check=False,
        )

        reset_cmd = [
            "python3",
            "runner.py",
            "-w",
            args.workflow,
            "-y",
            args.year,
            "--output_format",
            args.output_format,
            "-m",
            args.memory,
            "--executor",
            args.executor,
        ]

        if args.eos:
            reset_cmd.append("--eos")
        if args.label:
            reset_cmd.extend(["-l", args.label])

        print("Re-running full workflow via runner.py...")
        subprocess.run(reset_cmd, check=True)
        print("Reset complete.\n")

    # Discover job state
    jobnum, jobnum_done, error_files = get_jobs_info(
        job_dir,
        output_dir,
        log_dir,
        args.output_format,
        args.hours_ago,
    )

    # Calculate missing jobs and print the overall summary.
    jobnum_missing, datasets_with_missing_jobs = print_job_status(
        jobnum,
        jobnum_done,
        show_missing_datasets=False,
    )

    if not datasets_with_missing_jobs:
        return

    missing_datasets = sorted(datasets_with_missing_jobs)

    # The global mode modifies partitions.json for datasets with missing jobs.
    if getattr(args, "global"):
        print(
            f"Datasets with missing jobs " f"({len(missing_datasets)}/{len(jobnum)}):"
        )
        print("-" * 60)

        for dataset in missing_datasets:
            missing_partitions = get_missing_job_partitions(
                job_dir,
                dataset,
            )
            partitions = sorted(
                missing_partitions,
                key=lambda partition: (
                    not partition.isdigit(),
                    int(partition) if partition.isdigit() else partition,
                ),
            )
            partitions_text = ", ".join(partitions) or "not identified"

            print(f"- [{dataset}] (Partitions: {partitions_text})")

        update_partitions_to_global(job_dir, missing_datasets)

    else:
        errors_by_dataset = {}

        for dataset in missing_datasets:
            err_files = error_files.get(dataset, [])

            if not err_files and args.verbose:
                print(
                    f"[{dataset}] No recent error logs were found. "
                    "Trying endpoint discovery from missing partitions."
                )

            errors_by_dataset[dataset] = analyze_xrootd_errors_by_dataset(
                err_files,
                job_dir=job_dir,
                dataset=dataset,
                verbose=args.verbose,
            )

        # Print one concise line per dataset.
        print(
            f"Datasets with missing jobs " f"({len(missing_datasets)}/{len(jobnum)}):"
        )
        print("-" * 60)

        for dataset in missing_datasets:
            errors = errors_by_dataset.get(dataset, {})

            missing_partitions = get_missing_job_partitions(
                job_dir,
                dataset,
            )
            partitions = sorted(
                missing_partitions,
                key=lambda partition: (
                    not partition.isdigit(),
                    int(partition) if partition.isdigit() else partition,
                ),
            )
            sites = sorted(errors.get("sites", set()))

            partition_text = ", ".join(partitions) or "not identified"
            site_text = ", ".join(sites) or "not identified"

            print(
                f"- [{dataset}] "
                f"(Partitions: {partition_text}) "
                f"(Sites: {site_text})"
            )

        # Print detailed diagnostics only with --verbose.
        if args.verbose:
            print("\n" + "=" * 60)
            print("DETAILED XROOTD ERROR ANALYSIS")
            print("=" * 60)

            for dataset in missing_datasets:
                errors = errors_by_dataset.get(dataset, {})

                print(f"\nDataset: {dataset}")
                print(
                    "  Error types: "
                    + (
                        ", ".join(sorted(errors.get("error_types", set())))
                        or "unrecognized or no logs"
                    )
                )
                print(
                    "  Endpoints: "
                    + (
                        ", ".join(sorted(errors.get("endpoints", set())))
                        or "none identified"
                    )
                )
                print(
                    "  Sites: "
                    + (
                        ", ".join(sorted(errors.get("sites", set())))
                        or "none identified"
                    )
                )
                print(
                    "  Current missing-job partitions: "
                    + (
                        ", ".join(
                            sorted(
                                get_missing_job_partitions(job_dir, dataset),
                                key=lambda partition: (
                                    not partition.isdigit(),
                                    (
                                        int(partition)
                                        if partition.isdigit()
                                        else partition
                                    ),
                                ),
                            )
                        )
                        or "not identified"
                    )
                )

        # Change the executor only when explicitly requested.
        if args.executor == "iterative":
            for dataset in sorted(errors_by_dataset):
                update_executor_for_dataset(
                    job_dir,
                    dataset,
                    executor="iterative",
                )

        # Group only datasets for which site names were identified.
        dataset_sites = {
            dataset: errors["sites"]
            for dataset, errors in errors_by_dataset.items()
            if errors.get("sites")
        }

        if not dataset_sites:
            if args.verbose:
                print(
                    "\nNo problematic sites could be mapped to site names. "
                    "Input filesets will not be regenerated."
                )
        else:
            if args.single_fetch:
                all_sites = set().union(*dataset_sites.values())
                all_datasets = sorted(dataset_sites)
                groups = {frozenset(all_sites): all_datasets}
            else:
                groups = group_datasets_by_sites(dataset_sites)

            if args.verbose:
                print("\nDatasets grouped by problematic sites:")
                for sites, datasets in groups.items():
                    print(f"  Sites: {', '.join(sorted(sites))}")
                    print(f"  Datasets: {', '.join(sorted(datasets))}")

            if timed_input(
                "\nBuild input filesets for these groups? (y/n): "
            ).strip().lower() in {"y", "yes"}:
                for sites, datasets in groups.items():
                    update_input_filesets_for_group(
                        sites,
                        args.year,
                        fileset_dir,
                        job_dir,
                        datasets,
                    )

    # Optional job resubmission.
    if timed_input("Resubmit missing jobs now? (y/n): ").strip().lower() in {
        "y",
        "yes",
    }:
        resubmit_jobs(
            job_dir,
            jobnum_missing,
            missing_datasets,
            args.workflow,
            args.year,
        )


if __name__ == "__main__":
    main()
