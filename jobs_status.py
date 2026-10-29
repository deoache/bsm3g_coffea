"""
Inspect Condor job outputs, identify missing jobs, analyze recent XRootD
failures, and optionally regenerate input filesets with problematic sites
blacklisted before resubmitting only the missing jobs.
"""

import re
import sys
import yaml
import json
import select
import argparse
import subprocess
from pathlib import Path
from collections import defaultdict
from urllib.parse import urlsplit
from datetime import datetime, timedelta
from analysis.utils import make_output_directory
from analysis.filesets.xrootd_sites import xroot_to_site
from analysis.filesets.utils import divide_list

ENDPOINT_PATTERN = r"root://[a-zA-Z0-9._-]+(?::\d+)?"

EXECUTOR_ERROR_PATTERNS = {
    "Authentication": [
        r"Auth failed",
        r"No protocols left to try",
    ],
    "Connection": [
        r"\[FATAL\]\s*Connection error",
        r"Connection reset",
        r"Connection refused",
        r"Network is unreachable",
        r"XRootD error:.*Connection error",
        r"\bConnectionError\b",
    ],
    "TLS": [
        r"\bTLS error\b",
        r"SSL error",
    ],
}

FILE_ERROR_PATTERNS = {
    "File access / permission": [
        r"\[3010\]\s*Unable to open file",
        r"Unable to open file",
        r"Failed to open file",
        r"Error opening file",
        r"Operation not permitted",
        r"Permission denied",
    ],
    "File not found": [
        r"File not found",
        r"No such file or directory",
        r"\bFileNotFoundError\b",
    ],
    "Read error": [
        r"Read error",
        r"Input/output error",
        r"\bIOError\b",
    ],
    "Timeout": [
        r"Operation expired",
        r"timed?\s*out",
        r"\btimeout\b",
        r"\bTimeoutError\b",
    ],
    "Socket / network": [
        r"Socket error",
        r"XrdCl::",
    ],
    "Other I/O error": [
        r"\bOSError\b",
        r"XRootD error",
    ],
}

ERROR_PATTERNS = {
    **EXECUTOR_ERROR_PATTERNS,
    **FILE_ERROR_PATTERNS,
}


def find_nano_fileset(fileset_dir: Path, year: str) -> Path | None:
    """Find the existing NanoAOD-versioned fileset for a year, if present."""
    matches = sorted(fileset_dir.glob(f"fileset_{year}_NANO_lxplus.json"))
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple NanoAOD filesets found for {year}: "
            + ", ".join(path.name for path in matches)
        )
    return matches[0] if matches else None


def parse_args():
    """Parse command-line arguments."""
    workflows_dir = Path.cwd() / "analysis" / "workflows"

    parser = argparse.ArgumentParser(
        description=(
            "Inspect Condor job outputs, diagnose XRootD failures, "
            "regenerate filesets, and resubmit missing jobs."
        )
    )

    parser.add_argument(
        "-w",
        "--workflow",
        required=True,
        choices=[path.stem for path in workflows_dir.glob("*.yaml")],
        help="Workflow name corresponding to a YAML file in analysis/workflows",
    )
    parser.add_argument(
        "-y",
        "--year",
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
        "--output_format",
        default="coffea",
        choices=["coffea", "parquet"],
        help="Output format of the produced histograms",
    )
    parser.add_argument(
        "--eos",
        action="store_true",
        help="Read job outputs from EOS instead of local storage",
    )
    parser.add_argument(
        "--global",
        dest="use_global",
        action="store_true",
        help="Replace URLs only in missing partitions with the CMS global redirector",
    )
    parser.add_argument(
        "-m",
        "--memory",
        type=str,
        default="4000",
        help="Requested memory (in MB) for Condor jobs",
    )
    parser.add_argument(
        "-l",
        "--label",
        type=str,
        default="",
        help="Output directory label",
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
        help="Delete existing job outputs, logs, and filesets before rerunning",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show detailed XRootD error diagnostics",
    )
    parser.add_argument(
        "-e",
        "--executor",
        choices=["iterative", "futures"],
        default="futures",
        help="Executor to use when rerunning the workflow",
    )

    return parser.parse_args()


def timed_input(prompt, timeout=10, default="y"):
    """Read user input with a timeout and return the default if necessary."""
    print(
        f"{prompt} [{default.upper()}/n] (auto-selected in {timeout}s): ",
        end="",
        flush=True,
    )

    ready, _, _ = select.select([sys.stdin], [], [], timeout)

    if ready:
        response = sys.stdin.readline().strip()
        return response.lower() if response else default.lower()

    print(f"\n[Timeout reached] Using default option: '{default.upper()}'")
    return default.lower()


def modify_site_list(year: str, site: str, status: str) -> None:
    """Move a site to either the white or black list in the year's YAML file."""
    sites_file = Path.cwd() / "analysis" / "filesets" / f"{year}_sites.yaml"
    data = yaml.safe_load(sites_file.read_text())

    if status == "white":
        if site not in data["white"]:
            data["white"].append(site)
        if site in data["black"]:
            data["black"].remove(site)
    else:
        if site not in data["black"]:
            data["black"].append(site)
        if site in data["white"]:
            data["white"].remove(site)

    sites_file.write_text(yaml.dump(data, default_flow_style=False))


def identify_error_types(content: str) -> set[str]:
    """Identify recognized XRootD, connection, and file-access errors."""
    return {
        error_type
        for error_type, patterns in ERROR_PATTERNS.items()
        if any(re.search(pattern, content, flags=re.IGNORECASE) for pattern in patterns)
    }


def get_missing_job_partitions(job_dir: Path, dataset: str) -> set[str]:
    """Read missing partition identifiers from missing.txt, if available."""
    missing_path = Path(job_dir) / dataset / "missing.txt"

    if not missing_path.is_file():
        return set()

    return {
        line.strip() for line in missing_path.read_text().splitlines() if line.strip()
    }


def get_partition_endpoints(
    partitions_data: dict,
    missing_partitions: set[str],
    endpoint_pattern: re.Pattern,
) -> set[str]:
    """Extract XRootD endpoints from the specified missing partitions."""
    endpoints = set()

    for partition_id in missing_partitions:
        partition_data = partitions_data.get(partition_id, {})

        for sample_data in partition_data.values():
            for url in sample_data.get("files", {}):
                match = endpoint_pattern.match(url)
                if match:
                    endpoints.add(match.group(0))

    return endpoints


def analyze_xrootd_error_logs(
    error_files,
    job_dir: Path,
    dataset: str,
    missing_partitions=None,
    verbose: bool = False,
) -> dict:
    """
    Identify error types and XRootD endpoints for one dataset.

    Endpoints are extracted from recent error logs first. If no endpoint
    is found in the logs, the function falls back to the files belonging
    to the currently missing partitions.
    """
    dataset_dir = Path(job_dir) / dataset
    partitions_path = dataset_dir / "partitions.json"

    if missing_partitions is None:
        missing_partitions = get_missing_job_partitions(job_dir, dataset)

    missing_partitions = {str(partition) for partition in missing_partitions}

    if partitions_path.is_file():
        partitions_data = json.loads(partitions_path.read_text())
    else:
        partitions_data = {}
        if verbose:
            print(f"[{dataset}] Missing partitions.json: {partitions_path}")

    endpoint_pattern = re.compile(
        ENDPOINT_PATTERN,
        flags=re.IGNORECASE,
    )

    endpoints = set()
    error_types = set()

    for error_file in error_files:
        error_file = Path(error_file)

        try:
            content = error_file.read_text(errors="replace")
        except OSError as exc:
            if verbose:
                print(f"[{dataset}] Could not read {error_file}: {exc}")
            continue

        endpoints.update(endpoint_pattern.findall(content))
        error_types.update(identify_error_types(content))

    if not endpoints and missing_partitions:
        endpoints = get_partition_endpoints(
            partitions_data,
            missing_partitions,
            endpoint_pattern,
        )

        if verbose:
            print(f"[{dataset}] Fallback endpoints: {sorted(endpoints)}")

    return {
        "endpoints": endpoints,
        "error_types": error_types,
        "partitions": missing_partitions,
    }


def get_jobs_info(
    job_dir: Path,
    output_dir: Path,
    log_dir: Path,
    output_format: str,
    hours_ago: int = 3,
):
    """
    Collect expected jobs, completed jobs, and recent error logs per dataset.

    Returns
    -------
    jobnum : dict
        Dataset -> list of expected job numbers.
    jobnum_done : dict
        Dataset -> list of job numbers with output files.
    error_files : dict
        Dataset -> list of recent .err file paths.
    """
    jobnum = {}
    jobnum_done = {}
    error_files = {}

    cutoff_time = datetime.now() - timedelta(hours=hours_ago)

    for dataset_dir in job_dir.iterdir():
        if not dataset_dir.is_dir():
            continue

        dataset = dataset_dir.name
        jobnum_path = dataset_dir / "jobnum.txt"

        if not jobnum_path.is_file():
            raise FileNotFoundError(
                f"Missing jobnum.txt for dataset '{dataset}'. "
                f"Expected at: {jobnum_path}"
            )

        jobnum[dataset] = jobnum_path.read_text().splitlines()

        output_files = list((output_dir / dataset).glob(f"*.{output_format}"))
        jobnum_done[dataset] = [
            path.stem.replace(f"{dataset}_", "") for path in output_files
        ]

        dataset_log_dir = log_dir / dataset
        if dataset_log_dir.is_dir():
            recent_errors = [
                path
                for path in dataset_log_dir.glob("*.err")
                if datetime.fromtimestamp(path.stat().st_mtime) > cutoff_time
            ]

            if recent_errors:
                error_files[dataset] = recent_errors

    return jobnum, jobnum_done, error_files


def print_job_status(jobnum, jobnum_done, verbose=False):
    """
    Print job completion totals and return missing jobs per dataset.

    Returns
    -------
    jobnum_missing : dict
        Dataset -> set of missing job numbers.
    datasets_with_missing : list
        Datasets with at least one missing job.
    """
    jobnum_missing = {
        dataset: set(expected) - set(jobnum_done.get(dataset, []))
        for dataset, expected in jobnum.items()
    }

    n_expected = sum(len(jobs) for jobs in jobnum.values())
    n_done = sum(len(jobs) for jobs in jobnum_done.values())
    n_missing = sum(len(jobs) for jobs in jobnum_missing.values())

    print("-" * 60)
    print("JOB STATUS SUMMARY")
    print("-" * 60)
    print(f"Total jobs expected : {n_expected}")
    print(f"Total jobs finished : {n_done}")
    print(f"Total jobs missing  : {n_missing}")

    datasets_with_missing = [
        dataset for dataset, missing in jobnum_missing.items() if missing
    ]

    if n_missing and not verbose:
        print("-" * 60)
        print(
            f"Datasets with missing jobs "
            f"({len(datasets_with_missing)}/{len(jobnum)}):"
        )
        print("-" * 60)

        for dataset in sorted(datasets_with_missing):
            expected = set(jobnum[dataset])
            completed = expected & set(jobnum_done.get(dataset, []))
            print(f"- {dataset} ({len(completed)}/{len(expected)})")

    return jobnum_missing, datasets_with_missing


def update_executor_for_dataset(
    job_dir: Path,
    dataset: str,
    executor: str = "iterative",
) -> None:
    """Update the executor in a dataset's arguments.json."""
    args_json = job_dir / dataset / "arguments.json"

    if not args_json.is_file():
        print(f"[{dataset}] Missing arguments.json — cannot update executor")
        return

    arguments = json.loads(args_json.read_text())

    if arguments.get("executor") == executor:
        print(f"[{dataset}] Executor is already '{executor}'")
        return

    arguments["executor"] = executor
    args_json.write_text(json.dumps(arguments, indent=4) + "\n")

    print(f"[{dataset}] Updated executor to '{executor}'")


def map_endpoints_to_sites(endpoints: set[str]) -> set[str]:
    """Map known XRootD endpoints to CMS site names."""
    return {
        xroot_to_site[endpoint] for endpoint in endpoints if endpoint in xroot_to_site
    }


def analyze_xrootd_errors_by_dataset(
    error_files,
    job_dir,
    dataset,
    missing_partitions,
    verbose=False,
):
    """Summarize errors for a dataset without overriding unmapped log endpoints."""
    errors = analyze_xrootd_error_logs(
        error_files=error_files,
        job_dir=job_dir,
        dataset=dataset,
        missing_partitions=missing_partitions,
        verbose=verbose,
    )
    errors["sites"] = map_endpoints_to_sites(errors["endpoints"])

    for endpoint in sorted(errors["endpoints"]):
        if endpoint not in xroot_to_site:
            print(
                f"[{dataset}] Could not map XRootD endpoint "
                f"'{endpoint}' to a site name"
            )

    if not errors["sites"]:
        print(
            f"[{dataset}] No CMS site could be identified "
            "from the error logs or missing partitions."
        )

    return errors


def group_datasets_by_sites(dataset_sites):
    """Group datasets that share exactly the same set of failing sites."""
    groups = defaultdict(list)

    for dataset, sites in dataset_sites.items():
        groups[frozenset(sites)].append(dataset)

    return groups


def regenerate_partitions(
    dataset: str,
    current_fileset: dict,
    job_dir: Path,
) -> None:
    """Regenerate partitions.json for a dataset using its current arguments."""
    args_json = job_dir / dataset / "arguments.json"

    if not args_json.is_file():
        print(
            f"[{dataset}] Missing {args_json} — cannot repartition",
            flush=True,
        )
        return

    fileset_files = current_fileset["files"]
    root_files = list(fileset_files.items())
    nfiles = json.loads(args_json.read_text())["nfiles"]
    fileset_chunks = divide_list(root_files, nfiles)

    partition_dataset = {}

    for index, chunk in enumerate(fileset_chunks, start=1):
        dataset_key = f"{dataset}_{index}" if len(fileset_chunks) > 1 else dataset

        partition_dataset[index] = {
            dataset_key: {
                "files": dict(chunk),
                "metadata": current_fileset.get("metadata", {}),
            }
        }

    partition_file = job_dir / dataset / "partitions.json"
    partition_file.write_text(json.dumps(partition_dataset, indent=4))


def file_identity(url: str) -> str:
    """Return a replica-independent CMS logical file name."""
    path = urlsplit(url).path
    marker = "/store/"
    if marker not in path:
        raise ValueError(f"Cannot identify CMS LFN from URL: {url}")
    return path[path.index(marker) :]


def get_partition_sites(partitions_data: dict, partition_id: str) -> set[str]:
    """Identify sites used by one partition, not the entire dataset."""
    endpoints = get_partition_endpoints(
        partitions_data, {str(partition_id)}, re.compile(ENDPOINT_PATTERN, re.I)
    )
    return map_endpoints_to_sites(endpoints)


def get_failed_root_files(error_files, dataset: str, partition_id: str) -> set[str]:
    """Extract ROOT URLs from log entries belonging to exactly one WorkItem partition."""
    expected_work_item = f"{dataset}_{partition_id}"
    work_item_pattern = re.compile(
        r"WorkItem\(dataset=['\"](?P<dataset>[^'\"]+)['\"],\s*"
        r"filename=['\"](?P<url>root://[^'\"]+?\.root)['\"]",
        flags=re.IGNORECASE,
    )
    failed_urls = set()

    for error_file in error_files:
        try:
            content = Path(error_file).read_text(errors="replace")
        except OSError:
            continue

        for match in work_item_pattern.finditer(content):
            if match.group("dataset") == expected_work_item:
                failed_urls.add(match.group("url"))

    return failed_urls


def get_partition_failure_sites(
    error_files,
    dataset: str,
    partition_id: str,
    partitions_data: dict,
) -> tuple[set[str], set[str], bool]:
    """Return sites for one partition, preferring ROOT URLs recorded in its .err logs.

    If a matching log contains a ROOT URL but its endpoint is unmapped, do not
    fall back to the partition's existing URLs: that could blacklist an unrelated site.
    """
    failed_urls = get_failed_root_files(error_files, dataset, str(partition_id))

    if failed_urls:
        endpoints = {
            match.group(0)
            for url in failed_urls
            if (match := re.match(ENDPOINT_PATTERN, url, flags=re.IGNORECASE))
        }
        sites = map_endpoints_to_sites(endpoints)
        return failed_urls, sites, True

    sites = get_partition_sites(partitions_data, str(partition_id))
    return set(), sites, False


def regenerate_missing_partitions(
    year: str,
    fileset_dir: Path,
    job_dir: Path,
    dataset_partitions: dict[str, set[str]],
    error_files_by_dataset: dict[str, list[Path]],
    verbose: bool = False,
) -> None:
    """Regenerate each selected partition using its own site blacklist.

    Each fetch.py call is restricted to the dataset containing the partition
    being repaired. Existing partition IDs, trees, and file lists are preserved;
    replacement URLs are matched by CMS LFN.
    """
    sites_path = fileset_dir / f"{year}_sites.yaml"
    fileset_path = find_nano_fileset(fileset_dir, year)
    original_fileset = fileset_path.read_bytes() if fileset_path is not None else None

    tasks = []
    for dataset, partition_ids in sorted(dataset_partitions.items()):
        partition_path = job_dir / dataset / "partitions.json"
        if not partition_path.is_file():
            print(f"[{dataset}] Missing partitions.json; skipping")
            continue

        partitions = json.loads(partition_path.read_text())
        err_files = error_files_by_dataset.get(dataset, [])
        for partition_id in sort_job_numbers(partition_ids):
            partition_id = str(partition_id)
            entry = partitions.get(partition_id)
            if not entry or len(entry) != 1:
                print(f"[{dataset}:{partition_id}] Invalid/missing partition; skipping")
                continue

            _, sample_data = next(iter(entry.items()))
            old_files = sample_data.get("files", {})
            if not old_files:
                print(f"[{dataset}:{partition_id}] No files in partition; skipping")
                continue

            failed_urls, sites, urls_found_in_logs = get_partition_failure_sites(
                err_files, dataset, partition_id, partitions
            )
            if not sites:
                if urls_found_in_logs:
                    print(
                        f"[{dataset}:{partition_id}] Failed ROOT URL found in .err, "
                        "but its endpoint could not be mapped; skipping blacklist"
                    )
                else:
                    print(f"[{dataset}:{partition_id}] No mapped sites; skipping")
                continue

            try:
                old_by_lfn = {
                    file_identity(url): tree for url, tree in old_files.items()
                }
            except ValueError as exc:
                print(f"[{dataset}:{partition_id}] {exc}; skipping")
                continue

            tasks.append(
                {
                    "dataset": dataset,
                    "partition_id": partition_id,
                    "partition_path": partition_path,
                    "sites": frozenset(sites),
                    "old_by_lfn": old_by_lfn,
                }
            )

    updated = 0
    try:
        for task in tasks:
            sites = task["sites"]
            datasets = [task["dataset"]]
            labels = f"{task['dataset']}:{task['partition_id']}"
            print(f"[{labels}] Blacklisting: {', '.join(sorted(sites))}")

            all_sites = set(xroot_to_site.values())
            if not all_sites:
                raise RuntimeError("No CMS sites found in xrootd_sites.py")
            if not sites.issubset(all_sites):
                raise RuntimeError(
                    f"[{labels}] Unknown CMS sites: {sorted(sites - all_sites)}"
                )

            sites_config = {
                "white": sorted(all_sites - sites),
                "black": sorted(sites),
            }
            sites_path.write_text(yaml.safe_dump(sites_config, sort_keys=False))
            if verbose:
                print(f"[{labels}] Sites YAML: {sites_path.resolve()}")
                print(f"[{labels}] Whitelisted sites: {len(sites_config['white'])}")
                print(f"[{labels}] Blacklisted sites: {sites_config['black']}")

            # Remove stale output first; never consume a fileset from a failed fetch.
            if fileset_path is not None and fileset_path.exists():
                fileset_path.unlink()

            command = [
                "python3",
                "fetch.py",
                "--year",
                year,
                "--samples",
                " ".join(datasets),
                "--skip_site",
            ]
            result = subprocess.run(command, check=False)
            if result.returncode != 0:
                print(f"[{labels}] fetch.py failed; partitions unchanged")
                continue

            if fileset_path is None or not fileset_path.is_file():
                fileset_path = find_nano_fileset(fileset_dir, year)
            if fileset_path is None or not fileset_path.is_file():
                print(
                    f"[{labels}] fetch.py did not produce a NanoAOD-versioned fileset"
                )
                continue

            try:
                fetched = json.loads(fileset_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                print(f"[{labels}] Could not read generated fileset: {exc}")
                continue

            filesets_by_short_name = defaultdict(list)
            for sample in fetched.values():
                short_name = sample.get("metadata", {}).get("short_name")
                if short_name:
                    filesets_by_short_name[short_name].append(sample)

            dataset = task["dataset"]
            partition_id = task["partition_id"]
            candidates = filesets_by_short_name.get(dataset, [])
            if len(candidates) != 1:
                print(
                    f"[{dataset}:{partition_id}] Sample not uniquely found "
                    "in generated fileset; unchanged"
                )
                continue

            try:
                new_by_lfn = {
                    file_identity(url): url for url in candidates[0].get("files", {})
                }
            except ValueError as exc:
                print(f"[{dataset}:{partition_id}] {exc}; unchanged")
                continue

            old_by_lfn = task["old_by_lfn"]
            missing_lfns = set(old_by_lfn) - set(new_by_lfn)
            if missing_lfns:
                print(
                    f"[{dataset}:{partition_id}] {len(missing_lfns)} original files "
                    "not found with this blacklist; partition unchanged"
                )
                continue

            partition_path = task["partition_path"]
            partitions = json.loads(partition_path.read_text())
            entry = partitions.get(partition_id)
            if not entry or len(entry) != 1:
                print(
                    f"[{dataset}:{partition_id}] Partition changed during fetch; unchanged"
                )
                continue

            _, sample_data = next(iter(entry.items()))
            sample_data["files"] = {
                new_by_lfn[lfn]: tree for lfn, tree in old_by_lfn.items()
            }
            partition_path.write_text(json.dumps(partitions, indent=4) + "\n")
            updated += 1
            if verbose:
                print(
                    f"[{dataset}:{partition_id}] Updated "
                    f"{len(old_by_lfn)} file URL(s)"
                )
    finally:
        # Keep the latest per-partition sites.yaml configuration for inspection.
        # Restore the full original fileset because fetch.py generates a subset
        # containing only the sample processed in the current task.
        if original_fileset is not None and fileset_path is not None:
            fileset_path.write_bytes(original_fileset)
        elif fileset_path is not None and fileset_path.exists():
            fileset_path.unlink()

    print(f"Partition regeneration completed: {updated} partition(s) updated.")


def update_missing_partitions_to_global(
    job_dir: Path, dataset_partitions: dict[str, set[str]]
) -> None:
    """Replace URLs with the CMS global redirector in missing partitions only."""
    global_redirector = "root://cms-xrd-global.cern.ch//store/"

    for dataset, partition_ids in sorted(dataset_partitions.items()):
        partition_file = job_dir / dataset / "partitions.json"
        if not partition_file.is_file():
            print(f"[{dataset}] Missing partitions.json — skipping")
            continue

        partitions = json.loads(partition_file.read_text())
        changed = False
        for partition_id in sort_job_numbers(partition_ids):
            entry = partitions.get(str(partition_id))
            if not entry:
                print(f"[{dataset}:{partition_id}] Missing partition — skipping")
                continue
            for sample in entry.values():
                files = sample.get("files", {})
                sample["files"] = {
                    (
                        global_redirector + url.split("/store/", 1)[1]
                        if "/store/" in url
                        else url
                    ): tree
                    for url, tree in files.items()
                }
                changed = True
        if changed:
            partition_file.write_text(json.dumps(partitions, indent=4) + "\n")
            print(
                f"[{dataset}] Updated only missing partitions to CMS global redirector"
            )


def sort_job_numbers(job_numbers):
    """Sort numeric job identifiers numerically and other identifiers afterward."""
    return sorted(
        job_numbers,
        key=lambda value: (
            not str(value).isdigit(),
            int(value) if str(value).isdigit() else str(value),
        ),
    )


def resubmit_jobs(
    job_dir: Path,
    jobnum_missing,
    datasets_with_missing_jobs,
    workflow: str,
    year: str,
):
    """
    Prepare and resubmit only jobs without output files.

    The original submission file is backed up once as *_all.sub.
    """
    print("-" * 60)
    print("RESUBMITTING MISSING JOBS")
    print("-" * 60)

    submit_files = []

    for dataset in datasets_with_missing_jobs:
        missing_jobs = sort_job_numbers(jobnum_missing[dataset])
        dataset_dir = job_dir / dataset

        missing_file = dataset_dir / "missing.txt"
        missing_file.write_text("".join(f"{job}\n" for job in missing_jobs))

        condor_file = dataset_dir / f"{workflow}_{dataset}.sub"
        condor_backup = condor_file.with_name(f"{condor_file.stem}_all.sub")

        if not condor_backup.exists():
            subprocess.run(
                ["cp", str(condor_file), str(condor_backup)],
                check=True,
            )

        submit_text = condor_file.read_text().replace(
            "jobnum.txt",
            "missing.txt",
        )
        condor_file.write_text(submit_text)

        submit_files.append(condor_file)

    for submit_file in submit_files:
        print(f"Submitting {submit_file}")
        subprocess.run(
            ["condor_submit", str(submit_file)],
            check=True,
        )

    print("Job resubmission completed.\n")


def reset_workflow(args, job_dir, log_dir, sites_file, fileset_file):
    """Remove existing workflow outputs and rerun runner.py."""
    print("-" * 60)
    print("RESET REQUESTED")
    print("-" * 60)

    subprocess.run(["rm", "-rf", str(job_dir)], check=False)
    subprocess.run(["rm", "-rf", str(log_dir)], check=False)
    subprocess.run(["rm", "-rf", str(sites_file)], check=False)
    if fileset_file is not None:
        subprocess.run(["rm", "-f", str(fileset_file)], check=False)

    # Forward only arguments defined by this script.
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

    print("Rerunning the workflow via runner.py...")
    subprocess.run(reset_cmd, check=True)
    print("Reset complete.\n")


def main():
    args = parse_args()

    base_dir = Path.cwd()
    output_dir = Path(make_output_directory(args))
    condor_dir = base_dir / "condor"
    workflow_dir = f"{args.workflow}/{args.label}" if args.label else args.workflow
    job_dir = condor_dir / workflow_dir / args.year
    log_dir = condor_dir / "logs" / workflow_dir / args.year
    fileset_dir = base_dir / "analysis" / "filesets"

    if not job_dir.is_dir() and not args.reset:
        raise FileNotFoundError(f"Condor job directory does not exist: {job_dir}")

    sites_file = fileset_dir / f"{args.year}_sites.yaml"
    fileset_file = find_nano_fileset(fileset_dir, args.year)

    if args.reset:
        reset_workflow(args, job_dir, log_dir, sites_file, fileset_file)

    jobnum, jobnum_done, error_files = get_jobs_info(
        job_dir, output_dir, log_dir, args.output_format, args.hours_ago
    )
    jobnum_missing, datasets_with_missing_jobs = print_job_status(
        jobnum, jobnum_done, verbose=args.verbose
    )

    if not datasets_with_missing_jobs:
        print("All expected jobs have completed successfully.")
        return

    missing_datasets = sorted(datasets_with_missing_jobs)
    if args.reset:
        print("Workflow reset completed. Skipping XRootD diagnostics and blacklisting.")
        if timed_input("Resubmit missing jobs now?") in {"y", "yes"}:
            resubmit_jobs(
                job_dir, jobnum_missing, missing_datasets, args.workflow, args.year
            )
        return

    # Restrict partition IDs to jobs that are still missing according to outputs.
    dataset_partitions = {}
    for dataset in missing_datasets:
        recorded = get_missing_job_partitions(job_dir, dataset)
        missing = {str(value) for value in jobnum_missing[dataset]}
        selected = recorded & missing
        if selected:
            dataset_partitions[dataset] = selected
        elif args.verbose:
            print(f"[{dataset}] No matching missing partition IDs in missing.txt")

    if getattr(args, "use_global"):
        if dataset_partitions:
            print("Global redirector mode: updating only missing partitions.")
            update_missing_partitions_to_global(job_dir, dataset_partitions)
        else:
            print(
                "No missing partitions could be identified for global redirector update."
            )
    else:
        errors_by_dataset = {}
        for dataset in missing_datasets:
            err_files = error_files.get(dataset, [])
            missing_partitions = dataset_partitions.get(dataset, set())
            errors_by_dataset[dataset] = analyze_xrootd_errors_by_dataset(
                error_files=err_files,
                job_dir=job_dir,
                dataset=dataset,
                missing_partitions=missing_partitions,
                verbose=args.verbose,
            )

        if args.verbose:
            print("-" * 60)
            print(
                f"Datasets with missing jobs ({len(missing_datasets)}/{len(jobnum)}):"
            )
            print("-" * 60)
            for dataset in missing_datasets:
                partitions = sort_job_numbers(dataset_partitions.get(dataset, set()))
                sites = set()
                partition_file = job_dir / dataset / "partitions.json"
                if partition_file.is_file():
                    partitions_data = json.loads(partition_file.read_text())
                    for partition_id in partitions:
                        _, partition_sites, urls_found = get_partition_failure_sites(
                            error_files.get(dataset, []),
                            dataset,
                            partition_id,
                            partitions_data,
                        )
                        if partition_sites:
                            sites.update(partition_sites)
                        elif urls_found:
                            print(
                                f"[{dataset}:{partition_id}] A failed ROOT URL was found, "
                                "but no endpoint could be mapped."
                            )
                print(
                    f"- [{dataset}] (Partitions: {', '.join(partitions) or 'not identified'}) "
                    f"(Sites: {', '.join(sorted(sites)) or 'not identified'})"
                )

        if args.verbose:
            print("\nDETAILED XROOTD ERROR ANALYSIS")
            print("-" * 60)
            for dataset in missing_datasets:
                errors = errors_by_dataset[dataset]
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
                    "  Missing partitions: "
                    + (
                        ", ".join(
                            sort_job_numbers(dataset_partitions.get(dataset, set()))
                        )
                        or "not identified"
                    )
                )

        if args.executor == "iterative":
            for dataset in missing_datasets:
                update_executor_for_dataset(job_dir, dataset, executor="iterative")

        eligible_partitions = {
            dataset: partitions
            for dataset, partitions in dataset_partitions.items()
            if partitions
        }
        if not eligible_partitions:
            print("\nNo eligible missing partitions for fileset regeneration.")
        elif timed_input("\nRegenerate input files for missing partitions?") in {
            "y",
            "yes",
        }:
            regenerate_missing_partitions(
                args.year,
                fileset_dir,
                job_dir,
                eligible_partitions,
                error_files,
                verbose=args.verbose,
            )

    if timed_input("Resubmit missing jobs now?") in {"y", "yes"}:
        resubmit_jobs(
            job_dir, jobnum_missing, missing_datasets, args.workflow, args.year
        )


if __name__ == "__main__":
    main()
