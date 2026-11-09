"""
Inspect Condor job outputs, identify missing jobs, analyze recent XRootD
failures, regenerate input files for missing partitions with partition-specific
site blacklists, and optionally resubmit missing jobs.
Incorporates custom user features (Rucio/DAS, deep-reset, workers, dynamic cpu/memory update, skipbadfiles, global redirector).
"""

import argparse
import json
import re
import select
import subprocess
import sys

from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlsplit

import yaml

from analysis.utils import make_output_directory
from analysis.filesets.xrootd_sites import xroot_to_site
from analysis.filesets.utils import divide_list

ENDPOINT_PATTERN = r"root://[a-zA-Z0-9._-]+(?::\d+)?"

ROOT_URL_PATTERN = re.compile(
    r"""root://[^\s'"<>(),]+?\.root(?=[\s'",)]|$)""",
    flags=re.IGNORECASE,
)

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
        choices=["coffea", "parquet", "root"],
        help="Output format of the produced histograms",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Change the number of workers to process the analysis",
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
        "--deep-reset",
        action="store_true",
        help="Redo fileset discovery (deletes analysis/filesets files too)",
    )
    parser.add_argument(
        "--skipbadfiles",
        action="store_true",
        help="Skip root files with errors",
    )
    parser.add_argument(
        "--single_fetch",
        action="store_true",
        help="Run fetch.py only once (now default era behavior in new script)",
    )
    parser.add_argument(
        "--global",
        dest="use_global",
        action="store_true",
        help="Send xrd-cms-global partition filesets",
    )
    parser.add_argument(
        "-m",
        "--memory",
        type=str,
        default="4000",
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
        "-v",
        "--verbose",
        action="store_true",
        help="Show detailed XRootD error diagnostics",
    )
    parser.add_argument(
        "--executor",
        choices=["iterative", "futures"],
        default="futures",
        help="Executor to use when updating dataset arguments",
    )
    parser.add_argument(
        "-e",
        "--update-executor",
        action="store_true",
        help="Update the executor in arguments.json for affected datasets",
    )

    parser.add_argument(
        "--by_dataset",
        action="store_true",
        help="Diagnose and regenerate separately per dataset instead of using an era-wide blacklist",
    )
    parser.add_argument(
        "--eos",
        action="store_true",
        help="Read job outputs from EOS instead of local storage",
    )
    parser.add_argument(
        "-l",
        "--label",
        type=str,
        default="",
        help="Output directory label",
    )
    
    # Mutually exclusive group for dataset discovery strategy
    discovery_group = parser.add_mutually_exclusive_group()
    discovery_group.add_argument(
        "--das",
        dest="discovery",
        action="store_const",
        const="das",
        help="Use dasgoclient for dataset discovery.",
    )
    discovery_group.add_argument(
        "--rucio",
        dest="discovery",
        action="store_const",
        const="rucio",
        help="Use DataDiscoveryCLI (Rucio) for dataset discovery (Default).",
    )
    parser.set_defaults(discovery="rucio")
    
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

    if not sites_file.is_file():
        raise FileNotFoundError(f"Site configuration not found: {sites_file}")

    data = yaml.safe_load(sites_file.read_text(encoding="utf-8")) or {}
    data.setdefault("white", [])
    data.setdefault("black", [])

    if status == "white":
        if site not in data["white"]:
            data["white"].append(site)
        if site in data["black"]:
            data["black"].remove(site)
    elif status == "black":
        if site not in data["black"]:
            data["black"].append(site)
        if site in data["white"]:
            data["white"].remove(site)
    else:
        raise ValueError(f"Unknown site status: {status}")

    sites_file.write_text(
        yaml.safe_dump(data, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )


def reset_site_blacklist(year: str) -> None:
    """Clear the current blacklist using modify_site_list()."""
    sites_file = Path.cwd() / "analysis" / "filesets" / f"{year}_sites.yaml"

    if not sites_file.is_file():
        raise FileNotFoundError(f"Site configuration not found: {sites_file}")

    data = yaml.safe_load(sites_file.read_text(encoding="utf-8")) or {}

    for site in list(data.get("black", [])):
        modify_site_list(year, site, "white")


def identify_error_types(content: str) -> set[str]:
    """Identify recognized XRootD, connection, and file-access errors."""
    return {
        error_type
        for error_type, patterns in ERROR_PATTERNS.items()
        if any(re.search(pattern, content, flags=re.IGNORECASE) for pattern in patterns)
    }


def get_missing_job_partitions(jobnum_missing: dict[str, set[str]], dataset: str,) -> set[str]:
    """Return the missing partition identifiers for a dataset."""
    return set(jobnum_missing.get(dataset, set()))


def get_partition_endpoints(
    partitions_data: dict,
    missing_partitions: set[str],
    endpoint_pattern: re.Pattern,
) -> set[str]:
    """Extract XRootD endpoints from the specified partitions."""
    endpoints = set()

    for partition_id in missing_partitions:
        partition_data = partitions_data.get(str(partition_id), {})

        for sample_data in partition_data.values():
            for url in sample_data.get("files", {}):
                match = endpoint_pattern.match(url)
                if match:
                    endpoints.add(match.group(0))

    return endpoints


def get_partition_error_endpoints(
    error_files,
    dataset: str,
    partition_id: str,
    verbose: bool = False,
) -> tuple[set[str], set[str]]:
    """
    Extract ROOT URLs and endpoints from error logs for one exact partition.

    A log is associated with the partition only if it contains a matching
    WorkItem dataset identifier, such as dataset_80 for partition 80.
    """
    target_dataset = f"{dataset}_{partition_id}"
    work_item_pattern = re.compile(
        rf"""WorkItem\s*\(\s*dataset\s*=\s*['"]{re.escape(target_dataset)}['"]""",
        flags=re.IGNORECASE,
    )
    endpoint_pattern = re.compile(ENDPOINT_PATTERN, flags=re.IGNORECASE)

    root_urls = set()
    endpoints = set()

    for error_file in error_files:
        error_file = Path(error_file)

        try:
            content = error_file.read_text(
                encoding="utf-8",
                errors="replace",
            )
        except OSError as exc:
            if verbose:
                print(
                    f"[{dataset}:{partition_id}] " f"Could not read {error_file}: {exc}"
                )
            continue

        if not work_item_pattern.search(content):
            continue

        urls = set(ROOT_URL_PATTERN.findall(content))
        root_urls.update(urls)

        for url in urls:
            match = endpoint_pattern.match(url)
            if match:
                endpoints.add(match.group(0))

    return root_urls, endpoints


def map_endpoints_to_sites(endpoints: set[str]) -> set[str]:
    """Map known XRootD endpoints to CMS site names."""
    return {
        xroot_to_site[endpoint] for endpoint in endpoints if endpoint in xroot_to_site
    }


def get_partition_sites(partitions_data: dict, partition_id: str) -> set[str]:
    """Identify sites from the files belonging to one partition."""
    endpoints = get_partition_endpoints(
        partitions_data,
        {str(partition_id)},
        re.compile(ENDPOINT_PATTERN, flags=re.IGNORECASE),
    )
    return map_endpoints_to_sites(endpoints)


def analyze_xrootd_error_logs(
    error_files,
    job_dir: Path,
    dataset: str,
    missing_partitions=None,
    verbose: bool = False,
) -> dict:
    """
    Identify error types and XRootD endpoints for one dataset.

    ROOT URLs are extracted from the logs first. Partition endpoints are used
    only when no ROOT URL is present in any of the supplied logs.
    """
    dataset_dir = Path(job_dir) / dataset
    partitions_path = dataset_dir / "partitions.json"

    missing_partitions = {str(partition) for partition in missing_partitions}

    if partitions_path.is_file():
        partitions_data = json.loads(partitions_path.read_text(encoding="utf-8"))
    else:
        partitions_data = {}

    endpoint_pattern = re.compile(ENDPOINT_PATTERN, flags=re.IGNORECASE)
    root_urls = set()
    endpoints = set()
    error_types = set()

    for error_file in error_files:
        error_file = Path(error_file)

        try:
            content = error_file.read_text(
                encoding="utf-8",
                errors="replace",
            )
        except OSError:
            continue

        urls = set(ROOT_URL_PATTERN.findall(content))
        root_urls.update(urls)
        error_types.update(identify_error_types(content))

        for url in urls:
            match = endpoint_pattern.match(url)
            if match:
                endpoints.add(match.group(0))

    # Use partition endpoints only when no ROOT URLs appear in the logs.
    if not root_urls and missing_partitions:
        endpoints = get_partition_endpoints(
            partitions_data,
            missing_partitions,
            endpoint_pattern,
        )

    return {
        "root_urls": root_urls,
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

    Returns expected job numbers, completed job numbers, and recent .err
    paths grouped by dataset.
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

        jobnum[dataset] = jobnum_path.read_text(encoding="utf-8").splitlines()

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


def print_job_status(jobnum, jobnum_done):
    """Print job completion totals and return missing jobs per dataset."""
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

    if n_missing:
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
    skipbadfiles: bool,
    workers: int,
    memory: str,
    cpus: int,
    executor: str = "iterative",
) -> None:
    """Update the executor and resources in a dataset's arguments.json."""
    args_json = job_dir / dataset / "arguments.json"

    if not args_json.is_file():
        print(f"[{dataset}] Missing arguments.json — cannot update executor")
        return

    arguments = json.loads(args_json.read_text(encoding="utf-8"))

    arguments["executor"] = executor
    arguments["skipbadfiles"] = skipbadfiles
    arguments["workers"] = workers
    arguments["memory"] = memory
    arguments["cpus"] = cpus
    
    args_json.write_text(
        json.dumps(arguments, indent=4) + "\n",
        encoding="utf-8",
    )

    if skipbadfiles:
        print(f"[{dataset}] Executing with '{executor}' executor, {workers} workers, {memory}MB, {cpus} CPUs, and skiping bad files")
    else:
        print(f"[{dataset}] Executing with '{executor}' executor, {workers} workers, {memory}MB, and {cpus} CPUs")


def update_partitions_to_global(job_dir: Path, datasets: list):
    """Rewrite missing partition files to use the CMS global redirector."""
    global_redirector = "root://cms-xrd-global.cern.ch//store/"

    for dataset in datasets:
        partition_file = job_dir / dataset / "partitions.json"
        if not partition_file.exists():
            print(f"[{dataset}] Missing partitions.json — skipping")
            continue

        partitions = json.loads(partition_file.read_text(encoding="utf-8"))
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

        partition_file.write_text(json.dumps(partitions, indent=4), encoding="utf-8")
        print(f"[{dataset}] Updated partitions to CMS global redirector")


def analyze_xrootd_errors_by_dataset(
    error_files,
    job_dir,
    dataset,
    missing_partitions,
    verbose=False,
):
    """Diagnose a dataset without replacing present but unmapped ROOT endpoints."""
    errors = analyze_xrootd_error_logs(
        error_files=error_files,
        job_dir=job_dir,
        dataset=dataset,
        missing_partitions=missing_partitions,
        verbose=verbose,
    )

    errors["sites"] = map_endpoints_to_sites(errors["endpoints"])
    return errors


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

    root_files = list(current_fileset["files"].items())
    nfiles = json.loads(args_json.read_text(encoding="utf-8"))["nfiles"]
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
    partition_file.write_text(
        json.dumps(partition_dataset, indent=4) + "\n",
        encoding="utf-8",
    )


def file_identity(url: str) -> str:
    """Return a replica-independent CMS logical file name."""
    path = urlsplit(url).path
    marker = "/store/"

    if marker not in path:
        raise ValueError(f"Cannot identify CMS LFN from URL: {url}")

    return path[path.index(marker) :]


class UnmappedXRootDError(RuntimeError):
    """A ROOT file in an error log has no known CMS site mapping."""


def require_mapped_root_sites(root_urls: set[str], context: str) -> set[str]:
    """Fail explicitly for every ROOT URL whose endpoint is not mapped."""
    endpoint_pattern = re.compile(ENDPOINT_PATTERN, flags=re.IGNORECASE)
    sites = set()
    unknown = []
    for url in sorted(root_urls):
        match = endpoint_pattern.match(url)
        endpoint = match.group(0) if match else None
        if endpoint not in xroot_to_site:
            unknown.append(f"[{context}] Could not find the site associated with ROOT file {url} (endpoint: {endpoint or 'unknown'}).")
        else:
            sites.add(xroot_to_site[endpoint])
    if unknown:
        raise UnmappedXRootDError("\n".join(unknown))
    return sites


def get_problematic_sites_for_partition(
    year: str,
    dataset: str,
    partition_id: str,
    partitions_data: dict,
    error_files,
    verbose: bool = False,
) -> set[str]:
    """Prefer partition-specific error URLs; fall back only if none exist."""
    root_urls, _ = get_partition_error_endpoints(
        error_files=error_files,
        dataset=dataset,
        partition_id=str(partition_id),
        verbose=verbose,
    )
    if root_urls:
        sites = require_mapped_root_sites(root_urls, f"{dataset}:{partition_id}")
        if verbose:
            print(f"[{dataset}:{partition_id}] Sites from error logs: {sorted(sites)}")
        return sites

    if verbose:
        print(f"[{dataset}:{partition_id}] No ROOT URLs in matching error logs; falling back to partition endpoints.")
    sites = get_partition_sites(partitions_data, str(partition_id))
    if verbose:
        print(f"[{dataset}:{partition_id}] Fallback sites: {sorted(sites)}")
    return sites


def endpoints_from_root_urls(root_urls):
    """Return the exact endpoints present in ROOT URLs."""
    pattern = re.compile(ENDPOINT_PATTERN, flags=re.IGNORECASE)
    return {match.group(0) for url in root_urls if (match := pattern.match(url))}


def partition_endpoints(partitions, partition_id):
    """Return the endpoints of one partition in partitions.json."""
    return get_partition_endpoints(
        partitions, {str(partition_id)},
        re.compile(ENDPOINT_PATTERN, flags=re.IGNORECASE),
    )


def select_missing_job_error_files(error_files, dataset, missing_ids, verbose=False):
    """Select recent .err logs that can be attributed to a missing job.

    Accept an exact job ID in the filename (e.g. dataset_7.err, 7.err,
    dataset_7.123.err), or a WorkItem identifier in the log body.
    Logs containing WorkItems for both missing and completed jobs are
    excluded unless their filename identifies a missing job unambiguously.
    """
    missing_ids = {str(x) for x in missing_ids}
    selected = []
    for path in error_files:
        path = Path(path)
        stem = path.stem
        filename_ids = set()
        for pid in missing_ids:
            if re.search(rf"(?:^|[_\.-]){re.escape(pid)}(?:$|[_\.-])", stem):
                filename_ids.add(pid)
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            if verbose:
                print(f"[{dataset}] Could not read {path}: {exc}")
            continue
        workitems = set(re.findall(
            rf"WorkItem\s*\(\s*dataset\s*=\s*['\"]{re.escape(dataset)}_(\d+)['\"]",
            content, flags=re.IGNORECASE,
        ))
        if filename_ids and (not workitems or workitems <= missing_ids):
            selected.append(path)
        elif workitems and workitems <= missing_ids:
            selected.append(path)
    return selected


def plan_regeneration(job_dir, dataset_partitions, dataset_error_files,
                      by_dataset=False, verbose=False):
    """Preflight mappings and choose era, dataset, or partition fallback.

    Only logs attributable to missing jobs are supplied by the caller.
    Default: union of ROOT URLs from all those logs, applied to every
    eligible dataset with the same era-wide blacklist. Dataset mode:
    each dataset gets its own log-derived blacklist, or partition fallback.
    If no ROOT URLs exist in the era, use partition fallback everywhere.
    """
    data = {}
    era_urls = set()
    for dataset, missing_ids in sorted(dataset_partitions.items()):
        path = job_dir / dataset / "partitions.json"
        if not path.is_file():
            print(f"[{dataset}] Missing partitions.json; skipping")
            continue
        partitions = json.loads(path.read_text(encoding="utf-8"))
        urls = set()
        for log in dataset_error_files.get(dataset, []):
            try:
                urls.update(ROOT_URL_PATTERN.findall(
                    Path(log).read_text(encoding="utf-8", errors="replace")))
            except OSError:
                continue
        data[dataset] = (partitions, set(missing_ids), urls)
        era_urls.update(urls)

    # Validate ALL observed ROOT URLs before any mutation, even in by_dataset mode.
    era_sites = require_mapped_root_sites(era_urls, "era")
    plans = {}
    for dataset, (partitions, missing_ids, urls) in data.items():
        if not by_dataset and era_urls:
            plans[dataset] = {
                "mode": "era", "sites": set(era_sites),
                "endpoints": endpoints_from_root_urls(era_urls),
                "source": "ROOT URLs in missing-job error logs (era-wide)",
                "partition_ids": missing_ids,
            }
        elif by_dataset and urls:
            plans[dataset] = {
                "mode": "dataset", "sites": require_mapped_root_sites(urls, dataset),
                "endpoints": endpoints_from_root_urls(urls),
                "source": "ROOT URLs in missing-job error logs (dataset)",
                "partition_ids": missing_ids,
            }
        else:
            sites_by_partition = {}
            endpoints_by_partition = {}
            for pid in sort_job_numbers(missing_ids):
                pid = str(pid)
                endpoints = partition_endpoints(partitions, pid)
                endpoints_by_partition[pid] = endpoints
                sites_by_partition[pid] = map_endpoints_to_sites(endpoints)
            plans[dataset] = {
                "mode": "partition", "sites_by_partition": sites_by_partition,
                "endpoints_by_partition": endpoints_by_partition,
                "sources_by_partition": {
                    str(pid): "Partition endpoints (partitions.json)"
                    for pid in missing_ids
                },
                "partition_ids": missing_ids,
            }
    return plans


def print_regeneration_strategy(regeneration_plan, by_dataset, verbose=False):
    """Explain the fileset update strategy; list datasets only for dataset verbose mode."""
    log_datasets = sorted(
        dataset for dataset, plan in regeneration_plan.items()
        if plan["mode"] in {"era", "dataset"}
    )
    partition_datasets = sorted(
        dataset for dataset, plan in regeneration_plan.items()
        if plan["mode"] == "partition"
    )
    count = len(regeneration_plan)
    noun = "dataset" if count == 1 else "datasets"

    print("\n" + "-" * 60)
    print("FILESET UPDATE STRATEGY")
    print("-" * 60)

    if not count:
        print("No eligible datasets were found for fileset regeneration.")
        return

    if not by_dataset:
        if log_datasets:
            print(f"Regenerating {count} {noun} using a common site blacklist")
            print("identified from ROOT URLs in the missing-job error logs")
            print("across the era.")
        else:
            print(f"Regenerating {count} {noun} partition by partition because")
            print("no ROOT URLs were found in the missing-job error logs")
            print("across the era. Sites will be taken from partitions.json.")
        return

    print(f"Regenerating {count} {noun} independently, using a separate")
    print("site blacklist for each dataset or partition.")

    if log_datasets:
        n = len(log_datasets)
        label = "dataset" if n == 1 else "datasets"
        print(f"\n{n} {label} will use sites identified from xrootd endpoints in")
        print("their missing-job error logs.")
        if verbose:
            for dataset in log_datasets:
                print(f"  - {dataset}")

    if partition_datasets:
        n = len(partition_datasets)
        label = "dataset" if n == 1 else "datasets"
        print(f"\n{n} {label} will use sites identified from partitions.json,")
        print("with regeneration performed partition by partition.")
        if verbose:
            for dataset in partition_datasets:
                print(f"  - {dataset}")


def regenerate_non_era_partitions(
    year: str,
    fileset_dir: Path,
    job_dir: Path,
    dataset_partitions: dict[str, set[str]],
    dataset_error_files: dict[str, list[Path]],
    discovery_tool: str = "rucio",
    verbose: bool = False,
    regeneration_plan=None,
) -> None:
    """
    Refetch each missing partition with its own blacklist and preserve job IDs.

    Before each fetch.py invocation, the current blacklist is cleared using
    modify_site_list(), and only the sites identified for that partition are
    blacklisted. The regenerated files are matched by LFN so unrelated
    partitions and their job identifiers remain unchanged.
    """
    fileset_path = fileset_dir / f"fileset_{year}_NANO_lxplus.json"
    original_fileset = fileset_path.read_bytes() if fileset_path.exists() else None
    updated = 0
    era_plans = [p for p in (regeneration_plan or {}).values() if p["mode"] == "era"]
    if era_plans:
        print(f"[{year}] Blacklisting: {', '.join(sorted(era_plans[0]['sites']))}")

    try:
        for dataset, partition_ids in sorted(dataset_partitions.items()):
            plan = regeneration_plan.get(dataset) if regeneration_plan is not None else None
            if plan is None and regeneration_plan is not None:
                continue
            if plan is not None:
                partition_ids = plan["partition_ids"]
            partition_path = job_dir / dataset / "partitions.json"

            if not partition_path.is_file():
                print(f"[{dataset}] Missing partitions.json; skipping")
                continue

            partitions = json.loads(partition_path.read_text(encoding="utf-8"))
            error_files = dataset_error_files.get(dataset, [])

            cached_fetched = None
            dataset_mode = plan is not None and plan["mode"] in {"dataset", "era"}
            dataset_updated = 0
            if dataset_mode:
                if plan["mode"] != "era":
                    print(f"[{dataset}] Blacklisting: {', '.join(sorted(plan['sites']))}")
                reset_site_blacklist(year)
                for site in sorted(plan["sites"]):
                    modify_site_list(year, site, "black")
            for partition_id in sort_job_numbers(partition_ids):
                partition_id = str(partition_id)
                entry = partitions.get(partition_id)

                if not entry or len(entry) != 1:
                    print(
                        f"[{dataset}:{partition_id}] "
                        "Invalid/missing partition; skipping"
                    )
                    continue

                _, sample_data = next(iter(entry.items()))
                old_files = sample_data.get("files", {})

                if not old_files:
                    print(f"[{dataset}:{partition_id}] No files; skipping")
                    continue

                try:
                    old_by_lfn = {
                        file_identity(url): tree for url, tree in old_files.items()
                    }
                except ValueError as exc:
                    print(f"[{dataset}:{partition_id}] {exc}; skipping")
                    continue

                if plan is not None:
                    sites = (plan["sites"] if plan["mode"] in {"dataset", "era"}
                             else plan["sites_by_partition"].get(partition_id, set()))
                else:
                    sites = get_problematic_sites_for_partition(
                        year=year, dataset=dataset, partition_id=partition_id,
                        partitions_data=partitions, error_files=error_files,
                        verbose=verbose,
                    )

                if not sites:
                    print(
                        f"[{dataset}:{partition_id}] "
                        "No problematic CMS sites identified; skipping"
                    )
                    continue

                if not dataset_mode:
                    print(
                        f"[{dataset}:{partition_id}] Blacklisting: "
                        f"{', '.join(sorted(sites))}"
                    )
                    reset_site_blacklist(year)
                    for site in sorted(sites):
                        modify_site_list(year, site, "black")

                if cached_fetched is not None and plan is not None and plan["mode"] in {"dataset", "era"}:
                    fetched = cached_fetched
                else:
                    if fileset_path.exists():
                        fileset_path.unlink()

                    command = ["python3", "fetch.py", "--year", year, "--samples", dataset, "--skip_site"]
                    if discovery_tool == "rucio":
                        command.append("--rucio")
                    else:
                        command.append("--das")

                    if verbose:
                        label = dataset if dataset_mode else f"{dataset}:{partition_id}"
                        print(f"[{label}] Running: {' '.join(command)}")
                    result = subprocess.run(command, check=False)
                    if result.returncode != 0 or not fileset_path.is_file():
                        print(f"[{dataset}:{partition_id}] fetch.py failed; partition unchanged")
                        if plan is not None and plan["mode"] in {"dataset", "era"}:
                            break
                        continue
                    fetched = json.loads(fileset_path.read_text(encoding="utf-8"))
                    if plan is not None and plan["mode"] in {"dataset", "era"}:
                        cached_fetched = fetched

                candidates = [
                    value
                    for value in fetched.values()
                    if value.get("metadata", {}).get("short_name") == dataset
                ]

                if len(candidates) != 1:
                    print(
                        f"[{dataset}:{partition_id}] "
                        "Sample not uniquely found; unchanged"
                    )
                    continue

                try:
                    new_by_lfn = {
                        file_identity(url): url
                        for url in candidates[0].get("files", {})
                    }
                except ValueError as exc:
                    print(f"[{dataset}:{partition_id}] {exc}; unchanged")
                    continue

                missing_lfns = set(old_by_lfn) - set(new_by_lfn)

                if missing_lfns:
                    print(
                        f"[{dataset}:{partition_id}] "
                        f"{len(missing_lfns)} original files not found "
                        "with this blacklist; partition unchanged"
                    )
                    continue

                replacement_files = {
                    new_by_lfn[lfn]: tree for lfn, tree in old_by_lfn.items()
                }
                sample_data["files"] = replacement_files

                partition_path.write_text(
                    json.dumps(partitions, indent=4) + "\n",
                    encoding="utf-8",
                )
                updated += 1
                if dataset_mode:
                    dataset_updated += 1
                elif verbose:
                    print(
                        f"[{dataset}:{partition_id}] "
                        f"Updated {len(replacement_files)} files"
                    )

            if dataset_mode:
                if dataset_updated == len(partition_ids):
                    print(f"[{dataset}] Partitions updated")
                else:
                    print(
                        f"[{dataset}] Partitions partially updated: "
                        f"{dataset_updated}/{len(partition_ids)}"
                    )

    finally:
        if original_fileset is not None:
            fileset_path.write_bytes(original_fileset)
        elif fileset_path.exists():
            fileset_path.unlink()

    print(f"Partition regeneration completed: {updated} partition(s) updated.")


def regenerate_era_batch(year, fileset_dir, job_dir, era_plans, discovery_tool="rucio", verbose=False):
    """Fetch all affected samples once using the shared era blacklist.

    Only missing partitions are rewritten; their job IDs and LFNs are preserved.
    """
    datasets = sorted(era_plans)
    if not datasets:
        return 0
    sites = set(next(iter(era_plans.values()))["sites"])
    if any(set(plan["sites"]) != sites for plan in era_plans.values()):
        raise ValueError("Era-wide plans have inconsistent site blacklists")

    fileset_path = fileset_dir / f"fileset_{year}_NANO_lxplus.json"
    original_fileset = fileset_path.read_bytes() if fileset_path.exists() else None
    updated = 0
    try:
        print(f"[{year}] Blacklisting: {', '.join(sorted(sites))}")
        reset_site_blacklist(year)
        for site in sorted(sites):
            modify_site_list(year, site, "black")

        if fileset_path.exists():
            fileset_path.unlink()
        
        command = ["python3", "fetch.py", "--year", year, "--samples", *datasets, "--skip_site"]
        if discovery_tool == "rucio":
            command.append("--rucio")
        else:
            command.append("--das")

        print(f"[{year}] Running fetch.py once for {len(datasets)} datasets")
        if verbose:
            print("  Command: " + " ".join(command))
        result = subprocess.run(command, check=False)
        if result.returncode != 0 or not fileset_path.is_file():
            print(f"[{year}] fetch.py failed; no partitions updated")
            return 0
        try:
            fetched = json.loads(fileset_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[{year}] Invalid fetched fileset: {exc}; no partitions updated")
            return 0

        for dataset in datasets:
            partition_path = job_dir / dataset / "partitions.json"
            if not partition_path.is_file():
                print(f"[{dataset}] Missing partitions.json; skipping")
                continue
            partitions = json.loads(partition_path.read_text(encoding="utf-8"))
            candidates = [v for v in fetched.values()
                          if v.get("metadata", {}).get("short_name") == dataset]
            if len(candidates) != 1:
                print(f"[{dataset}] Sample not uniquely found; partitions unchanged")
                continue
            try:
                new_by_lfn = {file_identity(url): url
                              for url in candidates[0].get("files", {})}
            except ValueError as exc:
                print(f"[{dataset}] {exc}; partitions unchanged")
                continue
            dataset_updated = 0
            missing_ids = era_plans[dataset]["partition_ids"]
            for partition_id in sort_job_numbers(missing_ids):
                partition_id = str(partition_id)
                entry = partitions.get(partition_id)
                if not entry or len(entry) != 1:
                    print(f"[{dataset}:{partition_id}] Invalid/missing partition; skipping")
                    continue
                sample_data = next(iter(entry.values()))
                old_files = sample_data.get("files", {})
                if not old_files:
                    print(f"[{dataset}:{partition_id}] No files; skipping")
                    continue
                try:
                    old_by_lfn = {file_identity(url): tree
                                  for url, tree in old_files.items()}
                except ValueError as exc:
                    print(f"[{dataset}:{partition_id}] {exc}; skipping")
                    continue
                missing_lfns = set(old_by_lfn) - set(new_by_lfn)
                if missing_lfns:
                    print(f"[{dataset}:{partition_id}] {len(missing_lfns)} original files "
                          "not found with this blacklist; partition unchanged")
                    continue
                sample_data["files"] = {new_by_lfn[lfn]: tree
                                        for lfn, tree in old_by_lfn.items()}
                dataset_updated += 1
            if dataset_updated:
                partition_path.write_text(
                    json.dumps(partitions, indent=4) + "\n", encoding="utf-8")
            updated += dataset_updated
            if dataset_updated == len(missing_ids):
                print(f"[{dataset}] Partitions updated")
            else:
                print(f"[{dataset}] Partitions partially updated: "
                      f"{dataset_updated}/{len(missing_ids)}")
        return updated
    finally:
        if original_fileset is not None:
            fileset_path.write_bytes(original_fileset)
        elif fileset_path.exists():
            fileset_path.unlink()


def regenerate_missing_partitions(
    year: str,
    fileset_dir: Path,
    job_dir: Path,
    dataset_partitions: dict[str, set[str]],
    dataset_error_files: dict[str, list[Path]],
    discovery_tool: str = "rucio",
    verbose: bool = False,
    regeneration_plan=None,
) -> None:
    """Batch era-wide fetches; preserve per-dataset/partition fallback behavior."""
    plans = regeneration_plan or {}
    era_plans = {name: plan for name, plan in plans.items()
                 if plan["mode"] == "era"}
    other_plans = {name: plan for name, plan in plans.items()
                   if plan["mode"] != "era"}
    total = 0
    if era_plans:
        total += regenerate_era_batch(year, fileset_dir, job_dir, era_plans, discovery_tool, verbose)
    if other_plans:
        regenerate_non_era_partitions(
            year, fileset_dir, job_dir,
            {name: dataset_partitions[name] for name in other_plans},
            {name: dataset_error_files.get(name, []) for name in other_plans},
            discovery_tool=discovery_tool,
            verbose=verbose, regeneration_plan=other_plans,
        )
    elif era_plans:
        print(f"Partition regeneration completed: {total} partition(s) updated.")


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
    memory: str,
    cpus: int,
):
    """
    Prepare and resubmit only jobs without output files.

    The original submission file is backed up once as *_all.sub, and resources are dynamically updated.
    """
    print("-" * 60)
    print("RESUBMITTING MISSING JOBS")
    print("-" * 60)

    submit_files = []

    for dataset in datasets_with_missing_jobs:
        missing_jobs = sort_job_numbers(jobnum_missing[dataset])
        dataset_dir = job_dir / dataset

        missing_file = dataset_dir / "missing.txt"
        missing_file.write_text(
            "".join(f"{job}\n" for job in missing_jobs),
            encoding="utf-8",
        )

        condor_file = dataset_dir / f"{workflow}_{dataset}.sub"
        condor_backup = condor_file.with_name(f"{condor_file.stem}_all.sub")

        if not condor_backup.exists():
            subprocess.run(
                ["cp", str(condor_file), str(condor_backup)],
                check=True,
            )

        submit_text = condor_file.read_text(encoding="utf-8").replace(
            "jobnum.txt",
            "missing.txt",
        )
        
        # Inject dynamic requested memory and cpus 
        submit_text = re.sub(
            r"(request_memory\s*=\s*)\d+", f"\\g<1>{str(memory)}", submit_text
        )
        submit_text = re.sub(
            r"(request_cpus\s*=\s*)\d+", f"\\g<1>{str(cpus)}", submit_text
        )

        condor_file.write_text(submit_text, encoding="utf-8")
        submit_files.append(condor_file)

    for submit_file in submit_files:
        print(f"Submitting {submit_file}")
        subprocess.run(
            ["condor_submit", str(submit_file)],
            check=True,
        )

    print("Job resubmission completed.\n")


def reset_workflow(args, job_dir, log_dir, fileset_dir):
    """Remove existing workflow outputs and rerun runner.py."""
    print("-" * 60)
    print("RESET REQUESTED")
    print("-" * 60)

    subprocess.run(["rm", "-rf", str(job_dir)], check=False)
    subprocess.run(["rm", "-rf", str(log_dir)], check=False)
    
    if getattr(args, "deep_reset", False):
        subprocess.run(["rm", "-rf", str(fileset_dir / f"{args.year}_sites.yaml")], check=False)
        subprocess.run(["rm", "-rf", str(fileset_dir / f"fileset_{args.year}_NANO_lxplus.json")], check=False)

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
        str(args.memory),
        "-c", 
        str(args.cpus),
        "--workers", 
        str(args.workers),
        "--executor",
        args.executor,
    ]
    if getattr(args, "eos", False):
        reset_cmd.append("--eos")
    if getattr(args, "label", ""):
        reset_cmd.extend(["-l", args.label])
    if getattr(args, "skipbadfiles", False):
        reset_cmd.append("--skipbadfiles")
    if getattr(args, "use_global", False):
        reset_cmd.append("--global")
    if getattr(args, "discovery", "rucio") == "rucio":
        reset_cmd.append("--rucio")
    else:
        reset_cmd.append("--das")

    print("Rerunning the workflow via runner.py...")
    subprocess.run(reset_cmd, check=True)
    print("Reset complete.\n")


def main():
    args = parse_args()

    base_dir = Path.cwd()
    output_dir = make_output_directory(args)

    condor_dir = base_dir / "condor"
    workflow_dir = f"{args.workflow}/{args.label}" if args.label else args.workflow
    job_dir = condor_dir / workflow_dir / args.year
    log_dir = condor_dir / "logs" / args.workflow / args.year
    fileset_dir = base_dir / "analysis" / "filesets"

    if not job_dir.is_dir() and not args.reset and not args.deep_reset:
        raise FileNotFoundError(f"Condor job directory does not exist: {job_dir}")

    if args.reset or args.deep_reset:
        reset_workflow(
            args,
            job_dir,
            log_dir,
            fileset_dir,
        )

    jobnum, jobnum_done, error_files = get_jobs_info(
        job_dir,
        output_dir,
        log_dir,
        args.output_format,
        args.hours_ago,
    )

    jobnum_missing, datasets_with_missing_jobs = print_job_status(
        jobnum,
        jobnum_done,
    )

    if not datasets_with_missing_jobs:
        print("All expected jobs have completed successfully.")
        return

    missing_datasets = sorted(datasets_with_missing_jobs)

    if args.reset or args.deep_reset:
        print(
            "Workflow reset completed. " "Skipping XRootD diagnostics and blacklisting."
        )

        if timed_input("Resubmit missing jobs now?") in {"y", "yes"}:
            resubmit_jobs(
                job_dir,
                jobnum_missing,
                missing_datasets,
                args.workflow,
                args.year,
                args.memory,
                args.cpus,
            )
        return
        
    if args.update_executor:
        for dataset in missing_datasets:
            update_executor_for_dataset(
                job_dir,
                dataset,
                skipbadfiles=args.skipbadfiles,
                workers=args.workers,
                memory=args.memory,
                cpus=args.cpus,
                executor=args.executor,
            )

    # Filter logs BEFORE both diagnostics and planning. A recent log from a
    # completed job must never contribute sites to the blacklist.
    eligible_missing = {
        dataset: get_missing_job_partitions(jobnum_missing, dataset)
        for dataset in missing_datasets
    }
    missing_error_files = {
        dataset: select_missing_job_error_files(
            error_files.get(dataset, []), dataset, jobnum_missing[dataset],
            verbose=args.verbose,
        )
        for dataset in missing_datasets
    }

    if args.use_global:
        update_partitions_to_global(job_dir, missing_datasets)
    else:
        errors_by_dataset = {}

        for dataset in missing_datasets:
            err_files = missing_error_files.get(dataset, [])
            missing_partitions = get_missing_job_partitions(jobnum_missing, dataset)
            if not err_files:
                errors_by_dataset[dataset] = {
                    "root_urls": set(),
                    "endpoints": set(),
                    "error_types": set(),
                    "partitions": missing_partitions,
                    "sites": set(),
                }
                continue

            errors_by_dataset[dataset] = analyze_xrootd_errors_by_dataset(
                error_files=err_files,
                job_dir=job_dir,
                dataset=dataset,
                missing_partitions=missing_partitions,
                verbose=args.verbose,
            )

        # Every missing partition is eligible. If no relevant ROOT URL is found,
        # the plan falls back to partitions.json (even if no .err is available).
        dataset_partitions = {
            dataset: eligible_missing[dataset]
            for dataset in missing_datasets
        }
        dataset_partitions = {
            dataset: partitions
            for dataset, partitions in dataset_partitions.items()
            if partitions
        }

        regeneration_plan = {}
        if dataset_partitions:
            try:
                regeneration_plan = plan_regeneration(
                    job_dir, dataset_partitions,
                    {dataset: missing_error_files.get(dataset, []) for dataset in dataset_partitions},
                    by_dataset=args.by_dataset, verbose=args.verbose,
                )
            except UnmappedXRootDError as exc:
                print(f"\nERROR: {exc}", file=sys.stderr)
                print("Add the missing endpoint-to-site mapping in xroot_to_site and rerun the script. No regeneration or resubmission was performed.", file=sys.stderr)
                raise SystemExit(1)


        if args.verbose:
            print("\n" + "=" * 60)
            print("DETAILED XROOTD ERROR ANALYSIS")
            print("=" * 60)
            for dataset in missing_datasets:
                errors = errors_by_dataset[dataset]
                plan = regeneration_plan.get(dataset)
                print(f"\nDataset: {dataset}")
                print("  Error types: " + (
                    ", ".join(sorted(errors.get("error_types", set())))
                    or "unrecognized or no logs"
                ))
                if plan is not None and plan["mode"] in {"dataset", "era"}:
                    displayed_endpoints = plan["endpoints"]
                    displayed_sites = plan["sites"]
                    source = plan["source"]
                elif plan is not None:
                    displayed_endpoints = set().union(*plan["endpoints_by_partition"].values()) if plan["endpoints_by_partition"] else set()
                    displayed_sites = set().union(*plan["sites_by_partition"].values()) if plan["sites_by_partition"] else set()
                    sources = set(plan["sources_by_partition"].values())
                    source = next(iter(sources)) if len(sources) == 1 else "Mixed (error logs and partitions.json)"
                else:
                    displayed_endpoints = errors.get("endpoints", set())
                    displayed_sites = errors.get("sites", set())
                    source = ("ROOT URLs in error logs" if errors.get("root_urls")
                              else "Partition endpoints (partitions.json)" if displayed_endpoints
                              else "No endpoints identified")
                print("  Source: " + source)
                print("  Endpoints: " + (
                    ", ".join(sorted(displayed_endpoints)) or "none identified"
                ))
                print("  Sites: " + (
                    ", ".join(sorted(displayed_sites)) or "none identified"
                ))
                print("  Missing jobs: " + (
                    ", ".join(sort_job_numbers(jobnum_missing[dataset])) or "none"
                ))


        print_regeneration_strategy(regeneration_plan, args.by_dataset, args.verbose)

        if not dataset_partitions:
            print("\nNo eligible partitions for fileset regeneration.")
        elif timed_input("\nRegenerate input files for missing partitions?") in {
            "y",
            "yes",
        }:
            regenerate_missing_partitions(
                year=args.year,
                fileset_dir=fileset_dir,
                job_dir=job_dir,
                dataset_partitions=dataset_partitions,
                dataset_error_files={
                    dataset: missing_error_files.get(dataset, []) for dataset in dataset_partitions
                },
                discovery_tool=args.discovery,
                verbose=args.verbose,
                regeneration_plan=regeneration_plan,
            )

    if timed_input("Resubmit missing jobs now?") in {"y", "yes"}:
        resubmit_jobs(
            job_dir,
            jobnum_missing,
            missing_datasets,
            args.workflow,
            args.year,
            args.memory,
            args.cpus,
        )


if __name__ == "__main__":
    main()