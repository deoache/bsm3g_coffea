import re
import glob
import yaml
import json
import subprocess
import numpy as np
from pathlib import Path
from collections import defaultdict
from analysis.workflows.config import WorkflowConfigBuilder


def get_rootfiles(year: str, dataset: str) -> list[str]:
    """Return ROOT files found in the dataset directory or its subdirectories."""
    fileset_path = Path.cwd() / "analysis" / "filesets"

    with (fileset_path / f"{year}_fileset.yaml").open() as f:
        dataset_config = yaml.safe_load(f)[dataset]

    dataset_path = Path(dataset_config["path"])
    root_files = sorted(dataset_path.glob("*.root"))

    if not root_files:
        root_files = sorted(dataset_path.glob("*/*.root"))

    if not root_files:
        raise FileNotFoundError(
            f"No .root files found in {dataset_path} "
            "or its immediate subdirectories."
        )

    return [str(path) for path in root_files]


def divide_list(lst: list, nfiles: int = 20) -> list:
    """Divide a list into sublists such that each sublist has at least 20 elements."""
    if len(lst) < nfiles:
        return [lst]
    # Dynamically calculate the number of sublists such that each has at least 20 elements
    n = len(lst) // nfiles  # This gives the number of groups with at least 20 elements
    if len(lst) % nfiles != 0:
        n += 1  # Increase n by 1 if there is a remainder, to accommodate extra elements
    # Divide the list into 'n' sublists
    size = len(lst) // n
    remainder = len(lst) % n
    result = []
    start = 0
    for i in range(n):
        if i < remainder:
            end = start + size + 1
        else:
            end = start + size
        result.append(lst[start:end])
        start = end
    return result


def get_dataset_config(year):
    aux_year_map = {
        "2016": "2016preVFP",
        "2022": "2022preEE",
        "2023": "2023preBPix",
    }
    aux_year = aux_year_map.get(year, year)
    fileset_path = Path(f"{Path.cwd()}/analysis/filesets")
    run_key = "Run3" if year.startswith("202") else "Run2"
    nano_version = (
        "nanov15" if year == "2024" else ("nanov9" if run_key == "Run2" else "nanov12")
    )
    with open(f"{fileset_path}/{aux_year}_{nano_version}.yaml", "r") as f:
        dataset_config = yaml.safe_load(f)
    return dataset_config


def get_dataset_key(dataset, year):
    dataset_config = get_dataset_config(year)
    for dataset_key in dataset_config:
        if dataset.startswith(dataset_key):
            if dataset_config[dataset_key]["process"] == "Data":
                return dataset_config[dataset_key]["key"]
            else:
                return "MC"


def get_dataset_era(dataset, year):
    dataset_config = get_dataset_config(year)
    for dataset_key in dataset_config:
        if dataset.startswith(dataset_key):
            return dataset_config[dataset_key]["era"]


def modify_site_list(year: str, site: str, status: str) -> None:
    """
    Move a given site to the specified status list ("white" or "black")

    Parameters:
    -----------
        site: The site identifier to modify.
        status: Desired list to move the site to ("white" or "black").
    """
    yaml_file = Path.cwd() / "analysis" / "filesets" / f"{year}_sites.yaml"
    with open(yaml_file, "r") as f:
        data = yaml.safe_load(f)

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

    with open(yaml_file, "w") as f:
        yaml.dump(data, f, default_flow_style=False)


def get_dataset_partition(content, partitions):
    """
    Retrieve the partition entry corresponding to the dataset in a log.

    Parameters
    ----------
    content : str
        Contents of a Condor error log.
    partitions : dict
        Partition information loaded from partitions.json.

    Returns
    -------
    dict or None
        Dataset partition entry, or None if the dataset cannot be found.
    """
    dataset_match = re.search(
        r"""--dataset(?:=|\s+)['"]?([^\s'"]+)""",
        content,
    )

    if not dataset_match:
        return None

    dataset = dataset_match.group(1)

    partition = next(
        (
            partition_data[dataset]
            for partition_data in partitions.values()
            if dataset in partition_data
        ),
        None,
    )

    if partition is None:
        print(f"[{dataset}] Dataset not found in partitions.json")

    return partition


def extract_file_error_endpoints(content, partitions):
    """
    Identify XRootD endpoints associated with ROOT files mentioned in errors.

    ROOT filenames are matched against the files in the dataset partition.
    Matching is based on the basename, allowing the same file to be
    identified even when the error and partition use different storage paths.

    Parameters
    ----------
    content : str
        Contents of a Condor error log.
    partitions : dict
        Partition information loaded from partitions.json.

    Returns
    -------
    set[str]
        Endpoints associated with the ROOT files mentioned in the errors.
    """
    endpoints = set()
    partition = get_dataset_partition(content, partitions)

    if partition is None:
        return endpoints

    # Extract ROOT filenames appearing in the error log.
    root_file_pattern = r"""[^\s"'<>;]+\.root\b"""

    failing_files = re.findall(
        root_file_pattern,
        content,
        flags=re.IGNORECASE,
    )

    for failing_file in failing_files:
        filename = failing_file.rsplit("/", 1)[-1]

        matching_urls = [
            url for url in partition["files"] if url.rsplit("/", 1)[-1] == filename
        ]

        if not matching_urls:
            continue

        for url in matching_urls:
            match = re.match(ENDPOINT_PATTERN, url, flags=re.IGNORECASE)

            if match:
                endpoints.add(match.group())

    return endpoints


def get_partition_endpoints(content, partitions):
    """
    Return all unique XRootD endpoints in the dataset partition.

    This is the fallback when an error is detected but its specific
    failing site cannot be identified from the log.
    """
    endpoints = set()
    partition = get_dataset_partition(content, partitions)

    if partition is None:
        return endpoints

    for url in partition["files"]:
        match = re.match(ENDPOINT_PATTERN, url, flags=re.IGNORECASE)

        if match:
            endpoints.add(match.group())

    return endpoints


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


def identify_error_types(content):
    """Identify recognized error types in a log's contents."""
    error_types = set()

    for error_type, patterns in {
        **EXECUTOR_ERROR_PATTERNS,
        **FILE_ERROR_PATTERNS,
    }.items():
        if any(
            re.search(pattern, content, flags=re.IGNORECASE) for pattern in patterns
        ):
            error_types.add(error_type)

    return error_types


def get_dataset_partition_entries(partitions, dataset):
    """
    Return partition entries belonging to the specified dataset.

    Supports sample names such as 'dataset' and 'dataset_1', 'dataset_2'.
    """
    entries = []

    sample_pattern = re.compile(rf"{re.escape(dataset)}_\d+")

    for partition_id, partition_data in partitions.items():
        for sample_name, sample_data in partition_data.items():
            if sample_name == dataset or sample_pattern.fullmatch(sample_name):
                entries.append((str(partition_id), sample_name, sample_data))

    return entries


def extract_root_filenames(content):
    """Extract ROOT filenames mentioned in a log."""
    paths = re.findall(
        r"""[^\s"'<>;]+\.root\b""",
        content,
        flags=re.IGNORECASE,
    )

    return {path.rstrip(".,:)").rsplit("/", 1)[-1] for path in paths}


def extract_file_error_endpoints(content, partitions, dataset):
    """
    Match ROOT filenames in a log to files in dataset partitions.

    Returns
    -------
    tuple[set[str], set[str]]
        Identified XRootD endpoints and partition IDs.
    """
    endpoints = set()
    partition_ids = set()
    filenames = extract_root_filenames(content)

    if not filenames:
        return endpoints, partition_ids

    for partition_id, _, sample_data in get_dataset_partition_entries(
        partitions, dataset
    ):
        for url in sample_data.get("files", {}):
            filename = url.rsplit("/", 1)[-1]

            if filename not in filenames:
                continue

            match = re.match(
                ENDPOINT_PATTERN,
                url,
                flags=re.IGNORECASE,
            )

            if match:
                endpoints.add(match.group())
                partition_ids.add(partition_id)

    return endpoints, partition_ids


def get_failed_job_partitions(content, partitions, dataset):
    """
    Identify partition IDs associated with dataset arguments in a log.

    Supports sample names such as 'dataset', 'dataset_1', and 'dataset_2'.
    If the base dataset name is supplied, all matching partition IDs are
    returned.
    """
    sample_partitions = defaultdict(set)

    for partition_id, partition_data in partitions.items():
        for sample_name in partition_data:
            sample_partitions[sample_name].add(str(partition_id))

    matches = re.findall(
        r"""--dataset(?:=|\s+)['"]?([^'"\s]+)""",
        content,
    )

    partition_ids = set()

    for sample_name in matches:
        if sample_name in sample_partitions:
            partition_ids.update(sample_partitions[sample_name])
        elif sample_name == dataset:
            for sample_name_in_partitions, ids in sample_partitions.items():
                if sample_name_in_partitions == dataset or re.fullmatch(
                    rf"{re.escape(dataset)}_\d+",
                    sample_name_in_partitions,
                ):
                    partition_ids.update(ids)

    return partition_ids


def get_partition_endpoints(partitions, dataset):
    """
    Collect all XRootD endpoints and partition IDs for a dataset.
    """
    endpoints = set()
    partition_ids = set()

    for partition_id, _, sample_data in get_dataset_partition_entries(
        partitions, dataset
    ):
        for url in sample_data.get("files", {}):
            match = re.match(
                ENDPOINT_PATTERN,
                url,
                flags=re.IGNORECASE,
            )

            if match:
                endpoints.add(match.group())
                partition_ids.add(partition_id)

    return endpoints, partition_ids


def find_endpoint_partitions(endpoints, partitions, dataset):
    """Find partitions containing any of the specified endpoints."""
    partition_ids = set()

    for partition_id, _, sample_data in get_dataset_partition_entries(
        partitions, dataset
    ):
        for url in sample_data.get("files", {}):
            match = re.match(
                ENDPOINT_PATTERN,
                url,
                flags=re.IGNORECASE,
            )

            if match and match.group().lower() in {
                endpoint.lower() for endpoint in endpoints
            }:
                partition_ids.add(partition_id)

    return partition_ids


def get_failed_job_partitions(content, partitions, dataset):
    """
    Identify partition IDs associated with a failed job from its log.

    The job command is expected to contain the sample name passed through
    the --dataset argument.
    """
    sample_names = {
        sample_name: str(partition_id)
        for partition_id, partition_data in partitions.items()
        for sample_name in partition_data
    }

    matches = re.findall(
        r"--dataset(?:=|\s+)([^\s]+)",
        content,
    )

    partition_ids = set()

    for sample_name in matches:
        if sample_name in sample_names:
            partition_ids.add(sample_names[sample_name])
        elif sample_name == dataset:
            partition_ids.update(
                str(partition_id)
                for partition_id, partition_data in partitions.items()
                if dataset in partition_data
            )

    return partition_ids


def get_missing_job_partitions(job_dir, dataset):
    """
    Return the partition IDs listed in the dataset's missing.txt file.

    Parameters
    ----------
    job_dir : str or pathlib.Path
        Root directory containing the dataset subdirectories.
    dataset : str
        Dataset name.

    Returns
    -------
    set[str]
        IDs of the currently missing jobs/partitions.
    """
    missing_path = job_dir / dataset / "missing.txt"

    if not missing_path.is_file():
        return set()

    return {
        line.strip() for line in missing_path.read_text().splitlines() if line.strip()
    }


def analyze_xrootd_error_logs(
    error_files,
    job_dir,
    dataset,
    verbose=False,
):
    """
    Analyze XRootD errors and identify endpoints from logs or missing partitions.

    Endpoint discovery follows this order:
      1. Extract endpoints directly from recent .err logs.
      2. If no endpoints are found, extract endpoints from the files assigned
         to the partitions listed in missing.txt.

    Parameters
    ----------
    error_files : iterable
        Paths to Condor error logs.
    job_dir : str or pathlib.Path
        Root directory containing the dataset subdirectories.
    dataset : str
        Dataset name.
    verbose : bool, optional
        Print detailed diagnostics.

    Returns
    -------
    dict
        Contains sets of endpoints, error types, and missing partition IDs.
    """
    dataset_dir = Path(job_dir) / dataset
    partitions_path = dataset_dir / "partitions.json"

    missing_partition_ids = get_missing_job_partitions(
        Path(job_dir),
        dataset,
    )

    if partitions_path.is_file():
        partitions_data = json.loads(partitions_path.read_text())
    else:
        partitions_data = {}

        if verbose:
            print(f"[{dataset}] Missing partitions.json: {partitions_path}")

    endpoint_pattern = re.compile(
        r"root://[a-zA-Z0-9._-]+(?::\d+)?",
        flags=re.IGNORECASE,
    )

    endpoints = set()
    error_types = set()

    # Search the error logs first.
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

    # Fall back to endpoints associated with currently missing partitions.
    if not endpoints:
        for partition_id in missing_partition_ids:
            partition_data = partitions_data.get(str(partition_id), {})

            for sample_data in partition_data.values():
                for url in sample_data.get("files", {}):
                    match = endpoint_pattern.match(url)

                    if match:
                        endpoints.add(match.group(0))

        if verbose:
            print(
                f"[{dataset}] Endpoint fallback from missing partitions: "
                f"{sorted(endpoints)}"
            )

    if verbose:
        print(f"[{dataset}] Error types: {sorted(error_types)}")
        print(f"[{dataset}] Endpoints: {sorted(endpoints)}")
        print(f"[{dataset}] Missing partitions: " f"{sorted(missing_partition_ids)}")

    return {
        "endpoints": endpoints,
        "error_types": error_types,
        "partitions": missing_partition_ids,
    }


def get_datasets_map(year: str):
    dataset_config = get_dataset_config(year)
    datasets_map = {}
    for sample in dataset_config:
        key = dataset_config[sample]["key"]
        if key in datasets_map:
            datasets_map[key] += [sample]
        else:
            datasets_map[key] = [sample]
    return datasets_map


def get_datasets_to_run_over(workflow: str, year: str):
    config_builder = WorkflowConfigBuilder(workflow=workflow)
    workflow_config = config_builder.build_workflow_config()

    samples_keys_to_run_over = []
    if "data" in workflow_config.datasets:
        data_samples = workflow_config.datasets["data"]
        samples_keys_to_run_over += data_samples
    if "mc" in workflow_config.datasets:
        mc_samples = workflow_config.datasets["mc"]
        samples_keys_to_run_over += mc_samples
    if "signal" in workflow_config.datasets:
        signal_samples = workflow_config.datasets["signal"]
        samples_keys_to_run_over += signal_samples

    datasets_to_run_over = []
    datasets_map = get_datasets_map(year)
    for key in samples_keys_to_run_over:
        if key not in datasets_map:
            print(f"\n{key} not availabe for {year}!")
            continue
        datasets_to_run_over += datasets_map[key]
    return datasets_to_run_over


def fileset_checker(samples: list, year: str):
    """check if the fileset for the given year exists, generate it otherwise"""

    filesets_path = Path.cwd() / "analysis" / "filesets"
    fileset_file = filesets_path / f"fileset_{year}_NANO_lxplus.json"

    build_input_fileset = False
    if not fileset_file.exists():
        build_input_fileset = True
    else:
        with open(fileset_file, "r") as f:
            filesets = json.load(f)
        build_input_fileset = any(ds not in filesets for ds in samples)

    if build_input_fileset:
        print("\nBuilding input filesets for:")
        print(yaml.dump(samples, default_flow_style=False, sort_keys=False, indent=2))
        cmd = f"python3 fetch.py --year {year} --samples {' '.join(samples)}"
        subprocess.run(cmd, shell=True)


def get_process_maps(workflow_config, year):
    datasets = workflow_config.datasets
    dataset_configs = get_dataset_config(year)
    sample_keys = np.concatenate(list(datasets.values())).tolist()
    processes = []
    process_name_map = {}
    key_process_map = {}
    for sample in dataset_configs:
        sample_key = dataset_configs[sample]["key"]
        if sample_key in sample_keys:
            sample_process = dataset_configs[sample]["process"]
            if sample_process not in processes:
                processes.append(sample_process)
                process_name_map[sample_process] = sample
                key_process_map[sample_key] = sample_process
    return sorted(processes), process_name_map, key_process_map
