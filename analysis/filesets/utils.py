import re
import glob
import yaml
import json
import subprocess
import numpy as np
from pathlib import Path
from analysis.workflows.config import WorkflowConfigBuilder


def get_rootfiles(year: str, dataset: str):
    main_dir = Path.cwd()
    fileset_path = Path(f"{main_dir}/analysis/filesets")
    with open(f"{fileset_path}/{year}_fileset.yaml", "r") as f:
        dataset_config = yaml.safe_load(f)[dataset]
    # check for .root files in the specified path
    root_files = glob.glob(f"{dataset_config['path']}/*.root")
    if not root_files:
        # if no files found, check in the subdirectories
        root_files = glob.glob(f"{dataset_config['path']}/*/*.root")
    elif not root_files:
        raise FileNotFoundError(
            f"No .root files found in {dataset_config['path']} or its subdirectories."
        )
    return root_files


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


def extract_xrootd_errors(error_files: list) -> tuple[set, bool]:
    """
    Analyze XRootD error logs to identify failing endpoints and executor errors.

    The function distinguishes between two types of failures:
    - Read errors: logs containing XRootD endpoints, which can be mapped to
      sites and used by the existing site-blacklisting workflow.
    - Authentication or connection errors: logs containing known XRootD
      authentication or connection failure messages. These errors indicate
      that the executor may need to be changed to ``iterative``.

    Logs identified as authentication or connection failures are not used
    to extract endpoints. This is appropriate when these logs do not contain
    endpoint information, as observed in the current workflow.

    Parameters
    ----------
    error_files : list[pathlib.Path]
        Paths to the Condor error log files to analyze for a single dataset.

    Returns
    -------
    tuple[set, bool]
        A tuple containing:

        - endpoints (set[str]): Unique XRootD endpoints extracted from logs
          that do not match the authentication or connection error patterns.
          These endpoints can subsequently be mapped to sites.
        - has_auth_or_connection_error (bool): ``True`` if at least one log
          contains a recognized authentication or connection error; otherwise
          ``False``.

    Notes
    -----
    The function recognizes the following error messages:
    - ``Auth failed``
    - ``No protocols left to try``
    - ``[FATAL] Connection error``
    - ``[FATAL] TLS error``

    Error matching is case-insensitive. Endpoint extraction recognizes URLs
    in the form ``root://hostname`` or ``root://hostname:port``.

    The function only identifies error types and extracts endpoints; it
    does not modify input filesets, update executor settings, or resubmit jobs.
    Those actions are handled by the calling workflow.
    """
    endpoints = set()
    has_auth_or_connection_error = False

    endpoint_pattern = r"root://[a-zA-Z0-9\-.]+(?::\d+)?"
    executor_error_patterns = [
        r"Auth failed",
        r"No protocols left to try",
        r"\[FATAL\]\s*Connection error",
        r"\[FATAL\]\s*TLS error",
    ]

    for error_file in error_files:
        content = error_file.read_text(errors="replace")

        if any(
            re.search(pattern, content, flags=re.IGNORECASE)
            for pattern in executor_error_patterns
        ):
            has_auth_or_connection_error = True
            continue

        endpoints.update(
            re.findall(
                endpoint_pattern,
                content,
                flags=re.IGNORECASE,
            )
        )

    return endpoints, has_auth_or_connection_error


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
