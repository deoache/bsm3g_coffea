import copy
import yaml
import glob
import logging
import numpy as np
import pandas as pd
import dask.dataframe as dd
from pathlib import Path
from coffea.util import load, save
from coffea.processor import accumulate
from analysis.filesets.utils import get_dataset_config
from analysis.histograms import HistBuilder, fill_histogram
from analysis.postprocess.utils import (
    print_header,
    get_variations_keys,
    find_kin_and_axis,
    get_lumi_weight,
    accumulate_histograms,
    accumulate_metadata,
    save_cutflows,
    accumulate_and_save_cutflows,
)


def fill_histograms_from_parquets(
    year,
    sample,
    categories,
    workflow_config,
    output_dir,
    include_weights,
    exclude_weights,
):
    """Build and fill histograms from parquet files for a given sample"""
    dataset_config = get_dataset_config(year)
    histogram_config = workflow_config.histogram_config
    variables = list(histogram_config.axes.keys())
    histograms = HistBuilder(workflow_config).build_histogram()

    sample_histograms = copy.deepcopy(histograms)
    for category in categories:
        # merge sample parquets
        sample_parquets_dir = output_dir / f"parquets_{sample}" / category
        sample_parquets = glob.glob(f"{sample_parquets_dir}/*.parquet")
        if len(sample_parquets) == 0:
            continue
        for i, parquet_file in enumerate(sample_parquets, start=1):
            sample_df = pd.read_parquet(parquet_file)
            if len(sample_df) == 0:
                continue
            sample_df = sample_df.replace({None: np.nan})

            logging.info(
                f"filling histograms with partition {i}/{len(sample_parquets)}"
            )
            # build variables map
            variables_map = {}
            variables_mask_map = {}
            for variable in variables:
                if variable in sample_df.columns:
                    variable_array = sample_df[variable].values
                else:
                    logging.info(
                        f"Could not found variable {variable} for sample {sample}"
                    )
                if variable_array.dtype.type is np.object_:
                    variable_array = np.array(
                        [x if x is not None else np.nan for x in variable_array],
                        dtype=bool,
                    )
                variables_map[variable] = variable_array

            # compute nominal weights
            partial_weights = list(
                set(
                    [
                        w.replace("Up", "").replace("Down", "")
                        for w in sample_df.columns
                        if w.startswith("weight") and "nominal" not in w
                    ]
                )
            )
            if include_weights != "all":
                weights_to_include = include_weights.split(",")
                weights_to_include = [f"weight_{w}" for w in weights_to_include]
                partial_weights = [
                    w for w in partial_weights if w in weights_to_include
                ]
            if exclude_weights:
                weights_to_exclude = exclude_weights.split(",")
                weights_to_exclude = [f"weight_{w}" for w in weights_to_exclude]
                partial_weights = [
                    w for w in partial_weights if w not in weights_to_exclude
                ]

            # fill nominal histograms
            nominal_weights = sample_df[partial_weights].prod(axis=1).values
            fill_args = {
                "histograms": sample_histograms,
                "histogram_config": histogram_config,
                "variables_map": variables_map,
                "category": category,
                "flow": True,
                "weights": nominal_weights,
                "variation": "nominal",
            }
            fill_histogram(**fill_args)

            # fill syst variation histograms
            if dataset_config[sample]["era"] in ["mc", "signal"]:
                for syst in partial_weights:
                    for variation in ["Up", "Down"]:
                        syst_name = f"{syst}{variation}"
                        if syst_name in sample_df.columns:
                            fill_args["weights"] = sample_df[syst_name].values
                            fill_args["variation"] = syst_name.replace("weight_", "")
                            fill_histogram(**fill_args)

    logging.info(f"weights: {[w.replace('weight_','') for w in partial_weights]}")
    return sample_histograms


def save_histograms_by_sample(
    grouped_outputs,
    sample,
    year,
    output_dir,
    workflow_config,
    categories,
    nocutflow,
    output_format,
    include_weights,
    exclude_weights,
):
    """Accumulate, scale, and save histograms for a single sample"""
    print_header(f"Processing {sample} outputs")

    # accumulate metadata and compute lumi weight
    metadata = accumulate_metadata(grouped_outputs, sample)
    if len(metadata.keys()) > 1:
        # get histograms
        if output_format == "coffea":
            histograms = accumulate_histograms(grouped_outputs, sample)
        elif output_format == "parquet":
            histograms = fill_histograms_from_parquets(
                year,
                sample,
                categories,
                workflow_config,
                output_dir,
                include_weights,
                exclude_weights,
            )
        else:
            raise ValueError(f"Unsupported output_format: {output_format}")

        # scale histograms by lumi-xsec weight
        weight = get_lumi_weight(year, sample, metadata)
        scaled_histograms = {
            variable: histograms[variable] * weight for variable in histograms
        }
        logging.info(f"histograms scaled with lumi-xsec weight: {weight}")
        save(scaled_histograms, Path(output_dir) / f"{sample}.coffea")

        # save cutflows if requested
        if not nocutflow:
            save_cutflows(metadata, categories, sample, weight, output_dir)


def save_histograms_by_process(
    process: str,
    output_dir: str,
    process_samples_map: dict,
    categories: list,
    nocutflow: bool,
    output_format: str,
):
    """Accumulate and save all outputs for a given physics process"""
    print_header(f"Processing {process} outputs")

    # accumulate and save all histograms into a single dictionary
    coffea_files = []
    for sample in process_samples_map[process]:
        coffea_files += glob.glob(f"{output_dir}/{sample}*.coffea", recursive=True)

    logging.info(f"Accumulating histograms for process {process}")
    hist_to_accumulate = [load(f) for f in coffea_files]
    output_histograms = {process: accumulate(hist_to_accumulate)}
    save(output_histograms, Path(output_dir) / f"{process}.coffea")

    # accumulate and save all parquets into a single parquet file
    """
    if output_format == "parquet":
        logging.info(f"Accumulating parquets for process {process}")
        parquet_files = []
        for sample in process_samples_map[process]:
            parquet_files += glob.glob(
                f"{output_dir}/{sample}*.parquet", recursive=True
            )
        process_df = dd.read_parquet(
            parquet_files, engine="pyarrow", calculate_divisions=False
        ).compute()
        process_df.to_parquet(Path(output_dir) / f"{process}.parquet")
    """
    # accumulate and save cutflows if requested
    if not nocutflow:
        accumulate_and_save_cutflows(
            process, process_samples_map, output_dir, categories
        )
