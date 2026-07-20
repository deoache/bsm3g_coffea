import os
import shutil
import pathlib
import awkward as ak
from typing import List, Optional


def dump_pa_table(
    arrays: dict, fname: str, location: str, subdirs: Optional[List[str]] = None
):
    subdirs = subdirs or []
    xrd_prefix = "root://"
    pfx_len = len(xrd_prefix)
    xrootd = False
    if xrd_prefix in location:
        try:
            import XRootD
            import XRootD.client

            xrootd = True
        except ImportError as err:
            raise ImportError(
                "Install XRootD python bindings with: conda install -c conda-forge xroot"
            ) from err
    local_file = (
        os.path.abspath(os.path.join(".", fname))
        if xrootd
        else os.path.join(".", fname)
    )
    merged_subdirs = "/".join(subdirs) if xrootd else os.path.sep.join(subdirs)
    destination = (
        location + merged_subdirs + f"/{fname}"
        if xrootd
        else os.path.join(location, os.path.join(merged_subdirs, fname))
    )
    out = {}
    for variable, array in arrays.items():
        if array.ndim == 2:
            out[variable] = ak.firsts(array)
        else:
            out[variable] = array

    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pydict(out)
    if len(table) != 0:  # skip dataframes with empty entries
        pq.write_table(table, local_file)
        if xrootd:
            copyproc = XRootD.client.CopyProcess()
            copyproc.add_job(local_file, destination, force=True)
            copyproc.prepare()
            status, response = copyproc.run()
            if status.status != 0:
                raise Exception(status.message)
            del copyproc
        else:
            dirname = os.path.dirname(destination)
            pathlib.Path(dirname).mkdir(parents=True, exist_ok=True)
            shutil.copy(local_file, destination)
            assert os.path.isfile(destination)
        pathlib.Path(local_file).unlink()


# Function taken from HiggsDNA
# https://gitlab.cern.ch/HiggsDNA-project/HiggsDNA/-/blob/master/higgs_dna/utils/dumping_utils.py
def dump_ak_array(
    akarr: ak.Array,
    fname: str,
    location: str,
    subdirs: Optional[List[str]] = None,
) -> None:
    """
    Dump an ak array to disk at location/'/'.join(subdirs)/fname.
    """
    subdirs = subdirs or []
    xrd_prefix = "root://"
    pfx_len = len(xrd_prefix)
    xrootd = False
    if xrd_prefix in location:
        try:
            import XRootD  # type: ignore
            import XRootD.client  # type: ignore

            xrootd = True
        except ImportError as err:
            raise ImportError(
                "Install XRootD python bindings with: conda install -c conda-forge xroot"
            ) from err
    local_file = (
        os.path.abspath(os.path.join(".", fname))
        if xrootd
        else os.path.join(".", fname)
    )
    merged_subdirs = "/".join(subdirs) if xrootd else os.path.sep.join(subdirs)
    destination = (
        location + merged_subdirs + f"/{fname}"
        if xrootd
        else os.path.join(location, os.path.join(merged_subdirs, fname))
    )
    ak.to_parquet(ak.fill_none(akarr, [], axis=0), local_file)
    if xrootd:
        copyproc = XRootD.client.CopyProcess()
        copyproc.add_job(local_file, destination, force=True)
        copyproc.prepare()
        status, response = copyproc.run()
        if status.status != 0:
            raise Exception(status.message)
        del copyproc
    else:
        dirname = os.path.dirname(destination)
        pathlib.Path(dirname).mkdir(parents=True, exist_ok=True)
        shutil.copy(local_file, destination)
        assert os.path.isfile(destination)
    pathlib.Path(local_file).unlink()


def dump_parquet(
    events,
    weights_container,
    variables_map,
    workflow,
    year,
    category,
    output_location,
    shift_name,
):
    if shift_name == "nominal":
        if hasattr(events, "genWeight"):
            # add weights to variables_map
            variations = ["nominal", *weights_container.variations]
            for variation in variations:
                if variation == "nominal":
                    variables_map[f"weight_nominal"] = weights_container.weight()
                    for partial_weight in weights_container.weightStatistics:
                        variables_map[f"weight_{partial_weight}"] = (
                            weights_container.partial_weight(include=[partial_weight])
                        )
                else:
                    variables_map[f"weight_{variation}"] = weights_container.weight(
                        modifier=variation
                    )

        # save parquet files
        fname = (
            events.behavior["__events_factory__"]._partition_key.replace("/", "_")
            + ".parquet"
        )
        dataset = events.metadata["dataset"]
        subdirs = [workflow, year, dataset, category]
        dump_pa_table(variables_map, fname, output_location, subdirs)
