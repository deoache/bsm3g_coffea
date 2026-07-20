import os
import shutil
import pathlib
import awkward as ak
from typing import List, Optional


def dump_pa_table(
    arrays: dict,
    fname: str,
    location: str,
    subdirs: Optional[List[str]] = None,
    extra_metadata: Optional[dict] = None,
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
        # plain Python lists (e.g. dump_chunk_sumw's {"sumw": [val]}) have no .ndim;
        # only awkward arrays need the 2D->firsts flattening.
        if hasattr(array, "ndim") and array.ndim == 2:
            out[variable] = ak.firsts(array)
        else:
            out[variable] = array

    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pydict(out)
    if extra_metadata:
        existing = dict(table.schema.metadata or {})
        for k, v in extra_metadata.items():
            kk = k.encode() if isinstance(k, str) else k
            vv = v.encode() if isinstance(v, str) else v
            existing[kk] = vv
        table = table.replace_schema_metadata(existing)
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


def dump_chunk_sumw(events, workflow, year, output_location):
    """Write the chunk's full generator sumw to a dedicated record, ALWAYS.

    The data-shard sumw (in dump_parquet) is correct per chunk, but a chunk that
    selects zero events writes no shard at all (base.py guards dump on
    nevents_after > 0, and dump_pa_table skips empty tables), so its sumw is
    silently lost. For low-efficiency samples (vjets, hadronic) most chunks select
    nothing, so the summed parquet sumw can undercount the true generator sumw by
    10-30x -- breaking the lumi*xsec/sumw normalisation.

    This writes a one-row `{sumw: <full chunk genWeight sum>}` parquet per chunk
    into <dataset>/sumw_records/, computed on the pre-selection events, regardless
    of how many events survive. Summing this directory gives the true generator
    sumw (validated to <0.5% against the .coffea cutflow and the Runs-tree
    genEventSumw). MC only.
    """
    if not hasattr(events, "genWeight"):
        return
    dataset = events.metadata["dataset"]
    chunk_sumw = float(ak.sum(events.genWeight))
    pkey = events.behavior["__events_factory__"]._partition_key.replace("/", "_")
    fname = pkey + ".parquet"
    subdirs = ["sumw_records"]
    dump_pa_table(
        {"sumw": [chunk_sumw]},
        fname,
        output_location,
        subdirs,
        extra_metadata={"sumw": str(chunk_sumw)},
    )


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
    from analysis.filesets.utils import get_dataset_config

    is_mc = hasattr(events, "genWeight")
    dataset = events.metadata["dataset"]
        
    if shift_name == "nominal":
        # nominal: full weight set (nominal + per-syst Up/Down columns)
        if is_mc:
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
        subdirs = [category]
    else:
        # object shift: kinematics already reflect the shift; only the nominal
        # weight is needed (weight-systematics are not crossed with object shifts).
        # Output goes to its own <shift> subdir so the MVA/combine path can read
        # each variation independently.
        if is_mc:
            variables_map["weight_nominal"] = weights_container.weight()
        subdirs = [category, shift]

    # Self-normalising metadata: per-shard sumw (genWeight unchanged by object
    # shifts) + xsec/era from the dataset config. merge_parquets aggregates sumw
    # across shards. Lets downstream compute lumi*xsec/sumw without a sidecar.
    extra_metadata = None
    if is_mc:
        dataset_info = get_dataset_config(year).get(dataset, {})
        extra_metadata = {
            "sumw": str(float(ak.sum(events.genWeight))),
            "xsec": str(dataset_info.get("xsec", "")),
            "era":  str(dataset_info.get("era", "")),
        }

    fname = (
        events.behavior["__events_factory__"]._partition_key.replace("/", "_")
        + ".parquet"
    )
    dump_pa_table(variables_map, fname, output_location, subdirs, extra_metadata=extra_metadata)