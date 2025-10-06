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
