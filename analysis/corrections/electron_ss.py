import correctionlib
import numpy as np
import awkward as ak
from pathlib import Path
from analysis.corrections.met import corrected_polar_met


def filter_boundaries(pt_corr, pt, nested=True):
    if not nested:
        pt_corr = np.asarray(pt_corr)
        pt = np.asarray(pt)

    # Check for pt values outside the range
    outside_bounds = (pt < 20) | (pt > 250)

    if nested:
        n_pt_outside = ak.sum(ak.any(outside_bounds, axis=-1))
    else:
        n_pt_outside = np.sum(outside_bounds)

    if n_pt_outside > 0:
        # print(
        #    f"There are {n_pt_outside} events with muon pt outside of [26,200] GeV. "
        #    "Setting those entries to their initial value."
        # )
        pt_corr = np.where(pt > 250, pt, pt_corr)
        pt_corr = np.where(pt < 20, pt, pt_corr)

    # Check for NaN entries in pt_corr
    nan_entries = np.isnan(pt_corr)

    if nested:
        n_nan = ak.sum(ak.any(nan_entries, axis=-1))
    else:
        n_nan = np.sum(nan_entries)

    if n_nan > 0:
        # print(
        #    f"There are {n_nan} nan entries in the corrected pt. "
        #    "This might be due to the number of tracker layers hitting boundaries. "
        #    "Setting those entries to their initial value."
        # )
        pt_corr = np.where(np.isnan(pt_corr), pt, pt_corr)

    return pt_corr


def apply_electron_ss_corrections(
    events: ak.Array, year: str, shifts: dict, corrections_config: dict
):
    """
    Apply electron scale and smearing corrections for Run3

    from docs https://egammapog.docs.cern.ch/Run3/SaS/

    The purpose of scale and smearing (more formal: energy scale and resolution corrections) is to correct and calibrate electron and photon energies in data and MC. This step is performed after the MC-based semi-parametric EGamma energy regression, which is applied to both MC and data (aiming to correct for inherent imperfections that are not in principle related to data/MC differences like crystal-by-crystal differences, intermodule gaps, ...). The remaining differences between the electron and photon energy scales and the resolution in the data and simulation need to be corrected for. To address this, a multistep procedure is implemented to calibrate the residual energy scale in data. Additionally, an extra smearing is applied to the electron or photon energy in simulation to ensure that the energy resolution matches that observed in data

    https://gitlab.cern.ch/cms-analysis-corrections/EGM/examples/-/blob/latest/egmScaleAndSmearingExample.py
    """
    electron_ss_files = {
        "2022preEE": "/cvmfs/cms-griddata.cern.ch/cat/metadata/EGM/Run3-22CDSep23-Summer22-NanoAODv12/latest/electronSS_EtDependent.json.gz",
        "2022postEE": "/cvmfs/cms-griddata.cern.ch/cat/metadata/EGM/Run3-22EFGSep23-Summer22EE-NanoAODv12/latest/electronSS_EtDependent.json.gz",
        "2023preBPix": "/cvmfs/cms-griddata.cern.ch/cat/metadata/EGM/Run3-23CSep23-Summer23-NanoAODv12/latest/electronSS_EtDependent.json.gz",
        "2023postBPix": "/cvmfs/cms-griddata.cern.ch/cat/metadata/EGM/Run3-23DSep23-Summer23BPix-NanoAODv12/latest/electronSS_EtDependent.json.gz",
        "2024": "/cvmfs/cms-griddata.cern.ch/cat/metadata/EGM/Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15/latest/electronSS_EtDependent.json.gz",
    }
    cset = correctionlib.CorrectionSet.from_file(electron_ss_files[year])
    electrons = events.Electron
    counts = ak.num(events.Electron)

    flat_ele = ak.flatten(electrons)
    seedgain = flat_ele.seedGain
    run = np.repeat(events.run, counts)
    sceta = (
        flat_ele.superclusterEta
        if year == "2024"
        else flat_ele.eta + flat_ele.deltaEtaSC
    )
    r9 = flat_ele.r9
    pt = flat_ele.pt

    if hasattr(events, "genWeight"):
        # for MC, we apply the smearing correction
        smear = cset["SmearAndSyst"].evaluate("smear", pt, r9, sceta)
        rng = np.random.default_rng(seed=42)
        random_numbers = rng.normal(loc=0.0, scale=1.0, size=len(pt))
        smearing = 1 + smear * random_numbers
        pt_corrected = pt * smearing
    else:
        # the scale correction is applied to data
        scale = cset.compound["Scale"].evaluate("scale", run, sceta, r9, pt, seedgain)
        pt_corrected = pt * scale

    # add nominal correction to shifts
    electrons["pt"] = ak.unflatten(pt_corrected, counts)
    for i in range(len(shifts)):
        shifts[i][0]["Electron"] = electrons

    # scale and smearing uncertainties should be evaluated on the original MC only
    if hasattr(events, "genWeight") and corrections_config["apply_obj_syst"]:
        ele_smear_up, ele_smear_down = events.Electron, events.Electron
        ele_scale_up, ele_scale_down = events.Electron, events.Electron

        # Obtain the uncertainty on the smearing width using MC original variables (pt, r9, and ScEta).
        unc_smear = cset["SmearAndSyst"].evaluate("smear", pt, r9, sceta)
        smear_up = cset["SmearAndSyst"].evaluate("smear", pt, r9, sceta)
        smear_down = cset["SmearAndSyst"].evaluate("smear", pt, r9, sceta)

        # In 2022, the "smear_down" variation can lead to negative smearing width in some cases, which is unphysical.
        # Therefore, we use max(smear - unc_smear, 0) to ensure the smearing width is non-negative.
        smearing_up = 1 + smear_up * random_numbers
        smearing_down = 1 + smear_down * random_numbers

        ele_smear_up["pt"] = ak.unflatten(pt * smearing_up, counts)
        ele_smear_down["pt"] = ak.unflatten(pt * smearing_down, counts)

        # the scale uncertainty, is also evaluated on MC original variables (pt, r9, and ScEta) BUT applied on the smeared pt
        unc_scale = cset["SmearAndSyst"].evaluate("escale", pt, r9, sceta)
        scale_up = cset["SmearAndSyst"].evaluate("scale_up", pt, r9, sceta)
        scale_down = cset["SmearAndSyst"].evaluate("scale_down", pt, r9, sceta)

        ele_scale_up["pt"] = ak.unflatten(scale_up * pt_corrected, counts)
        ele_scale_down["pt"] = ak.unflatten(scale_down * pt_corrected, counts)

        shifts += [
            (
                {
                    "Jet": shifts[0][0]["Jet"],
                    "MET": shifts[0][0]["MET"],
                    "Muon": shifts[0][0]["Muon"],
                    "Electron": ele_scale_up,
                },
                f"CMS_scale_e_{year[:4]}Up",
            )
        ]
        shifts += [
            (
                {
                    "Jet": shifts[0][0]["Jet"],
                    "MET": shifts[0][0]["MET"],
                    "Muon": shifts[0][0]["Muon"],
                    "Electron": ele_scale_down,
                },
                f"CMS_scale_e_{year[:4]}Down",
            )
        ]
        shifts += [
            (
                {
                    "Jet": shifts[0][0]["Jet"],
                    "MET": shifts[0][0]["MET"],
                    "Muon": shifts[0][0]["Muon"],
                    "Electron": ele_smear_up,
                },
                f"CMS_res_e_{year[:4]}Up",
            )
        ]
        shifts += [
            (
                {
                    "Jet": shifts[0][0]["Jet"],
                    "MET": shifts[0][0]["MET"],
                    "Muon": shifts[0][0]["Muon"],
                    "Electron": ele_smear_down,
                },
                f"CMS_res_e_{year[:4]}Down",
            )
        ]
    return shifts
