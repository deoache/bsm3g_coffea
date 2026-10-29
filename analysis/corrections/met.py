import re
import correctionlib
import numpy as np
import awkward as ak
from typing import Tuple
from analysis.corrections import correction_files


def apply_met_phi_corrections(
    events: ak.Array,
    year: str,
    shifts: list,
    corrections_config: dict,
):
    """
    Apply MET phi modulation corrections

    Docs: (from https://twiki.cern.ch/twiki/bin/viewauth/CMS/MissingETRun2Corrections#xy_Shift_Correction_MET_phi_modu)
        The xy-Shift correction reduces the MET phi modulation. The distribution of true MET is independent of phi because of the rotational symmetry of the collisions around the beam axis. However, we observe that the reconstructed MET does depend on phi. The MET phi distribution has roughly a sinusoidal curve with the period of 2pi. The possible causes of the modulation include anisotropic detector responses, inactive calorimeter cells or tracking regions, the detector misalignment, the displacement of the beam spot. The amplitude of the modulation increases roughly linearly with the number of the pile-up interactions. We can reduce the amplitude of the phi modulation by shifting the origin of the coordinate in the transverse momentum plane as a function of different particle species and in bins of eta.
    """
    # load correction set
    cset = correctionlib.CorrectionSet.from_file(correction_files["met"][year])

    # take MET from nominal shift (corrected after JEC)
    met = shifts[0][0]["MET"]

    # make sure to not cross the maximum allowed value for uncorrected met
    met_pt_raw = ak.ones_like(met.pt) * met.pt
    met_pt = met_pt_raw
    met_pt = np.clip(met_pt, 0.0, 6499.0)

    met_phi_raw = ak.ones_like(met.phi) * met.phi
    met_phi = met_phi_raw
    met_phi = np.clip(met_phi, -3.15, 3.15)

    # use correct run ranges when working with data, otherwise use uniform run numbers in an arbitrary large window
    run_ranges = {
        "2016preVFP": [272007, 278771],
        "2016postVFP": [278769, 284045],
        "2017": [297020, 306463],
        "2018": [315252, 325274],
        # "2024": [378971, 387121],
    }
    data_kind = "MC" if hasattr(events, "genWeight") else "DATA"

    if year in run_ranges:
        if data_kind == "MC":
            run = np.random.randint(
                run_ranges[year][0], run_ranges[year][1], size=len(met_pt)
            )
        else:
            run = events.run

        run_key = "run3" if year.startswith("202") else "run2"
        pt_corr_file = (
            f"pt_metphicorr_pfmet_{data_kind.lower()}"
            if run_key == "run2"
            else f"pt_metphicorr_puppimet_{data_kind.lower()}"
        )
        corrected_met_pt = cset[pt_corr_file].evaluate(
            met_pt.to_numpy(),
            met_phi.to_numpy(),
            events.PV.npvsGood.to_numpy(),
            run,
        )
        phi_corr_file = (
            f"phi_metphicorr_pfmet_{data_kind.lower()}"
            if run_key == "run2"
            else f"phi_metphicorr_puppimet_{data_kind.lower()}"
        )
        corrected_met_phi = cset[phi_corr_file].evaluate(
            met_pt.to_numpy(),
            met_phi.to_numpy(),
            events.PV.npvsGood.to_numpy(),
            run,
        )
        # assign nominal values to MET
        met["pt"] = corrected_met_pt
        met["phi"] = corrected_met_phi
        shifts[0][0]["MET"] = met

    elif year.startswith("202") and year != "2024":
        # prepare common evaluation arguments (excluding pt_phi and variation)
        epoch_str = f"{year[:4] + re.search(r'post(.*)', year).group(1) if 'post' in year else year[:4]}"

        # schema: (met_type, epoch, dtmc, variation, met_pt, met_phi, npvGood)
        eval_inputs = lambda var_name, pu_var="nom": (
            var_name,
            "PuppiMET",
            epoch_str,
            data_kind,
            pu_var,
            met_pt.to_numpy(),
            met_phi.to_numpy(),
            events.PV.npvsGood.to_numpy(),
        )

        # evaluate nominal corrections
        pt_nom = cset["met_xy_corrections"].evaluate(*eval_inputs("pt"))
        phi_nom = cset["met_xy_corrections"].evaluate(*eval_inputs("phi"))

        # assign nominal values to MET
        met["pt"] = pt_nom
        met["phi"] = phi_nom
        shifts[0][0]["MET"] = met

        if hasattr(events, "genWeight") and corrections_config["apply_obj_syst"]:
            # evaluate individual statistical x/y variations
            pt_stat_xup = cset["met_xy_corrections"].evaluate(
                *eval_inputs("pt_stat_xup")
            )
            pt_stat_xdn = cset["met_xy_corrections"].evaluate(
                *eval_inputs("pt_stat_xdn")
            )
            pt_stat_yup = cset["met_xy_corrections"].evaluate(
                *eval_inputs("pt_stat_yup")
            )
            pt_stat_ydn = cset["met_xy_corrections"].evaluate(
                *eval_inputs("pt_stat_ydn")
            )

            phi_stat_xup = cset["met_xy_corrections"].evaluate(
                *eval_inputs("phi_stat_xup")
            )
            phi_stat_xdn = cset["met_xy_corrections"].evaluate(
                *eval_inputs("phi_stat_xdn")
            )
            phi_stat_yup = cset["met_xy_corrections"].evaluate(
                *eval_inputs("phi_stat_yup")
            )
            phi_stat_ydn = cset["met_xy_corrections"].evaluate(
                *eval_inputs("phi_stat_ydn")
            )

            # combine uncorrelated x/y variations in quadrature (up and down)
            delta_pt_up = np.sqrt(
                (pt_stat_xup - pt_nom) ** 2 + (pt_stat_yup - pt_nom) ** 2
            )
            delta_pt_dn = np.sqrt(
                (pt_stat_xdn - pt_nom) ** 2 + (pt_stat_ydn - pt_nom) ** 2
            )

            delta_phi_up = np.sqrt(
                (phi_stat_xup - phi_nom) ** 2 + (phi_stat_yup - phi_nom) ** 2
            )
            delta_phi_dn = np.sqrt(
                (phi_stat_xdn - phi_nom) ** 2 + (phi_stat_ydn - phi_nom) ** 2
            )

            met_up, met_down = met, met
            met_up["pt"] = pt_nom + delta_pt_up
            met_up["phi"] = phi_nom + delta_phi_up
            met_down["pt"] = pt_nom - delta_pt_dn
            met_down["phi"] = phi_nom - delta_phi_dn

            shifts += [
                (
                    {
                        "Jet": shifts[0][0]["Jet"],
                        "MET": met_up,
                        "Muon": shifts[0][0]["Muon"],
                        "Electron": shifts[0][0]["Electron"],
                        "Tau": shifts[0][0]["Tau"],
                    },
                    f"CMS_met_xy_{year[:4]}Up",
                )
            ]
            shifts += [
                (
                    {
                        "Jet": shifts[0][0]["Jet"],
                        "MET": met_down,
                        "Muon": shifts[0][0]["Muon"],
                        "Electron": shifts[0][0]["Electron"],
                        "Tau": shifts[0][0]["Tau"],
                    },
                    f"CMS_met_xy_{year[:4]}Down",
                )
            ]

    return shifts


def corrected_polar_met(
    met_pt,
    met_phi,
    other_phi,
    other_pt_old,
    other_pt_new,
    positive=None,
    dx=None,
    dy=None,
) -> tuple:
    """
    helper function to compute new MET polar components after some other object pT correction.

    https://github.com/CoffeaTeam/coffea/blob/master/src/coffea/jetmet_tools/CorrectedMETFactory.py#L6
    """
    sin, cos = np.sin(other_phi), np.cos(other_phi)
    met_px = met_pt * np.cos(met_phi) - ak.sum(
        (other_pt_new - other_pt_old) * cos, axis=1
    )
    met_py = met_pt * np.sin(met_phi) - ak.sum(
        (other_pt_new - other_pt_old) * sin, axis=1
    )
    if positive is not None and dx is not None and dy is not None:
        met_px = met_px + dx if positive else x - dx
        met_py = met_py + dy if positive else y - dy

    corrected_met_pt = np.hypot(met_px, met_py)
    corrected_met_phi = np.arctan2(met_py, met_px)
    return corrected_met_pt, corrected_met_phi
