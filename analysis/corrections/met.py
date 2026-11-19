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
    Apply MET XY corrections to the nominal MET and all existing object shifts.

    The nominal XY correction is evaluated independently for each MET shift,
    using the corresponding MET pt and phi.

    The statistical XY variations are evaluated only for the nominal MET and
    are added as independent systematic shifts.
    """
    # Load correction set
    cset = correctionlib.CorrectionSet.from_file(correction_files["met"][year])

    data_kind = "MC" if hasattr(events, "genWeight") else "DATA"
    npvs = events.PV.npvsGood.to_numpy()

    # Keep the nominal MET before applying XY corrections.
    # These inputs are needed to evaluate the statistical XY variations.
    met_nominal_raw = shifts[0][0]["MET"]
    met_nominal_pt_raw = met_nominal_raw.pt
    met_nominal_phi_raw = met_nominal_raw.phi

    run_ranges = {
        "2016preVFP": [272007, 278771],
        "2016postVFP": [278769, 284045],
        "2017": [297020, 306463],
        "2018": [315252, 325274],
    }

    # Apply the XY correction independently to every existing shift.
    if year in run_ranges:
        # Use the same run number for all shifts in a given event.
        # This is important for MC, where the run number is randomly assigned.
        if data_kind == "MC":
            run = np.random.randint(
                run_ranges[year][0],
                run_ranges[year][1],
                size=len(events),
            )
        else:
            run = events.run

        pt_corr_file = f"pt_metphicorr_pfmet_{data_kind.lower()}"
        phi_corr_file = f"phi_metphicorr_pfmet_{data_kind.lower()}"

        for collections, _ in shifts:
            met = collections["MET"]

            # Clamp MET inputs to the correction range.
            met_pt = np.clip(met.pt, 0.0, 6499.0)
            met_phi = np.clip(met.phi, -3.15, 3.15)

            # Evaluate the correction using this shift's own MET.
            corrected_met_pt = cset[pt_corr_file].evaluate(
                ak.to_numpy(met_pt),
                ak.to_numpy(met_phi),
                npvs,
                run,
            )

            corrected_met_phi = cset[phi_corr_file].evaluate(
                ak.to_numpy(met_pt),
                ak.to_numpy(met_phi),
                npvs,
                run,
            )

            # Replace MET fields without modifying other shifts.
            met = ak.with_field(met, corrected_met_pt, "pt")
            met = ak.with_field(met, corrected_met_phi, "phi")

            collections["MET"] = met

    elif year.startswith("202") and year != "2024":
        # Determine the correction epoch.
        epoch_str = (
            f"{year[:4] + re.search(r'post(.*)', year).group(1)}"
            if "post" in year
            else year[:4]
        )

        correction = cset["met_xy_corrections"]

        def evaluate_xy(var_name, met_pt, met_phi):
            """Evaluate an XY correction for a given MET and variation."""
            return correction.evaluate(
                var_name,
                "PuppiMET",
                epoch_str,
                data_kind,
                "nom",
                ak.to_numpy(np.clip(met_pt, 0.0, 6499.0)),
                ak.to_numpy(np.clip(met_phi, -3.15, 3.15)),
                npvs,
            )

        # Apply nominal XY corrections to all existing shifts.
        for collections, _ in shifts:
            met = collections["MET"]

            corrected_met_pt = evaluate_xy(
                "pt",
                met.pt,
                met.phi,
            )
            corrected_met_phi = evaluate_xy(
                "phi",
                met.pt,
                met.phi,
            )
            met = ak.with_field(met, corrected_met_pt, "pt")
            met = ak.with_field(met, corrected_met_phi, "phi")

            collections["MET"] = met

        # Evaluate statistical XY variations only for MC.
        if data_kind == "MC" and corrections_config["apply_obj_syst"]:
            # Use the original nominal MET inputs, before the nominal XY
            # correction, to evaluate the statistical variations.
            met_pt = met_nominal_pt_raw
            met_phi = met_nominal_phi_raw

            pt_nom = evaluate_xy("pt", met_pt, met_phi)
            phi_nom = evaluate_xy("phi", met_pt, met_phi)

            # Evaluate individual statistical x/y variations.
            pt_stat_xup = evaluate_xy("pt_stat_xup", met_pt, met_phi)
            pt_stat_xdn = evaluate_xy("pt_stat_xdn", met_pt, met_phi)
            pt_stat_yup = evaluate_xy("pt_stat_yup", met_pt, met_phi)
            pt_stat_ydn = evaluate_xy("pt_stat_ydn", met_pt, met_phi)

            phi_stat_xup = evaluate_xy("phi_stat_xup", met_pt, met_phi)
            phi_stat_xdn = evaluate_xy("phi_stat_xdn", met_pt, met_phi)
            phi_stat_yup = evaluate_xy("phi_stat_yup", met_pt, met_phi)
            phi_stat_ydn = evaluate_xy("phi_stat_ydn", met_pt, met_phi)

            # Combine uncorrelated x/y variations in quadrature.
            delta_pt_up = np.sqrt(
                (pt_stat_xup - pt_nom) ** 2 + (pt_stat_yup - pt_nom) ** 2
            )
            delta_pt_down = np.sqrt(
                (pt_stat_xdn - pt_nom) ** 2 + (pt_stat_ydn - pt_nom) ** 2
            )

            delta_phi_up = np.sqrt(
                (phi_stat_xup - phi_nom) ** 2 + (phi_stat_yup - phi_nom) ** 2
            )
            delta_phi_down = np.sqrt(
                (phi_stat_xdn - phi_nom) ** 2 + (phi_stat_ydn - phi_nom) ** 2
            )

            # Build independent MET variations around the nominal XY-corrected MET.
            met_nominal_xy = shifts[0][0]["MET"]

            met_up = ak.with_field(
                met_nominal_xy,
                pt_nom + delta_pt_up,
                "pt",
            )
            met_up = ak.with_field(
                met_up,
                phi_nom + delta_phi_up,
                "phi",
            )

            met_down = ak.with_field(
                met_nominal_xy,
                pt_nom - delta_pt_down,
                "pt",
            )
            met_down = ak.with_field(
                met_down,
                phi_nom - delta_phi_down,
                "phi",
            )

            # Keep all nominal object collections for the XY statistical shifts.
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
