import copy
import correctionlib
import numpy as np
import awkward as ak
from analysis.corrections.utils import get_pog_json
from analysis.corrections.met import corrected_polar_met

# ----------------------------------------------------------------------------------- #
# -- The tau energy scale (TES) corrections for taus are provided  ------------------ #
# --  to be applied to reconstructed tau_h Lorentz vector ----------------------------#
# --  It should be applied to a genuine tau -> genmatch = 5 --------------------------#
# -----------  (pT, mass and energy) in simulated data -------------------------------#
# tau_E  *= tes
# tau_pt *= tes
# tau_m  *= tes
# https://github.com/cms-tau-pog/TauIDsfs/tree/master
# ----------------------------------------------------------------------------------- #


def mask_energy_corrections(tau, year="2017"):
    """
    Build a boolean mask selecting tau objects supported by the correctionlib JSON.
    Filters out unsupported decay modes (dm) and genmatch combinations to prevent
    IndexError exceptions during evaluation.
    """
    run = "2" if year.startswith("201") else "3"

    if run == "2":
        # In Run 2, all decay modes [0, 1, 2, 10, 11] are valid for genmatch 1, 2, 5, 6
        tau_mask_gm = (
            (tau.genPartFlav == 5)
            | (tau.genPartFlav == 1)
            | (tau.genPartFlav == 2)
            | (tau.genPartFlav == 6)
        )
        tau_mask_dm = (
            (tau.decayMode == 0)
            | (tau.decayMode == 1)
            | (tau.decayMode == 2)
            | (tau.decayMode == 10)
            | (tau.decayMode == 11)
        )
        return tau_mask_gm & tau_mask_dm
    else:
        # Muon and jet fakes (genmatch 2 and 6): ALL decay modes [0, 1, 2, 10, 11] are valid
        mask_gm26 = ((tau.genPartFlav == 2) | (tau.genPartFlav == 6)) & (
            (tau.decayMode == 0)
            | (tau.decayMode == 1)
            | (tau.decayMode == 2)
            | (tau.decayMode == 10)
            | (tau.decayMode == 11)
        )

        # Genuine taus (genmatch 5): NEVER accept dm = 2 in Run 3 (2022, 2023, 2024)
        mask_gm5 = (tau.genPartFlav == 5) & (
            (tau.decayMode == 0)
            | (tau.decayMode == 1)
            | (tau.decayMode == 10)
            | (tau.decayMode == 11)
        )

        # Electron fakes (genmatch 1):
        # 2024 JSON does NOT accept dm = 2, whereas 2022/2023 JSONs accept it
        if year.startswith("2024"):
            mask_gm1 = (tau.genPartFlav == 1) & (
                (tau.decayMode == 0)
                | (tau.decayMode == 1)
                | (tau.decayMode == 10)
                | (tau.decayMode == 11)
            )
        else:
            mask_gm1 = (tau.genPartFlav == 1) & (
                (tau.decayMode == 0)
                | (tau.decayMode == 1)
                | (tau.decayMode == 2)
                | (tau.decayMode == 10)
                | (tau.decayMode == 11)
            )

        return mask_gm26 | mask_gm5 | mask_gm1


def apply_tau_energy_scale_corrections(
    events, year, shifts: dict, corrections_config: dict
):
    # define correction set
    cset = correctionlib.CorrectionSet.from_file(
        get_pog_json(json_name="tau", year=year)
    )

    # define tau pt_raw field
    events["Tau", "pt_raw"] = ak.ones_like(events.Tau.pt) * events.Tau.pt
    events["Tau", "mass_raw"] = ak.ones_like(events.Tau.mass) * events.Tau.mass

    flat_taus = ak.flatten(events.Tau)
    counts = ak.num(events.Tau)

    # it is defined the taus will be corrected with the energy scale factor: Only a subset of the initial taus.
    mask = mask_energy_corrections(flat_taus, year)
    taus_filter = flat_taus.mask[mask]

    # fill None values with valid entries
    pt_valid = ak.fill_none(taus_filter.pt_raw, 0)
    eta_valid = ak.fill_none(taus_filter.eta, 0)
    dm_valid = ak.fill_none(taus_filter.decayMode, 0)
    genmatch_valid = ak.fill_none(taus_filter.genPartFlav, 0)

    run = "2" if year.startswith("201") else "3"
    deep_tau = "DeepTau2017v2p1" if run == "2" else "DeepTau2018v2p5"

    # get scale factors
    if run == "2":
        sf_valid = cset["tau_energy_scale"].evaluate(
            pt_valid, eta_valid, dm_valid, genmatch_valid, deep_tau, "nom"
        )
    else:
        sf_valid = cset["tau_energy_scale"].evaluate(
            pt_valid,
            eta_valid,
            dm_valid,
            genmatch_valid,
            deep_tau,
            "Medium",
            "VVLoose",
            "nom",
        )

    corrected_pt = ak.where(mask, taus_filter.pt_raw * sf_valid, flat_taus.pt_raw)
    corrected_mass = ak.where(mask, taus_filter.mass_raw * sf_valid, flat_taus.mass_raw)

    events["Tau", "pt"] = ak.unflatten(corrected_pt, counts)
    events["Tau", "mass"] = ak.unflatten(corrected_mass, counts)
    for i in range(len(shifts)):
        shifts[i][0]["Tau"] = events.Tau

    # Propagate tau pT changes to MET
    met_name = "MET" if run == "2" else "PuppiMET"
    events[met_name, "pt_raw"] = ak.ones_like(events[met_name].pt) * events[met_name].pt
    events[met_name, "phi_raw"] = (
        ak.ones_like(events[met_name].phi) * events[met_name].phi
    )
    corrected_met_pt, corrected_met_phi = corrected_polar_met(
        met_pt=events[met_name].pt,
        met_phi=events[met_name].phi,
        other_phi=events.Tau.phi,
        other_pt_old=events.Tau.pt_raw,
        other_pt_new=events.Tau.pt,
    )
    events[met_name, "pt"] = corrected_met_pt
    events[met_name, "phi"] = corrected_met_phi
    for i in range(len(shifts)):
        shifts[i][0]["MET"] = events[met_name]

    # uncertainties
    if hasattr(events, "genWeight") and corrections_config["apply_obj_syst"]:
        tau_up, tau_down = events.Tau, events.Tau

        if run == "2":
            sf_up_valid = cset["tau_energy_scale"].evaluate(
                pt_valid, eta_valid, dm_valid, genmatch_valid, deep_tau, "up"
            )
            sf_down_valid = cset["tau_energy_scale"].evaluate(
                pt_valid, eta_valid, dm_valid, genmatch_valid, deep_tau, "down"
            )
        else:
            sf_up_valid = cset["tau_energy_scale"].evaluate(
                pt_valid,
                eta_valid,
                dm_valid,
                genmatch_valid,
                deep_tau,
                "Medium",
                "VVLoose",
                "up",
            )
            sf_down_valid = cset["tau_energy_scale"].evaluate(
                pt_valid,
                eta_valid,
                dm_valid,
                genmatch_valid,
                deep_tau,
                "Medium",
                "VVLoose",
                "down",
            )

        # up variation
        up = ak.flatten(events.Tau)
        up["pt"] = ak.where(mask, taus_filter.pt_raw * sf_up_valid, flat_taus.pt_raw)
        up["mass"] = ak.where(
            mask, taus_filter.mass_raw * sf_up_valid, flat_taus.mass_raw
        )
        tau_up = ak.unflatten(up, counts)

        # down variation
        down = ak.flatten(events.Tau)
        down["pt"] = ak.where(
            mask, taus_filter.pt_raw * sf_down_valid, flat_taus.pt_raw
        )
        down["mass"] = ak.where(
            mask, taus_filter.mass_raw * sf_down_valid, flat_taus.mass_raw
        )
        tau_down = ak.unflatten(down, counts)

        # Propagate muon pT changes to MET
        met_up, met_down = events[met_name], events[met_name]
        met_up["pt"], met_up["phi"] = corrected_polar_met(
            events[met_name].pt_raw,
            events[met_name].phi_raw,
            tau_up.phi,
            tau_up.pt_raw,
            tau_up.pt,
        )
        met_down["pt"], met_down["phi"] = corrected_polar_met(
            events[met_name].pt_raw,
            events[met_name].phi_raw,
            tau_down.phi,
            tau_down.pt_raw,
            tau_down.pt,
        )

        shifts += [
            (
                {
                    "Jet": shifts[0][0]["Jet"],
                    "MET": met_up,
                    "Muon": shifts[0][0]["Muon"],
                    "Electron": shifts[0][0]["Electron"],
                    "Tau": tau_up,
                },
                f"CMS_t_energy_{year[:4]}Up",
            )
        ]
        shifts += [
            (
                {
                    "Jet": shifts[0][0]["Jet"],
                    "MET": met_down,
                    "Muon": shifts[0][0]["Muon"],
                    "Electron": shifts[0][0]["Electron"],
                    "Tau": tau_down,
                },
                f"CMS_t_energy_{year[:4]}Down",
            )
        ]
    return shifts
