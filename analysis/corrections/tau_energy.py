import correctionlib
import numpy as np
import awkward as ak
from analysis.corrections.utils import get_pog_json
from analysis.corrections.met import corrected_polar_met

# ----------------------------------------------------------------------------------- #
# -- The tau energy scale (TES) corrections for taus are provided  ------------------ #
# -- to be applied to reconstructed tau_h Lorentz vector ---------------------------- #
# -- It should be applied to genuine taus and specific fake channels ---------------- #
# ----------- (pT, mass, and energy) in simulated data ------------------------------ #
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


def apply_tau_energy_scale_corrections(events, year):
    """
    Apply Tau Energy Scale (TES) corrections and propagate systemic variations
    to Tau objects and the corresponding MET collection (MET for Run 2, PuppiMET for Run 3).
    """
    run = "2" if year.startswith("201") else "3"
    met_name = "MET" if run == "2" else "PuppiMET"

    # Save original raw pT and mass values
    events["Tau", "pt_raw"] = ak.ones_like(events.Tau.pt) * events.Tau.pt
    events["Tau", "mass_raw"] = ak.ones_like(events.Tau.mass) * events.Tau.mass

    out = ak.flatten(events.Tau)
    counts = ak.num(events.Tau)

    # Build dynamically adapted mask based on year
    mask = mask_energy_corrections(out, year=year)

    # Filter ONLY valid Taus into clean 1D arrays without None values
    taus_valid = out[mask]

    pt_valid = taus_valid.pt_raw
    eta_valid = taus_valid.eta
    dm_valid = taus_valid.decayMode
    genmatch_valid = taus_valid.genPartFlav

    cset = correctionlib.CorrectionSet.from_file(
        get_pog_json(json_name="tau", year=year)
    )

    deep_tau = "DeepTau2017v2p1" if run == "2" else "DeepTau2018v2p5"

    # Evaluate scale factors on valid entries only
    if run == "2":
        sf_valid = cset["tau_energy_scale"].evaluate(
            pt_valid, eta_valid, dm_valid, genmatch_valid, deep_tau, "nom"
        )
        sf_up_valid = cset["tau_energy_scale"].evaluate(
            pt_valid, eta_valid, dm_valid, genmatch_valid, deep_tau, "up"
        )
        sf_down_valid = cset["tau_energy_scale"].evaluate(
            pt_valid, eta_valid, dm_valid, genmatch_valid, deep_tau, "down"
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

    # Reconstruct flat 1D arrays starting from raw values
    pt_nom = ak.to_numpy(out.pt_raw).copy()
    mass_nom = ak.to_numpy(out.mass_raw).copy()

    pt_up = ak.to_numpy(out.pt_raw).copy()
    mass_up = ak.to_numpy(out.mass_raw).copy()

    pt_down = ak.to_numpy(out.pt_raw).copy()
    mass_down = ak.to_numpy(out.mass_raw).copy()

    mask_np = ak.to_numpy(mask)

    # Multiply scale factors only at masked valid indices
    pt_nom[mask_np] *= ak.to_numpy(sf_valid)
    mass_nom[mask_np] *= ak.to_numpy(sf_valid)

    pt_up[mask_np] *= ak.to_numpy(sf_up_valid)
    mass_up[mask_np] *= ak.to_numpy(sf_up_valid)

    pt_down[mask_np] *= ak.to_numpy(sf_down_valid)
    mass_down[mask_np] *= ak.to_numpy(sf_down_valid)

    # Re-assemble Tau collection fields
    fields = ak.fields(out)
    out_dict = {field: out[field] for field in fields}

    out_dict["pt"] = ak.Array(pt_nom)
    out_dict["mass"] = ak.Array(mass_nom)

    up = ak.with_field(out, ak.Array(pt_up), where="pt")
    up = ak.with_field(up, ak.Array(mass_up), where="mass")

    down = ak.with_field(out, ak.Array(pt_down), where="pt")
    down = ak.with_field(down, ak.Array(mass_down), where="mass")

    out_dict["tau_energy"] = ak.zip(
        {"up": up, "down": down}, depth_limit=1, with_name="TauSystematic"
    )

    out_parms = out._layout.parameters
    out_updated = ak.zip(
        out_dict, depth_limit=1, parameters=out_parms, behavior=out.behavior
    )
    events["Tau"] = ak.unflatten(out_updated, counts)

    # Propagate nominal tau pT corrections to MET / PuppiMET
    corrected_met_pt, corrected_met_phi = corrected_polar_met(
        met_pt=events[met_name].pt,
        met_phi=events[met_name].phi,
        other_phi=events.Tau.phi,
        other_pt_old=events.Tau.pt_raw,
        other_pt_new=events.Tau.pt,
    )
    events[met_name, "pt"] = corrected_met_pt
    events[met_name, "phi"] = corrected_met_phi

    # Propagate TES systematic variations (up/down) to MET / PuppiMET
    met_up_pt, met_up_phi = corrected_polar_met(
        events[met_name].pt_raw,
        events[met_name].phi_raw,
        events.Tau.phi,
        events.Tau.pt_raw,
        events.Tau.tau_energy.up.pt,
    )
    met_down_pt, met_down_phi = corrected_polar_met(
        events[met_name].pt_raw,
        events[met_name].phi_raw,
        events.Tau.phi,
        events.Tau.pt_raw,
        events.Tau.tau_energy.down.pt,
    )

    met_up = ak.with_field(events[met_name], met_up_pt, where="pt")
    met_up = ak.with_field(met_up, met_up_phi, where="phi")

    met_down = ak.with_field(events[met_name], met_down_pt, where="pt")
    met_down = ak.with_field(met_down, met_down_phi, where="phi")

    met_tau_systematics = ak.zip(
        {"up": met_up, "down": met_down}, depth_limit=1, with_name="METSystematic"
    )
    events[met_name] = ak.with_field(
        events[met_name], met_tau_systematics, where="tau_energy"
    )
