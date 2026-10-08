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


def mask_energy_corrections(tau):
    # https://github.com/cms-tau-pog/TauFW/blob/4056e9dec257b9f68d1a729c00aecc8e3e6bf97d/PicoProducer/python/analysis/ETauFakeRate/ModuleETau.py#L320
    # https://gitlab.cern.ch/cms-tau-pog/jsonpog-integration/-/blob/TauPOG_v2/POG/TAU/scripts/tau_tes.py

    tau_mask_gm = (
        (tau.genPartFlav == 5)  # Genuine tau
        | (tau.genPartFlav == 1)  # e -> fake
        | (tau.genPartFlav == 2)  # mu -> fake
        | (tau.genPartFlav == 6)  # unmached
    )
    tau_mask_dm = (
        (tau.decayMode == 0)
        | (tau.decayMode == 1)  # 1 prong
        | (tau.decayMode == 2)  # 1 prong
        | (tau.decayMode == 10)  # 1 prong
        | (tau.decayMode == 11)  # 3 prongs  # 3 prongs
    )
    tau_eta_mask = (tau.eta >= 0) & (tau.eta < 2.5)
    tau_mask = tau_mask_gm & tau_mask_dm  # & tau_eta_mask
    return tau_mask


def apply_tau_energy_scale_corrections(
    events, year, shifts: dict, corrections_config: dict
):
    # define tau pt_raw field
    events["Tau", "pt_raw"] = ak.ones_like(events.Tau.pt) * events.Tau.pt
    events["Tau", "mass_raw"] = ak.ones_like(events.Tau.mass) * events.Tau.mass

    flat_taus = ak.flatten(events.Tau)
    counts = ak.num(events.Tau)

    # it is defined the taus will be corrected with the energy scale factor: Only a subset of the initial taus.
    mask = mask_energy_corrections(flat_taus)
    taus_filter = flat_taus.mask[mask]

    # fill None values with valid entries
    pt = ak.fill_none(taus_filter.pt_raw, 0)
    eta = ak.fill_none(taus_filter.eta, 0)
    dm = ak.fill_none(taus_filter.decayMode, 0)
    genmatch = ak.fill_none(taus_filter.genPartFlav, 2)

    # define correction set
    cset = correctionlib.CorrectionSet.from_file(
        get_pog_json(json_name="tau", year=year)
    )
    # get scale factors
    sf = cset["tau_energy_scale"].evaluate(
        pt, eta, dm, genmatch, "DeepTau2017v2p1", "nom"
    )

    corrected_pt = ak.where(mask, taus_filter.pt_raw * sf, flat_taus.pt_raw)
    corrected_mass = ak.where(mask, taus_filter.mass_raw * sf, flat_taus.mass_raw)

    events["Tau", "pt"] = ak.unflatten(corrected_pt, counts)
    events["Tau", "mass"] = ak.unflatten(corrected_mass, counts)
    for i in range(len(shifts)):
        shifts[i][0]["Tau"] = events.Tau

    # Propagate tau pT changes to MET
    events["MET", "pt_raw"] = events.MET.pt
    events["MET", "phi_raw"] = events.MET.phi
    met = events.MET
    met["pt"], met["phi"] = corrected_polar_met(
        events.MET.pt_raw,
        events.MET.phi_raw,
        events.Muon.phi,
        events.Muon.pt_raw,
        events.Muon.pt,
    )
    for i in range(len(shifts)):
        shifts[i][0]["MET"] = met

    # uncertainties
    if hasattr(events, "genWeight") and corrections_config["apply_obj_syst"]:
        tau_up, tau_down = events.Tau, events.Tau

        # up variation
        sf_up = cset["tau_energy_scale"].evaluate(
            pt, eta, dm, genmatch, "DeepTau2017v2p1", "up"
        )
        up = ak.flatten(events.Tau)
        up["pt"] = ak.where(mask, taus_filter.pt_raw * sf_up, flat_taus.pt_raw)
        up["mass"] = ak.where(mask, taus_filter.mass_raw * sf_up, flat_taus.mass_raw)
        tau_up = ak.unflatten(up, counts)

        # down variation
        sf_down = cset["tau_energy_scale"].evaluate(
            pt, eta, dm, genmatch, "DeepTau2017v2p1", "down"
        )
        down = ak.flatten(events.Tau)
        down["pt"] = ak.where(mask, taus_filter.pt_raw * sf_down, flat_taus.pt_raw)
        down["mass"] = ak.where(
            mask, taus_filter.mass_raw * sf_down, flat_taus.mass_raw
        )
        tau_down = ak.unflatten(down, counts)

        # Propagate muon pT changes to MET
        met_up, met_down = events.MET, events.MET
        met_up["pt"], met_up["phi"] = corrected_polar_met(
            events.MET.pt_raw,
            events.MET.phi_raw,
            tau_up.phi,
            tau_up.pt_raw,
            tau_up.pt,
        )
        met_down["pt"], met_down["phi"] = corrected_polar_met(
            events.MET.pt_raw,
            events.MET.phi_raw,
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
