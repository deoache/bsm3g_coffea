import correctionlib
import numpy as np
import awkward as ak
from analysis.corrections.utils import get_pog_json


def apply_jetvetomaps(events: ak.Array, year: str, mapname: str = "jetvetomap"):
    """
    These are the jet veto maps showing regions with an excess of jets (hot zones) and lack of jets
    (cold zones). Using the phi-symmetry of the CMS detector, these areas with detector and or
    calibration issues can be pinpointed.

    Non-zero value indicates that the region is vetoed
    """
    hname = {
        "2016preVFP": "Summer19UL16_V1",
        "2016postVFP": "Summer19UL16_V1",
        "2017": "Summer19UL17_V1",
        "2018": "Summer19UL18_V1",
        "2022preEE": "Summer22_23Sep2023_RunCD_V1",
        "2022postEE": "Summer22EE_23Sep2023_RunEFG_V1",
        "2023preBPix": "Summer23Prompt23_RunC_V1",
        "2023postBPix": "Summer23BPixPrompt23_RunD_V1",
    }
    cset = correctionlib.CorrectionSet.from_file(get_pog_json("jetvetomaps", year))
    
    jets = events.Jet
    j, n = ak.flatten(jets), ak.num(jets)
    
    jet_eta_mask = np.abs(j.eta) < 5.19
    jet_phi_mask = np.abs(j.phi) < 3.14
    in_jet_mask = jet_eta_mask & jet_phi_mask
    
    in_jets = j.mask[in_jet_mask]
    jets_eta = ak.fill_none(in_jets.eta, 0.0)
    jets_phi = ak.fill_none(in_jets.phi, 0.0)
    
    vetomaps = cset[hname[year]].evaluate(mapname, jets_eta, jets_phi)
    vetomaps = ak.any(ak.unflatten(vetomaps, n) == 0, axis=1)
    vetoed_events = events[vetomaps]
    return vetoed_events