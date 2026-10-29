import re
import json
import gzip
import cloudpickle
import correctionlib
import numpy as np
import awkward as ak
import importlib.resources
from coffea import util
from pathlib import Path
from typing import Type, Tuple
from coffea.lookup_tools import extractor
from coffea.analysis_tools import Weights
from coffea.nanoevents.methods.base import NanoEventsArray
from correctionlib.schemav2 import Correction, CorrectionSet

# CorrectionLib files are available from
POG_CORRECTION_PATH = "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration"

# summary of pog scale factors: https://cms-analysis-corrections.docs.cern.ch/
POG_JSONS = {
    "muon": ["MUO", "muon_Z.json.gz"],
    "muon_highpt": ["MUO", "muon_HighPt.json.gz"],
    "electron": ["EGM", "electron.json.gz"],
    "electron_hlt": ["EGM", "electronHlt.json.gz"],
    "tau": ["TAU", "tau.json.gz"],
    "pileup": ["LUM", "puWeights.json.gz"],
    "btag": ["BTV", "btagging.json.gz"],
    "met": ["JME", "met.json.gz"],
    "pujetid": ["JME", "jmar.json.gz"],
    "jetvetomaps": ["JME", "jetvetomaps.json.gz"],
    "jerc": ["JME", "jet_jerc.json.gz"],
}

pog_years = {
    "2016postVFP": "2016postVFP_UL",
    "2016preVFP": "2016preVFP_UL",
    "2017": "2017_UL",
    "2018": "2018_UL",
    "2022preEE": "2022_Summer22",
    "2022postEE": "2022_Summer22EE",
    "2023preBPix": "2023_Summer23",
    "2023postBPix": "2023_Summer23BPix",
    "2024": "2024_Summer24",
}

tau_years = {
    "2022preEE": "Run3-22CDSep23-Summer22-NanoAODv12",
    "2022postEE": "Run3-22EFGSep23-Summer22EE-NanoAODv12",
    "2023preBPix": "Run3-23CSep23-Summer23-NanoAODv12",
    "2023postBPix": "Run3-23DSep23-Summer23BPix-NanoAODv12",
    "2024": "Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15",
}


def get_pog_json(json_name: str, year: str) -> str:
    """
    returns the path to the pog json file

    Parameters:
    -----------
        json_name:
            json name {muon, muon_highpt, electron, tau, pileup, btag, met, pujetid, jetvetomaps}
        year:
            dataset year {2016preVFP, 2016postVFP, 2017, 2018, 2022preEE, 2022postEE, 2023preBPix, 2023postBPix}
    """
    if json_name in POG_JSONS:
        pog_json = POG_JSONS[json_name]
    else:
        print(f"No json for {json_name}")
    if json_name == "pileup" and year == "2024":
        return "/cvmfs/cms-griddata.cern.ch/cat/metadata/LUM/Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15/latest/puWeights_CDEFGHI.json.gz"
    if year == "2024" and (json_name.startswith("electron")):
        EGM_CORRECTION_PATH = "/cvmfs/cms-griddata.cern.ch/cat/metadata/EGM/Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15/latest"
        return f"{EGM_CORRECTION_PATH}/{pog_json[1]}"
    elif year == "2024" and json_name == "btag":
        BTV_CORRECTION_PATH = "/cvmfs/cms-griddata.cern.ch/cat/metadata/BTV/Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15/latest"
        return f"{BTV_CORRECTION_PATH}/{pog_json[1]}"
    elif year in tau_years.keys() and json_name == "tau":
        TAU_CORRECTION_PATH = (
            f"/cvmfs/cms-griddata.cern.ch/cat/metadata/TAU/{tau_years[year]}/latest"
        )
        return f"{TAU_CORRECTION_PATH}/{pog_json[1]}"
    else:
        return (
            f"{POG_CORRECTION_PATH}/POG/{pog_json[0]}/{pog_years[year]}/{pog_json[1]}"
        )


def get_muon_hlt_json(year: str) -> str:
    return f"{Path.cwd()}/analysis/data/{year}_Muon_HLT_Eff.json"


def get_electron_hlt_json(
    kind: str,
    year: str,
) -> str:
    return f"{Path.cwd()}/analysis/data/EGamma_HLT_{kind}_{year}.json.gz"


def unflat_sf(sf: ak.Array, in_limit_mask: ak.Array, n: ak.Array):
    """
    get scale factors for in-limit objects (otherwise assign 1).
    Unflat array to original shape and multiply scale factors event-wise

    Parameters:
    -----------
        sf:
            Array with 1D scale factors
        in_limit_mask:
            Array mask for events with objects within correction limits
        n:
            Array with number of objects per event
    """
    sf = ak.where(in_limit_mask, sf, ak.ones_like(sf))
    return ak.fill_none(ak.prod(ak.unflatten(sf, n), axis=1), value=1)


def get_jer_cset(jer_ptres_tag: str, jer_sf_tag: str, year: str):
    """
    returns correction set for jet smearing

    taken from: https://github.com/cms-nanoAOD/correctionlib/issues/130

    Parameters:
    -----------
        jer_ptres_tag:
            tag for jer pt resolution
        jer_sf_tag:
            tag for jer scale factor
        year:
            dataset year
    """
    with gzip.open(get_pog_json("jerc", year)) as fin:
        cset = CorrectionSet.parse_raw(fin.read())

    cset.corrections = [
        c
        for c in cset.corrections
        if c.name
        in (
            jer_ptres_tag,
            jer_sf_tag,
        )
    ]
    cset.compound_corrections = []
    res = Correction.parse_obj(
        {
            "name": "JERSmear",
            "description": "Jet smearing tool",
            "inputs": [
                {"name": "JetPt", "type": "real"},
                {"name": "JetEta", "type": "real"},
                {
                    "name": "GenPt",
                    "type": "real",
                    "description": "matched GenJet pt, or -1 if no match",
                },
                {"name": "Rho", "type": "real", "description": "entropy source"},
                {"name": "EventID", "type": "int", "description": "entropy source"},
                {"name": "JER", "type": "real", "description": "Jet energy resolution"},
                {
                    "name": "JERsf",
                    "type": "real",
                    "description": "Jet energy resolution scale factor",
                },
            ],
            "output": {"name": "smear", "type": "real"},
            "version": 1,
            "data": {
                "nodetype": "binning",
                "input": "GenPt",
                "edges": [-1, 0, 1],
                "flow": "clamp",
                "content": [
                    # stochastic
                    {
                        # rewrite gen_pt with a random gaussian
                        "nodetype": "transform",
                        "input": "GenPt",
                        "rule": {
                            "nodetype": "hashprng",
                            "inputs": ["JetPt", "JetEta", "Rho", "EventID"],
                            "distribution": "normal",
                        },
                        "content": {
                            "nodetype": "formula",
                            # TODO min jet pt?
                            "expression": "1+sqrt(max(x*x - 1, 0)) * y * z",
                            "parser": "TFormula",
                            # now gen_pt is actually the output of hashprng
                            "variables": ["JERsf", "JER", "GenPt"],
                        },
                    },
                    # deterministic
                    {
                        "nodetype": "formula",
                        # TODO min jet pt?
                        "expression": "1+(x-1)*(y-z)/y",
                        "parser": "TFormula",
                        "variables": ["JERsf", "JetPt", "GenPt"],
                    },
                ],
            },
        }
    )
    cset.corrections.append(res)
    ceval = cset.to_evaluator()
    return ceval


def get_era(input_str):
    # Check if the input string starts with "SingleMuon", "EGamma", "Tau" or "MET"
    if (
        input_str.startswith("SingleMuon")
        or input_str.startswith("SingleElectron")
        or input_str.startswith("Tau")
        or input_str.startswith("MET")
    ):
        match = re.search(
            r"SingleMuon([A-Za-z])|SingleElectron([A-Za-z])|Tau([A-Za-z])|MET([A-Za-z])",
            input_str,
        )
        if match:
            # Return the first matched group (the letter following "Muon" or "EGamma")
            return next(g for g in match.groups() if g is not None)
    # If the input doesn't start with "Muon" or "EGamma", return "MC"
    return "MC"
