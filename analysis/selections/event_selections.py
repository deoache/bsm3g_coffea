import json
import numpy as np
import awkward as ak
import importlib.resources
from coffea.lumi_tools import LumiMask
from coffea.analysis_tools import PackedSelection
from analysis.selections.trigger import trigger_mask, trigger_match_mask


def get_metfilters_mask(events, year):
    with importlib.resources.path("analysis.data", "metfilters.json") as path:
        with open(path, "r") as handle:
            metfilters = json.load(handle)[year]
    metfilters_mask = np.ones(len(events), dtype="bool")
    metfilterkey = "mc" if hasattr(events, "genWeight") else "data"
    for mf in metfilters[metfilterkey]:
        if mf in events.Flag.fields:
            metfilters_mask = metfilters_mask & events.Flag[mf]
    return metfilters_mask


def get_lumi_mask(events, year):
    year_map = {
        "2016": "analysis/data/Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt",
        "2017": "analysis/data/Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt",
        "2018": "analysis/data/Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt",
        "2022": "analysis/data/Cert_Collisions2022_355100_362760_Golden.txt",
        "2023": "analysis/data/Cert_Collisions2023_366442_370790_Golden.txt",
        "2024": "analysis/data/Cert_Collisions2024_378981_386951_Golden.txt"
    }
    for key in year_map:
        if year.startswith(key):
            goldenjson = year_map[key]
            break
    else:
        raise ValueError(f"Unrecognized year format: '{year}'")

    if hasattr(events, "genWeight"):
        lumi_mask = np.ones(len(events), dtype="bool")
    else:
        lumi_info = LumiMask(goldenjson)
        lumi_mask = lumi_info(events.run, events.luminosityBlock)
    return lumi_mask == 1


def get_trigger_mask(events, hlt_paths, dataset_key, year):
    return trigger_mask(events, hlt_paths, dataset_key, year)


def get_trigger_match_mask(events, hlt_paths, year, leptons):
    mask = trigger_match_mask(events, hlt_paths, year, leptons)
    return ak.sum(mask, axis=-1) > 0


def get_stitching_mask(events, dataset, dataset_key, ht_value):
    stitching_mask = np.ones(len(events), dtype="bool")
    if dataset.startswith(dataset_key):
        stitching_mask = events.LHE.HT < ht_value
    return stitching_mask


def get_hemcleaning_mask(events, year):
    # hem-cleaning selection
    # https://hypernews.cern.ch/HyperNews/CMS/get/JetMET/2000.html
    # Due to the HEM issue in year 2018, we veto the events with jets and electrons in the
    # region -3 < eta <-1.3 and -1.57 < phi < -0.87 to remove fake MET
    if year == "2018":
        hem_veto = ak.any(
            (
                (events.Jet.eta > -3.2)
                & (events.Jet.eta < -1.3)
                & (events.Jet.phi > -1.57)
                & (events.Jet.phi < -0.87)
            ),
            -1,
        ) | ak.any(
            (
                (events.Electron.pt > 30)
                & (events.Electron.eta > -3.2)
                & (events.Electron.eta < -1.3)
                & (events.Electron.phi > -1.57)
                & (events.Electron.phi < -0.87)
            ),
            -1,
        )
        hem_cleaning = (
            (
                (events.run >= 319077) & (not hasattr(events, "genWeight"))
            )  # if data check if in Runs C or D
            # else for MC randomly cut based on lumi fraction of C&D
            | ((np.random.rand(len(events)) < 0.632) & hasattr(events, "genWeight"))
        ) & (hem_veto)

        return ~hem_cleaning
    return np.ones(len(events), dtype=bool)
