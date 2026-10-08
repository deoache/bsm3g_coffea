import copy
import correctionlib
import awkward as ak
from pathlib import Path
import numpy as np


def getParticles(
    genparticles, lowid=24, highid=24, flags=["fromHardProcess", "isLastCopy"]
):
    absid = abs(genparticles.pdgId)
    return genparticles[
        ((absid >= lowid) & (absid <= highid)) & genparticles.hasFlags(flags)
    ]


def add_isr_weight(events, weights, year, variation, dataset, fit, one_dim, workflow):
    if "zplusjets" in workflow or "zto" in workflow:
        ds = "DYJetsToLL"
    elif "wplusjets" in workflow:
        ds = "WJets"

    if dataset.startswith(ds):

        # load correction set and compute SF
        if one_dim:
            fname = f"{Path.cwd()}/analysis/data/{year}_ztomumu_isr_weight_1d"
        else:
            fname = f"{Path.cwd()}/analysis/data/{year}_ztomumu_isr_weight"
            if fit:
                fname += "_fit"
        fname += ".json.gz"
        cset = correctionlib.CorrectionSet.from_file(fname)

        # select transverse momemntum
        if ds == "DYJetsToLL":
            # for DY+jets, select dimuon pT
            pt = ak.firsts(events.selected_dimuons.pt)
        else:
            # for W+jets, select gen-level W(-> mu nu) pT
            ws = getParticles(events.GenPart, 24)
            is_from_munu = ak.sum(ak.firsts(np.abs(ws.children.pdgId)), axis=1) == 27
            ws = ws.mask[is_from_munu]
            pt = ak.firsts(ws.pt)

        # select number of jets
        njet = ak.num(events.selected_jets)

        none_mask = ak.is_none(pt) | ak.is_none(njet)
        selected_pt = ak.fill_none(pt, 500.0)
        selected_njet = ak.fill_none(njet, 2.0)

        if one_dim:
            sf = cset["isr_weight"].evaluate(selected_pt)
        else:
            sf = cset["isr_weight"].evaluate(selected_pt, selected_njet)

        # add weight to weights container
        weights.add(
            name="isr_weight",
            weight=ak.where(none_mask, ak.ones_like(sf), sf),
        )
