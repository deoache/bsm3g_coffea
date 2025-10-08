import copy
import correctionlib
import awkward as ak
from pathlib import Path


def add_isr_weight(events, weights, year, variation, dataset, fit, one_dim):
    if dataset.startswith("DYJetsToLL"):
        # get input values
        dimuon_pt = ak.firsts(events.selected_dimuons.pt)
        njet = ak.num(events.selected_jets)
        if one_dim:
            in_binning = (dimuon_pt > 0.0) & (dimuon_pt < 1000.0)
        else: 
            in_binning = (dimuon_pt > 0.0) & (dimuon_pt < 1000.0) & (njet > 0) & (njet < 5)
        selected_dimuon_pt = dimuon_pt.mask[in_binning]
        selected_njet = njet.mask[in_binning]
        selected_dimuon_pt = ak.fill_none(selected_dimuon_pt, 500.0)
        selected_njet = ak.fill_none(selected_njet, 2.0)

        # load correction set
        if one_dim:
            fname = f"{Path.cwd()}/analysis/data/{year}_ztojets_isr_weight_1d"
        else:
            fname = f"{Path.cwd()}/analysis/data/{year}_ztojets_isr_weight"
            if fit:
                fname += "_fit"
        fname += ".json.gz"
        cset = correctionlib.CorrectionSet.from_file(fname)

        # compute weight
        if one_dim:
            sf = cset["isr_weight"].evaluate(selected_dimuon_pt)    
        else:
            sf = cset["isr_weight"].evaluate(selected_dimuon_pt, selected_njet)
        weight = ak.where(in_binning, sf, ak.ones_like(sf))

        sf_name = "isr_weight"
        if one_dim:
            sf_name += "_1D"
        else:
            sf_name += "_2D"
        if fit:
            sf_name += "_fit"
        weights.add(
            name=sf_name,
            weight=weight,
        )