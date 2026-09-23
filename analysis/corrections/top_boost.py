import copy
import correctionlib
import numpy as np
import awkward as ak
from pathlib import Path


def add_top_boost_weight(events, weights_container, year, workflow, dataset, variation):
    if dataset.startswith("TT"):
        # get input variables
        met_pt = events.selected_met.pt
        leading_bjet_pt = ak.pad_none(events.selected_bjets, target=2)[:, 0].pt
        subleading_bjet_pt = ak.pad_none(events.selected_bjets, target=2)[:, 1].pt

        if "mu" in workflow and workflow != "1b1e1mu":
            phase_space = "2b1mu"
            leptons_pt = ak.firsts(events.selected_muons).pt
        elif "e" in workflow and workflow != "1b1mu1e":
            phase_space = "2b1e"
            leptons_pt = ak.firsts(events.selected_electrons).pt

        st = leptons_pt + met_pt + leading_bjet_pt + subleading_bjet_pt
        njet = ak.num(events.selected_lightjets)

        in_binning = (st > 196.4) & (st < 1000.0) & (njet >= 0) & (njet < 9)
        selected_st = st.mask[in_binning]
        selected_njet = njet.mask[in_binning]
        selected_st = ak.fill_none(selected_st, 500.0)
        selected_njet = ak.fill_none(selected_njet, 2.0)

        # load correction set
        correction_file = (
            Path.cwd()
            / "analysis"
            / "data"
            / f"{year}_{phase_space}_boost_weight.json.gz"
        )
        cset = correctionlib.CorrectionSet.from_file(str(correction_file))

        # compute weight
        sf = cset["boost_weight"].evaluate(selected_st, selected_njet)
        weight = ak.where(in_binning, sf, np.ones_like(sf))
    else:
        weight = np.ones(len(events))

    weights_container.add(
        name="top_boost_weight",
        weight=weight,
    )
