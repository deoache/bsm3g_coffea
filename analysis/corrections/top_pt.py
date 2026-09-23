import numpy as np
import awkward as ak


def top_pt_sf(pt):
    # https://twiki.cern.ch/twiki/bin/viewauth/CMS/TopPtReweighting#TOP_PAG_corrections_based_on_the
    return 0.103 * np.exp(-0.0118 * pt) - 0.000134 * pt + 0.973


def add_top_pt_weight(events, weights_container, dataset, variation):
    # https://twiki.cern.ch/twiki/bin/viewauth/CMS/TopPtReweighting#How_to_practically_apply_default
    if dataset.startswith("TT"):
        top = events.GenPart[
            (events.GenPart.pdgId == 6) & events.GenPart.hasFlags(["isLastCopy"])
        ]
        anti_top = events.GenPart[
            (events.GenPart.pdgId == -6) & events.GenPart.hasFlags(["isLastCopy"])
        ]

        top_pt_weight = np.sqrt(
            top_pt_sf(ak.flatten(top.pt, axis=-1))
            * top_pt_sf(ak.flatten(anti_top.pt, axis=-1))
        )
    else:
        top_pt_weight = ak.ones(events)

    weights_container.add(
        name="top_pt_weight",
        weight=top_pt_weight,
    )
