import awkward as ak
from coffea.nanoevents.methods import candidate

def delta_r_mask(first, second, threshold=0.4):
    """select objects from 'first' which are at least 'threshold' away from all objects in 'second'."""
    mval = first.metric_table(second)
    return ak.all(mval > threshold, axis=-1)


def select_dileptons(objects, key):
    leptons = ak.zip(
        {
            "pt": objects[key].pt,
            "eta": objects[key].eta,
            "phi": objects[key].phi,
            "mass": objects[key].mass,
            "charge": objects[key].charge,
        },
        with_name="PtEtaPhiMCandidate",
        behavior=candidate.behavior,
    )
    # create pair combinations with all muons
    dileptons = ak.combinations(leptons, 2, fields=["l1", "l2"])
    dileptons = dileptons[ak.argsort(dileptons.l1.pt, axis=1)]
    # add dimuon 4-momentum field
    dileptons["p4"] = dileptons.l1 + dileptons.l2
    dileptons["pt"] = dileptons.p4.pt
    return dileptons


def select_dileptons_qcd(objects, key):
    leptons = ak.zip(
        {
            "pt": objects[key].pt,
            "eta": objects[key].eta,
            "phi": objects[key].phi,
            "mass": objects[key].mass,
            "charge": objects[key].charge,
            "is_loose": objects[key].is_loose,
            "is_tight": objects[key].is_tight,
            "is_loose_not_tight": objects[key].is_loose_not_tight
        },
        with_name="PtEtaPhiMCandidate",
        behavior=candidate.behavior,
    )
    leptons = leptons[ak.argsort(leptons.pt, ascending=False, axis=1)]
    # create pair combinations with all muons
    dileptons = ak.combinations(leptons, 2, fields=["l1", "l2"])
    # add dimuon 4-momentum field
    dileptons["p4"] = dileptons.l1 + dileptons.l2
    dileptons["pt"] = dileptons.p4.pt
    return dileptons