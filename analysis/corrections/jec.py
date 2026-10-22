# taken from https://github.com/cms-btv-pog/BTVNanoCommissioning/blob/master/src/BTVNanoCommissioning/utils/correction.py
import yaml
import copy
import gzip
import json
import correctionlib
import numpy as np
import awkward as ak
import importlib.resources
from pathlib import Path

with importlib.resources.open_text(
    f"analysis.corrections", f"jec_params_correctionlib.yaml"
) as file:
    config = yaml.safe_load(file)


CAMPAIGNS = {
    "2016preVFP": "2016preVFP-UL",
    "2016postVFP": "2016postVFP-UL",
    "2017": "2017-UL",
    "2018": "2018-UL",
    "2022preEE": "Summer22",
    "2022postEE": "Summer22EE",
    "2023preBPix": "Summer23",
    "2023postBPix": "Summer23BPix",
    "2024": "Summer24",
}


cset_jersmear_paths = [
    "/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/JER-Smearing/latest/jer_smear.json.gz",
    "/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/jer_smear.json.gz",
]
for _jersmear_path in cset_jersmear_paths:
    if Path(_jersmear_path).exists:
        cset_jersmear = correctionlib.CorrectionSet.from_file(_jersmear_path)
        break
else:
    warnings.warn(
        "JER smearing JSON not found in CVMFS. JER smearing corrections will be disabled.",
        RuntimeWarning,
        stacklevel=2,
    )
    cset_jersmear = {"JERSmear": None}
sf_jersmear = cset_jersmear["JERSmear"]


def get_MET_corr_keys():
    return [
        "pt_raw",
        "eta",
        "phi",
        "area",
        "EmEF",
        "muonSubtrFactor",
        "muonSubtrDeltaPhi",
        "muonSubtrDeltaEta",
        "Genpt",
        "event_rho",
        "EventID",
        "run",
    ]


def fixPhiRange(phi):
    if phi < -np.pi:
        phi = phi + 2.0 * np.pi
    if phi > np.pi:
        phi = phi - 2.0 * np.pi
    return phi


# from https://gitlab.cern.ch/cms-analysis-corrections/JME/examples/-/blob/latest/jercExample.py
def get_corr_inputs(input_dict, corr_obj, jersyst="nom"):
    """
    Helper function for getting values of input variables
    given a dictionary and a correction object.
    """
    input_values = []
    for inputs in corr_obj.inputs:
        if "systematic" in inputs.name:
            input_values.append(jersyst)
        else:
            input_values.append(
                np.array(
                    input_dict[
                        inputs.name.replace("Jet", "")
                        .replace("Pt", "pt")
                        .replace("Phi", "phi")
                        .replace("Eta", "eta")
                        .replace("Mass", "mass")
                        .replace("Rho", "event_rho")
                        .replace("A", "area")
                    ]
                )
            )
    return input_values


# JER systematic eta bins according to
# https://cms-jerc.web.cern.ch/Recommendations/#jet-energy-resolution_1
def get_JER_bins():
    return {
        "split": {
            "eta0to1p93": [0.0, 1.93],
            "eta1p93to2p5": [1.93, 2.5],
        },
        "total": {
            "Total": [0.0, 2.5],
        },
    }


def get_JER(
    correct_map,
    jername,
    j,
    jersyst,
    jet_algo="AK4PFPuppi",
):
    """
    Retrieve JER for each jet pt shift.

    Parameters:
    correct_map (dict): A dictionary containing correction factors and settings for JER.
    jername (str): name of the JER
    j (jets): jet collection
    """

    # New JERC format (2026): ScaleFactor and SFUncertainty are separate keys,
    # both taking (JetEta, JetPt) with no systematic string argument.
    # Variations are computed as SF * (1 ± SFUncertainty).
    JERSF_corr = correct_map["JME"][f"{jername}_ScaleFactor_{jet_algo}"]
    sf_nom = JERSF_corr.evaluate(*get_corr_inputs(j, JERSF_corr))

    JERSF_unc_corr = correct_map["JME"][f"{jername}_SFUncertainty_{jet_algo}"]
    sf_unc = JERSF_unc_corr.evaluate(*get_corr_inputs(j, JERSF_unc_corr))

    if jersyst == "up":
        j["JERSF"] = sf_nom * (1 + sf_unc)
    elif jersyst == "down":
        j["JERSF"] = sf_nom * (1 - sf_unc)
    else:
        j["JERSF"] = sf_nom

    JERptres = correct_map["JME"][f"{jername}_PtResolution_{jet_algo}"]
    JERptres_input = get_corr_inputs(j, JERptres)
    j["JER"] = JERptres.evaluate(*JERptres_input)

    if sf_jersmear is None:
        warnings.warn(
            "JER smearing coefficients unavailable. "
            "Proceeding without applying JER smearing.",
            RuntimeWarning,
            stacklevel=2,
        )
        return ak.ones_like(j.pt)
    else:
        JERsmear_input = get_corr_inputs(j, sf_jersmear)
        return sf_jersmear.evaluate(*JERsmear_input)


def campaign_map():
    dirs = [
        Path("/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/"),
        Path("/cvmfs/cms-griddata.cern.ch/cat/metadata/BTV/"),
        Path("/cvmfs/cms-griddata.cern.ch/cat/metadata/EGM/"),
        Path("/cvmfs/cms-griddata.cern.ch/cat/metadata/MUO/"),
        Path("/cvmfs/cms-griddata.cern.ch/cat/metadata/LUM/"),
    ]
    subdirs = [p.name for d in dirs if d.is_dir() for p in d.iterdir() if p.is_dir()]
    dirnames = {}
    for i in range(len(subdirs)):
        if "Run3" in subdirs[i]:
            if "25Prompt" in subdirs[i]:
                dirnames["Prompt25"] = subdirs[i]
            else:
                dirnames[subdirs[i].split("-")[2]] = subdirs[i]
        elif "Run2" in subdirs[i]:
            key = subdirs[i].split("-")[1] + "-UL"
            dirnames[key] = subdirs[i]
        elif "Run3" not in subdirs[i] and "Run2" not in subdirs[i]:
            continue
        else:
            raise ValueError("Unknown campaign name")
    return dirnames


def _cvmfs_dir(campaign, pog):
    """Get CVMFS subdirectory for a campaign and POG, with optional per-POG overrides.

    If the campaign config contains a 'cvmfs_override' dict with a key matching *pog*,
    return that value directly. Otherwise fall back to campaign_map()[campaign].
    This allows campaigns like Winter25 to route MUO/EGM to a different CVMFS era
    than the default one derived from the campaign name.
    """
    overrides = config.get(campaign, {}).get("default", {}).get("cvmfs_override", {})
    if pog in overrides:
        return overrides[pog]
    return campaign_map()[campaign]


def get_correct_map(year):
    campaign = CAMPAIGNS.get(year, year)
    correct_map = {}
    _jme_path = f"/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/{_cvmfs_dir(campaign, 'JME')}/latest/jet_jerc.json.gz"
    # If MC corrections live in a different CVMFS era (e.g.
    # Summer24 MC truth for Winter25 data), merge at JSON
    # level so we get a single native CorrectionSet.
    _jme_mc_era = _cvmfs_dir(campaign, "JME_MC")
    _jme_data_era = _cvmfs_dir(campaign, "JME")
    _mc_path = f"/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/{_jme_mc_era}/latest/jet_jerc.json.gz"
    if _jme_mc_era != _jme_data_era and os.path.exists(_mc_path):
        with gzip.open(_jme_path, "rt") as f:
            _data_json = json.load(f)
        with gzip.open(_mc_path, "rt") as f:
            _mc_json = json.load(f)
        existing = {c["name"] for c in _data_json["corrections"]}
        _data_json["corrections"] += [
            c for c in _mc_json["corrections"] if c["name"] not in existing
        ]
        existing_comp = {c["name"] for c in _data_json.get("compound_corrections", [])}
        _data_json.setdefault("compound_corrections", []).extend(
            c
            for c in _mc_json.get("compound_corrections", [])
            if c["name"] not in existing_comp
        )
        correct_map["JME"] = correctionlib.CorrectionSet.from_string(
            json.dumps(_data_json)
        )
    else:
        correct_map["JME"] = correctionlib.CorrectionSet.from_file(_jme_path)
    correct_map["JME_cfg"] = config[year]
    # FIXME need to store the JSON path in case we want to parse the run binning edges for ad-hoc JEC run clamp fix (see _get_jec_run_edges)
    correct_map["JME_json_path"] = _jme_path
    for dataset in correct_map["JME_cfg"].keys():
        if (
            np.all(
                np.char.find(
                    np.array(list(correct_map["JME"].keys())),
                    correct_map["JME_cfg"][dataset],
                )
            )
            == -1
        ):
            raise (
                f"{dataset} has no JEC map : {correct_map['JME_cfg'][dataset]} available"
            )
    return correct_map


def _infer_jec_jet_algo(correct_map, jecname):
    """Infer jet algorithm suffix (e.g. AK4PFPuppi, AK4PFchs) from JEC compound keys."""
    preferred_algos = ["AK4PFPuppi", "AK4PFchs"]
    compound_map = correct_map["JME"].compound

    for algo in preferred_algos:
        key = f"{jecname}_L1L2L3Res_{algo}"
        if key in compound_map.keys():
            return algo

    available = [
        k for k in compound_map.keys() if k.startswith(f"{jecname}_L1L2L3Res_")
    ]
    if len(available) > 0:
        return available[0].replace(f"{jecname}_L1L2L3Res_", "")

    raise KeyError(
        f"No L1L2L3Res JEC key found for '{jecname}'. Available matches: {available}"
    )


def _resolve_jecname(correct_map, jecname):
    """Resolve configured JEC era name to an available L1L2L3Res prefix.

    Some configs use data-era strings containing a run token (e.g. ``..._RunCD_V3_DATA``)
    while the JSON can store the corresponding key without that token (e.g. ``..._V3_DATA``).
    """
    compound_map = correct_map["JME"].compound
    if any(k.startswith(f"{jecname}_L1L2L3Res_") for k in compound_map.keys()):
        return jecname

    candidates = []
    if "_Run" in jecname:
        candidates.append(re.sub(r"_Run[^_]+", "", jecname))

    for cand in candidates:
        if any(k.startswith(f"{cand}_L1L2L3Res_") for k in compound_map.keys()):
            warnings.warn(
                f"Configured JEC name '{jecname}' not found, using '{cand}' instead.",
                RuntimeWarning,
                stacklevel=2,
            )
            return cand

    return jecname


# JEC/JES sources for the full set according to
# https://cms-jerc.web.cern.ch/Recommendations/#jet-energy-scale_1
def get_JES_keys(year: int) -> dict[str, set]:
    jes_year_map = {
        "2016preVFP": "2016APV",
        "2022preEE": "2022",
        "2022postEE": "2022EE",
        "2023preBPix": "2023",
        "2023postBPix": "2023BPix",
    }
    return {
        "full": {
            "AbsoluteMPFBias",
            "AbsoluteScale",
            "AbsoluteStat",
            "FlavorQCD",
            "Fragmentation",
            "PileUpDataMC",
            "PileUpPtBB",
            "PileUpPtEC1",
            "PileUpPtEC2",
            "PileUpPtHF",
            "PileUpPtRef",
            "RelativeFSR",
            "RelativeJEREC1",
            "RelativeJEREC2",
            "RelativeJERHF",
            "RelativePtBB",
            "RelativePtEC1",
            "RelativePtEC2",
            "RelativePtHF",
            "RelativeBal",
            "RelativeSample",
            "RelativeStatEC",
            "RelativeStatFSR",
            "RelativeStatHF",
            "SinglePionECAL",
            "SinglePionHCAL",
            "TimePtEta",
        },
        "reduced": {
            f"Regrouped_Absolute_{jes_year_map.get(year, year)}",
            "Regrouped_Absolute",
            f"Regrouped_BBEC1_{jes_year_map.get(year, year)}",
            "Regrouped_BBEC1",
            f"Regrouped_EC2_{jes_year_map.get(year, year)}",
            "Regrouped_EC2",
            "Regrouped_FlavorQCD",
            f"Regrouped_HF_{jes_year_map.get(year, year)}",
            "Regrouped_HF",
            "Regrouped_RelativeBal",
            f"Regrouped_RelativeSample_{jes_year_map.get(year, year)}",
        },
        "total": {"Total"},
    }


def calc_T1_MET(
    met_raw,
    met_nano,
    shifted_jets,
    campaign,
    systematic=False,
):
    """
    Type-1 MET correction following the recipe from:
      https://cms-jerc.web.cern.ch/Type1MET/
    Reference talk (JME General, 2 Feb 2026, Fikri):
      https://indico.cern.ch/event/1644923/#5-met-type-1-correction-comput

    Expects the following pre-computed fields on shifted_jets
    (set by JME_shifts before calling this function):
      - pt_noMuL1:        muon-subtracted raw pT × L1FastJet
      - pt_noMu_fullcorr: muon-subtracted raw pT × L1L2L3Res (× JER/JES-syst)
    CMS Talk discussion:
      https://cms-talk.web.cern.ch/t/unclustered-met-uncertainty-2022-2023-nanoaodv12/141752/5
    """

    MET_x_baseline = met_raw.pt * np.cos(met_raw.phi)
    MET_y_baseline = met_raw.pt * np.sin(met_raw.phi)

    if campaign in ["Summer24", "Winter25", "Prompt25"]:  # NanoAODv15
        jet_mask = (
            (shifted_jets["pt_noMu_fullcorr"] > 15.0)
            & (abs(shifted_jets["eta"]) < 5.2)
            & (shifted_jets["EmEF"] < 0.9)
        )
    else:  # NanoAODv12
        jet_mask = (shifted_jets["pt_noMu_fullcorr"] > 15.0) & (
            abs(shifted_jets["eta"]) < 5.2
        )

    delta_pt = (
        shifted_jets["pt_noMu_fullcorr"][jet_mask] - shifted_jets["pt_noMuL1"][jet_mask]
    )
    try:
        # NanoAODv15: use muon-subtracted phi (phi + muonSubtrDeltaPhi)
        jet_phi = shifted_jets["phi_noMuRaw"][jet_mask]
    except Exception:
        # Fallback for older NanoAOD versions without muonSubtrDeltaPhi
        jet_phi = shifted_jets["phi"][jet_mask]

    MET_x_shifts = delta_pt * np.cos(jet_phi)
    MET_y_shifts = delta_pt * np.sin(jet_phi)
    MET_x = MET_x_baseline - ak.sum(MET_x_shifts, axis=-1)
    MET_y = MET_y_baseline - ak.sum(MET_y_shifts, axis=-1)
    pt_miss_final = np.sqrt(MET_x * MET_x + MET_y * MET_y)
    phi_miss_final = np.arctan2(MET_y, MET_x)
    phi_miss_final = ak.Array([fixPhiRange(_phi) for _phi in phi_miss_final])

    if systematic:
        if hasattr(met_nano, "ptUnclusteredUp"):
            # NanoAODv9 and earlier: extract the unclustered delta from NanoAOD
            # by comparing the shifted MET to the nominal MET in x/y components,
            # then apply that same delta to our recomputed MET.
            met_nano_x = met_nano.pt * np.cos(met_nano.phi)
            met_nano_y = met_nano.pt * np.sin(met_nano.phi)
            dx = (
                met_nano.ptUnclusteredUp * np.cos(met_nano.phiUnclusteredUp)
                - met_nano_x
            )
            dy = (
                met_nano.ptUnclusteredUp * np.sin(met_nano.phiUnclusteredUp)
                - met_nano_y
            )
            pt_unclustered_up = np.sqrt((MET_x + dx) ** 2 + (MET_y + dy) ** 2)
            phi_unclustered_up = np.arctan2(MET_y + dy, MET_x + dx)
            dx_dn = (
                met_nano.ptUnclusteredDown * np.cos(met_nano.phiUnclusteredDown)
                - met_nano_x
            )
            dy_dn = (
                met_nano.ptUnclusteredDown * np.sin(met_nano.phiUnclusteredDown)
                - met_nano_y
            )
            pt_unclustered_down = np.sqrt((MET_x + dx_dn) ** 2 + (MET_y + dy_dn) ** 2)
            phi_unclustered_down = np.arctan2(MET_y + dy_dn, MET_x + dx_dn)
            phi_unclustered_up = ak.Array(
                [fixPhiRange(_phi) for _phi in phi_unclustered_up]
            )
            phi_unclustered_down = ak.Array(
                [fixPhiRange(_phi) for _phi in phi_unclustered_down]
            )
        elif hasattr(met_nano, "MetUnclustEnUpDeltaX"):
            # NanoAODv12+: directional x/y shifts for unclustered energy
            dx = met_nano.MetUnclustEnUpDeltaX
            dy = met_nano.MetUnclustEnUpDeltaY
            pt_unclustered_up = np.sqrt((MET_x + dx) ** 2 + (MET_y + dy) ** 2)
            pt_unclustered_down = np.sqrt((MET_x - dx) ** 2 + (MET_y - dy) ** 2)
            phi_unclustered_up = np.arctan2(MET_y + dy, MET_x + dx)
            phi_unclustered_down = np.arctan2(MET_y - dy, MET_x - dx)
            phi_unclustered_up = ak.Array(
                [fixPhiRange(_phi) for _phi in phi_unclustered_up]
            )
            phi_unclustered_down = ak.Array(
                [fixPhiRange(_phi) for _phi in phi_unclustered_down]
            )
        else:
            # Fallback: isotropic approximation from sumPtUnclustered
            delta = np.sqrt(met_nano.sumPtUnclustered)
            pt_unclustered_up = np.sqrt((MET_x + delta) ** 2 + (MET_y + delta) ** 2)
            pt_unclustered_down = np.sqrt((MET_x - delta) ** 2 + (MET_y - delta) ** 2)
            phi_unclustered_up = np.arctan2(MET_y + delta, MET_x + delta)
            phi_unclustered_down = np.arctan2(MET_y - delta, MET_x - delta)
            phi_unclustered_up = ak.Array(
                [fixPhiRange(_phi) for _phi in phi_unclustered_up]
            )
            phi_unclustered_down = ak.Array(
                [fixPhiRange(_phi) for _phi in phi_unclustered_down]
            )
        return (
            pt_miss_final,
            phi_miss_final,
            pt_unclustered_up,
            pt_unclustered_down,
            phi_unclustered_up,
            phi_unclustered_down,
        )
    else:
        return pt_miss_final, phi_miss_final


_jec_run_edge_cache = {}


def _get_jec_run_edges(correct_map, jecname, jet_algo="AK4PFPuppi"):
    """Return (lo, hi) run clamp range from L2L3Residual bin edges.

    Parses the JME JSON to find the run binning edges directly.
    correctionlib uses half-open [lo, hi) bins, so the last valid
    integer run is edges[-1] - 1.
    Returns None if no run-binned L2L3Residual exists.
    The result is cached per jecname.
    """
    cache_key = (jecname, jet_algo)
    if cache_key in _jec_run_edge_cache:
        return _jec_run_edge_cache[cache_key]

    jme_json_path = correct_map.get("JME_json_path")
    if jme_json_path is None:
        _jec_run_edge_cache[cache_key] = None
        return None

    l2l3_name = f"{jecname}_L2L3Residual_{jet_algo}"
    result = None
    with gzip.open(jme_json_path, "rt") as f:
        for corr in json.load(f).get("corrections", []):
            if corr["name"] != l2l3_name:
                continue
            inputs = corr.get("inputs", [])
            data = corr.get("data", {})
            if (
                inputs
                and inputs[0].get("name") == "run"
                and data.get("nodetype") == "binning"
                and data.get("flow") == "error"
            ):
                edges = data["edges"]
                result = (edges[0], edges[-1] - 1)
            break

    _jec_run_edge_cache[cache_key] = result
    return result


def apply_jerc_corrections(
    events,
    year,
    shifts,
    corrections_config,
):
    """
    Apply Jet Energy Corrections (JEC) and Jet Energy Resolutions (JER) shifts to events.

    This function applies JEC and JER shifts to the jets in the events based on the provided correction map and campaign.
    It handles both real data and simulated data, and can optionally apply systematic variations and exclude jet vetoes.

    Raises:
    KeyError: If required keys are missing in the correct_map.
    ValueError: If the campaign is not recognized or supported.
    """
    campaign = CAMPAIGNS[year]
    isRealData = not hasattr(events, "genWeight")
    correct_map = get_correct_map(year)
    dataset = events.metadata["dataset"]
    systematic = "total" if corrections_config["apply_obj_syst"] else False

    # Year-dependent JES uncertainty names (e.g. Regrouped_Absolute_YYYY)
    # must use the MC JEC campaign year, not the data year.  When MC JECs are
    # borrowed from a different era (e.g. Summer24 JECs for Winter25 data),
    # the config can set "JES_MC_year" to override.
    jes_year = config.get(campaign, {}).get("default", {}).get("JES_MC_year", year)

    if not isRealData and systematic:
        jerc_id_arr = systematic.split("_")
        jes_sources = get_JES_keys(jes_year)
        if len(jerc_id_arr) >= 2 and jerc_id_arr[0] == "JEC":
            jes_sources_id = jerc_id_arr[1]
        else:
            jes_sources_id = "total"  # Default case
        jer_split = get_JER_bins()
        if len(jerc_id_arr) == 4 and jerc_id_arr[2] == "JER":
            jer_split_id = jerc_id_arr[3]
        else:
            jer_split_id = "total"  # Default case
        if systematic == "all":
            jes_sources_id = "full"
            jer_split_id = "split"

    jecname = ""
    if "JME" in correct_map.keys():
        ## correctionlib
        if "JME_cfg" in correct_map.keys():
            if isRealData:
                jecname = [v for k, v in correct_map["JME_cfg"].items() if k in dataset]
                if len(jecname) > 1:
                    raise ValueError("Multiple uncertainties match to this era")
                elif len(jecname) == 0:
                    raise ValueError(
                        "Available JEC variations in this era are not compatible with this file. Did you choose the correct dataset-era combination?"
                    )
                else:
                    jecname = jecname[0] + "_DATA"
            else:
                jecname = correct_map["JME_cfg"]["MC"].split(" ")[0] + "_MC"
                jername = correct_map["JME_cfg"]["MC"].split(" ")[1] + "_MC"

            jecname = _resolve_jecname(correct_map, jecname)
            jet_algo = _infer_jec_jet_algo(correct_map, jecname)

            ## store the original jet info
            nocorrjet = events.Jet
            nocorrjet["pt_nano"] = ak.values_astype(events.Jet["pt"], np.float32)
            nocorrjet["mass_nano"] = ak.values_astype(events.Jet["mass"], np.float32)
            nocorrjet["pt_raw"] = (1 - events.Jet["rawFactor"]) * events.Jet["pt"]
            nocorrjet["mass_raw"] = (1 - events.Jet["rawFactor"]) * events.Jet["mass"]
            nocorrjet["EmEF"] = events.Jet["chEmEF"] + events.Jet["neEmEF"]
            if "muonSubtrDeltaPhi" in events.Jet.fields:
                nocorrjet["muonSubtrDeltaPhi"] = events.Jet["muonSubtrDeltaPhi"]
            else:
                nocorrjet["muonSubtrDeltaPhi"] = ak.zeros_like(nocorrjet.pt)
            if "muonSubtrDeltaEta" in events.Jet.fields:
                nocorrjet["muonSubtrDeltaEta"] = events.Jet["muonSubtrDeltaEta"]
            else:
                nocorrjet["muonSubtrDeltaEta"] = ak.zeros_like(nocorrjet.pt)
            if not isRealData:
                genjetidx = events.Jet.genJetIdx
                # In older NanoAOD versions (e.g. Run 2 NanoAODv9), Jet_genJetIdx
                # can contain non-negative indices that are outside the available
                # GenJet collection. This issue is not observed in the tested
                # newer versions (e.g. NanoAODv15).
                # Therefore, validate both conditions:
                #   - genjetidx >= 0: the jet has a potential GenJet association.
                #   - genjetidx < nGenJet: the index is within the event's GenJet range.
                # Invalid indices must be masked before indexing events.GenJet to
                # avoid out-of-range errors. The resulting unmatched jets receive
                # Genpt = -1.
                valid_genjet = (genjetidx >= 0) & (
                    genjetidx < ak.num(events.GenJet, axis=1)
                )
                # Replace invalid indices with None while preserving the jagged
                # structure of the reconstructed jet collection.
                safe_genjetidx = ak.mask(genjetidx, valid_genjet)
                # Retrieve the matched GenJet transverse momentum.
                # Fill missing associations with -1.
                nocorrjet["Genpt"] = ak.fill_none(
                    events.GenJet[safe_genjetidx].pt,
                    -1,
                )

            nocorrjet["event_rho"] = (
                ak.ones_like(nocorrjet.pt) * events.Rho.fixedGridRhoFastjetAll
                if hasattr(events, "Rho")
                else ak.broadcast_arrays(events.fixedGridRhoFastjetAll, nocorrjet.pt)[0]
            )
            nocorrjet["EventID"] = ak.broadcast_arrays(events.event, nocorrjet.pt)[0]
            nocorrjet["run"] = ak.broadcast_arrays(events.run, nocorrjet.pt)[0]
            jets = copy.copy(nocorrjet)
            jets["pt"] = nocorrjet["pt_raw"]
            jets["mass"] = nocorrjet["mass_raw"]

            ## build jet collection for type-1 MET correction
            nocorrt1metjet = copy.copy(events.CorrT1METJet)
            nocorrt1metjet["pt_raw"] = events.CorrT1METJet["rawPt"]
            if not isRealData:
                nocorrt1metjet["Genpt"] = ak.broadcast_arrays(
                    -1, events.CorrT1METJet.rawPt
                )[0]
            nocorrt1metjet["event_rho"] = (
                ak.ones_like(events.CorrT1METJet.rawPt)
                * events.Rho.fixedGridRhoFastjetAll
                if hasattr(events, "Rho")
                else ak.broadcast_arrays(
                    events.fixedGridRhoFastjetAll, events.CorrT1METJet.rawPt
                )[0]
            )
            nocorrt1metjet["EventID"] = ak.broadcast_arrays(
                events.event, events.CorrT1METJet.rawPt
            )[0]
            nocorrt1metjet["run"] = ak.broadcast_arrays(
                events.run, events.CorrT1METJet.rawPt
            )[0]
            if "muonSubtrDeltaPhi" not in nocorrt1metjet.fields:
                nocorrt1metjet["muonSubtrDeltaPhi"] = ak.zeros_like(
                    events.CorrT1METJet.rawPt
                )
            if "muonSubtrDeltaEta" not in nocorrt1metjet.fields:
                nocorrt1metjet["muonSubtrDeltaEta"] = ak.zeros_like(
                    events.CorrT1METJet.rawPt
                )
            # adding dummy value of 0 will still pass the EmEF < 0.9 cut
            if "EmEF" not in nocorrt1metjet.fields:
                nocorrt1metjet["EmEF"] = ak.zeros_like(events.CorrT1METJet.rawPt)

            keys_keep = get_MET_corr_keys()
            t1jets_1 = nocorrjet[[key for key in nocorrjet.fields if key in keys_keep]]
            t1jets_2 = nocorrt1metjet[
                [key for key in nocorrt1metjet.fields if key in keys_keep]
            ]
            t1jets = ak.concatenate([t1jets_1, t1jets_2], axis=1)
            t1jets["pt"] = t1jets["pt_raw"]

            ## flatten jets
            j, nj = ak.flatten(jets), ak.num(jets)
            t1j, nt1j = ak.flatten(t1jets), ak.num(t1jets)

            ## Pre-compute muon-subtracted Type-1 MET quantities on flat t1 jets.
            ## These are constant: L1 and L1L2L3Res evaluated at pt_noMuRaw.
            t1j["pt_noMuRaw"] = t1j["pt_raw"] * (1.0 - t1j["muonSubtrFactor"])
            t1j["phi_noMuRaw"] = t1j["phi"] + t1j["muonSubtrDeltaPhi"]
            t1j["eta_noMuRaw"] = t1j["eta"] + t1j["muonSubtrDeltaEta"]

            # Type-1 MET recipe depends on jet algorithm:
            # - AK4PFPuppi (PUPPI MET): L2L3-only, no L1 subtraction
            # - AK4PFchs  (PF MET): L1L2L3 - L1
            # See: https://cms-jerc.web.cern.ch/Type1MET/
            if jet_algo == "AK4PFPuppi":
                t1j["pt_noMuL1"] = t1j["pt_noMuRaw"]
            else:
                l1_key = f"{jecname}_L1FastJet_{jet_algo}"
                if l1_key in correct_map["JME"].keys():
                    L1corr = correct_map["JME"][l1_key]
                    if isRealData:
                        t1j["pt_noMuL1"] = t1j["pt_noMuRaw"] * L1corr.evaluate(
                            np.array(t1j["run"]),
                            np.array(t1j["area"]),
                            np.array(t1j["eta_noMuRaw"]),
                            np.array(t1j["pt_noMuRaw"]),
                            np.array(t1j["event_rho"]),
                        )
                    else:
                        t1j["pt_noMuL1"] = t1j["pt_noMuRaw"] * L1corr.evaluate(
                            np.array(t1j["area"]),
                            np.array(t1j["eta_noMuRaw"]),
                            np.array(t1j["pt_noMuRaw"]),
                            np.array(t1j["event_rho"]),
                        )
                else:
                    warnings.warn(
                        f"Missing {l1_key}; falling back to no L1 subtraction for Type-1 MET.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    t1j["pt_noMuL1"] = t1j["pt_noMuRaw"]

            ## store the original met info (nocorrmet), raw met, nanoaod met
            ## Use MET object matching the JEC jet algorithm.
            if (
                jet_algo == "AK4PFPuppi"
                and hasattr(events, "PuppiMET")
                and hasattr(events, "RawPuppiMET")
            ):
                nocorrmet = events.PuppiMET
                met_raw = events.RawPuppiMET
                met_nano = events.PuppiMET
            else:
                nocorrmet = events.MET
                met_raw = events.RawMET
                met_nano = events.MET
            met = copy.copy(nocorrmet)

            ## JES/JEC
            JECcorr = correct_map["JME"].compound[f"{jecname}_L1L2L3Res_{jet_algo}"]
            JEC_input = get_corr_inputs(j, JECcorr)
            JEC_input_t1 = get_corr_inputs(t1j, JECcorr)

            # JES/JEC recipe for 2024 (slide 26)
            # https://indico.cern.ch/event/1602054/timetable/?view=standard_inline_minutes#12-jet-performance-in-run-3
            if year == "2024":
                tmp_pt = copy.copy(JEC_input[2])
                tmp_pt = np.clip(tmp_pt, 30, None)
                JEC_input[2] = tmp_pt
                tmp_pt_t1 = JEC_input_t1[2]
                tmp_pt_t1 = np.clip(tmp_pt_t1, 30, None)
                JEC_input_t1[2] = tmp_pt_t1

            # FIXME Ad-hoc fix: clamp run number for data L2L3Residual boundary bug.
            # correctionlib uses half-open [low, high) bins, so the last run in
            # each era is excluded.  Clamp to (first_edge, last_edge - 1) so
            # that boundary runs still get a valid correction.
            # See: https://cms-talk.web.cern.ch/t/bug-in-2025-jerc-json-file/47675
            if isRealData:
                run_idx = next(
                    (i for i, inp in enumerate(JECcorr.inputs) if inp.name == "run"),
                    None,
                )
                if run_idx is not None:
                    _run_edges = _get_jec_run_edges(correct_map, jecname, jet_algo)
                    if _run_edges is not None:
                        lo, hi = _run_edges
                        JEC_input[run_idx] = np.clip(JEC_input[run_idx], lo, hi)
                        JEC_input_t1[run_idx] = np.clip(JEC_input_t1[run_idx], lo, hi)

            JECflatCorrFactor = JECcorr.evaluate(*JEC_input)
            j["pt_JECnom"] = j["pt"] * JECflatCorrFactor
            j["mass_JECnom"] = j["mass"] * JECflatCorrFactor

            JECflatCorrFactor_t1 = JECcorr.evaluate(*JEC_input_t1)
            t1j["pt_JECnom"] = t1j["pt"] * JECflatCorrFactor_t1

            ## Compound L1L2L3Res at muon-subtracted raw pT (for Type-1 MET)
            JEC_input_t1_nomu = get_corr_inputs(t1j, JECcorr)
            # Override pT and eta with muon-subtracted values
            pt_idx_t1 = next(
                i
                for i, inp in enumerate(JECcorr.inputs)
                if inp.name.replace("Jet", "").replace("Pt", "pt") == "pt"
            )
            JEC_input_t1_nomu[pt_idx_t1] = np.array(t1j["pt_noMuRaw"])
            eta_idx_t1 = next(
                i
                for i, inp in enumerate(JECcorr.inputs)
                if inp.name.replace("Jet", "").replace("Eta", "eta") == "eta"
            )
            JEC_input_t1_nomu[eta_idx_t1] = np.array(t1j["eta_noMuRaw"])
            if year == "2024":
                JEC_input_t1_nomu[pt_idx_t1] = np.clip(
                    JEC_input_t1_nomu[pt_idx_t1], 30, None
                )
            if isRealData:
                run_idx = next(
                    (i for i, inp in enumerate(JECcorr.inputs) if inp.name == "run"),
                    None,
                )
                if run_idx is not None:
                    _run_edges = _get_jec_run_edges(correct_map, jecname, jet_algo)
                    if _run_edges is not None:
                        lo, hi = _run_edges
                        JEC_input_t1_nomu[run_idx] = np.clip(
                            JEC_input_t1_nomu[run_idx], lo, hi
                        )
            t1j["pt_noMu_JEC"] = t1j["pt_noMuRaw"] * JECcorr.evaluate(
                *JEC_input_t1_nomu
            )

            if isRealData:
                ## Only JEC/JES applied to data, no JER
                jets["pt"] = ak.unflatten(j["pt_JECnom"], nj)
                jets["mass"] = ak.unflatten(j["mass_JECnom"], nj)
                t1jets["pt"] = ak.unflatten(t1j["pt_JECnom"], nt1j)
                t1jets["pt_noMuL1"] = ak.unflatten(t1j["pt_noMuL1"], nt1j)
                t1jets["phi_noMuRaw"] = ak.unflatten(t1j["phi_noMuRaw"], nt1j)
                t1jets["pt_noMu_fullcorr"] = ak.unflatten(t1j["pt_noMu_JEC"], nt1j)
                met_pt, met_phi = calc_T1_MET(met_raw, met_nano, t1jets, campaign)
                met["pt"] = met_pt
                met["phi"] = met_phi
            else:
                ## apply nominal JER to nominal JES/JEC
                j["pt"] = j["pt_JECnom"]
                j["mass"] = j["mass_JECnom"]
                jer_smear_nom = get_JER(correct_map, jername, j, "nom", jet_algo)
                j["pt_JECnom_JERnom"] = j["pt_JECnom"] * jer_smear_nom
                j["mass_JECnom_JERnom"] = j["mass_JECnom"] * jer_smear_nom
                jets["pt"] = ak.unflatten(j["pt_JECnom_JERnom"], nj)
                jets["mass"] = ak.unflatten(j["mass_JECnom_JERnom"], nj)

                t1j["pt"] = t1j["pt_JECnom"]
                jer_smear_nom_t1 = get_JER(correct_map, jername, t1j, "nom", jet_algo)
                t1j["pt_JECnom_JERnom"] = t1j["pt_JECnom"] * jer_smear_nom_t1
                t1jets["pt"] = ak.unflatten(t1j["pt_JECnom_JERnom"], nt1j)
                t1jets["pt_noMuL1"] = ak.unflatten(t1j["pt_noMuL1"], nt1j)
                t1jets["phi_noMuRaw"] = ak.unflatten(t1j["phi_noMuRaw"], nt1j)
                t1jets["pt_noMu_fullcorr"] = ak.unflatten(
                    t1j["pt_noMu_JEC"] * jer_smear_nom_t1, nt1j
                )
                (
                    met_pt,
                    met_phi,
                    pt_unclustered_up,
                    pt_unclustered_down,
                    phi_unclustered_up,
                    phi_unclustered_down,
                ) = calc_T1_MET(
                    met_raw,
                    met_nano,
                    t1jets,
                    campaign,
                    True,
                )
                met["pt"] = met_pt
                met["phi"] = met_phi

                ## JES/JEC & JER systematics
                if systematic != False:
                    unc_jets, unc_met = {}, {}
                    for var in ["up", "down"]:
                        fac = 1.0 if var == "up" else -1.0

                        ## apply up/down JER to nominal JES/JEC
                        j["pt"] = j["pt_JECnom"]
                        j["mass"] = j["mass_JECnom"]
                        jer_smear_var = get_JER(correct_map, jername, j, var, jet_algo)
                        j[f"pt_JECnom_JER{var}"] = ak.values_astype(
                            j["pt_JECnom"] * jer_smear_var,
                            np.float32,
                        )
                        j[f"mass_JECnom_JER{var}"] = ak.values_astype(
                            j["mass_JECnom"] * jer_smear_var,
                            np.float32,
                        )
                        t1j["pt"] = t1j["pt_JECnom"]
                        jer_smear_var_t1 = get_JER(
                            correct_map, jername, t1j, var, jet_algo
                        )
                        t1j[f"pt_JECnom_JER{var}"] = ak.values_astype(
                            t1j["pt_JECnom"] * jer_smear_var_t1,
                            np.float32,
                        )
                        t1j["pt"] = t1j[f"pt_JECnom_JER{var}"]
                        _t1jets_var = ak.unflatten(t1j, nt1j)
                        _t1jets_var["pt_noMuL1"] = ak.unflatten(t1j["pt_noMuL1"], nt1j)
                        _t1jets_var["phi_noMuRaw"] = ak.unflatten(
                            t1j["phi_noMuRaw"], nt1j
                        )
                        _t1jets_var["pt_noMu_fullcorr"] = ak.unflatten(
                            t1j["pt_noMu_JEC"] * jer_smear_var_t1, nt1j
                        )

                        ## loop over JER uncertainties
                        if jer_split_id in jer_split.keys():
                            jer_bins = jer_split[jer_split_id]
                            for jer_bin_name in jer_bins.keys():
                                jer_bin_low = jer_bins[jer_bin_name][0]
                                jer_bin_high = jer_bins[jer_bin_name][1]

                                jer_shifted_pt = np.where(
                                    (np.abs(j["eta"]) >= jer_bin_low)
                                    & (np.abs(j["eta"]) < jer_bin_high),
                                    j[f"pt_JECnom_JER{var}"],
                                    j["pt_JECnom_JERnom"],
                                )
                                jer_shifted_mass = np.where(
                                    (np.abs(j["eta"]) >= jer_bin_low)
                                    & (np.abs(j["eta"]) < jer_bin_high),
                                    j[f"mass_JECnom_JER{var}"],
                                    j["mass_JECnom_JERnom"],
                                )
                                _t1jets_var["pt"] = np.where(
                                    (np.abs(_t1jets_var["eta"]) >= jer_bin_low)
                                    & (np.abs(_t1jets_var["eta"]) < jer_bin_high),
                                    _t1jets_var[f"pt_JECnom_JER{var}"],
                                    _t1jets_var["pt_JECnom_JERnom"],
                                )
                                _t1jets_var["pt"] = np.where(
                                    (np.abs(_t1jets_var["eta"]) >= jer_bin_low)
                                    & (np.abs(_t1jets_var["eta"]) < jer_bin_high),
                                    _t1jets_var["pt_noMu_fullcorr"],
                                    t1jets["pt_noMu_fullcorr"],
                                )
                                (
                                    met_pt_JECnom_JERvar,
                                    met_phi_JECnom_JERvar,
                                ) = calc_T1_MET(
                                    met_raw,
                                    met_nano,
                                    _t1jets_var,
                                    campaign,
                                )

                                unc_jets[f"JER{jer_bin_name}{var}"] = copy.copy(
                                    nocorrjet
                                )
                                unc_jets[f"JER{jer_bin_name}{var}"]["pt"] = (
                                    ak.values_astype(
                                        ak.unflatten(jer_shifted_pt, nj),
                                        np.float32,
                                    )
                                )
                                unc_jets[f"JER{jer_bin_name}{var}"]["mass"] = (
                                    ak.values_astype(
                                        ak.unflatten(jer_shifted_mass, nj),
                                        np.float32,
                                    )
                                )

                                unc_met[f"JER{jer_bin_name}{var}"] = copy.copy(
                                    nocorrmet
                                )
                                unc_met[f"JER{jer_bin_name}{var}"]["pt"] = (
                                    ak.values_astype(
                                        met_pt_JECnom_JERvar,
                                        np.float32,
                                    )
                                )
                                unc_met[f"JER{jer_bin_name}{var}"]["phi"] = (
                                    ak.values_astype(
                                        met_phi_JECnom_JERvar,
                                        np.float32,
                                    )
                                )

                        ## unclustered MET uncertainty
                        if var == "up":
                            shifted_met_pt = pt_unclustered_up
                            shifted_met_phi = phi_unclustered_up
                        else:
                            shifted_met_pt = pt_unclustered_down
                            shifted_met_phi = phi_unclustered_down

                        unc_met[f"MET_UnclusteredEnergy{var}"] = copy.copy(nocorrmet)
                        unc_met[f"MET_UnclusteredEnergy{var}"]["pt"] = ak.values_astype(
                            shifted_met_pt,
                            np.float32,
                        )
                        unc_met[f"MET_UnclusteredEnergy{var}"]["phi"] = (
                            ak.values_astype(
                                shifted_met_phi,
                                np.float32,
                            )
                        )

                        ## loop over JES/JEC uncertainties
                        if jes_sources_id in jes_sources.keys():
                            for jes_syst in jes_sources[jes_sources_id]:
                                jesuncmap = correct_map["JME"][
                                    f"{jecname}_{jes_syst}_{jet_algo}"
                                ]

                                jesunc = jesuncmap.evaluate(j.eta, j.pt_JECnom)
                                jesunc_t1 = jesuncmap.evaluate(t1j.eta, t1j.pt_JECnom)

                                ## apply shift up/down to JES/JEC
                                j[f"pt_JEC{jes_syst}{var}"] = ak.values_astype(
                                    j["pt_JECnom"] * (1 + fac * jesunc),
                                    np.float32,
                                )
                                j[f"mass_JEC{jes_syst}{var}"] = ak.values_astype(
                                    j["mass_JECnom"] * (1 + fac * jesunc),
                                    np.float32,
                                )
                                t1j[f"pt_JEC{jes_syst}{var}"] = ak.values_astype(
                                    t1j["pt_JECnom"] * (1 + fac * jesunc_t1),
                                    np.float32,
                                )

                                ## apply nominal JER to up/down JES/JEC
                                j["pt"] = j[f"pt_JEC{jes_syst}{var}"]
                                j["mass"] = j[f"mass_JEC{jes_syst}{var}"]
                                jer_smear_nom = get_JER(
                                    correct_map, jername, j, "nom", jet_algo
                                )
                                j[f"pt_JEC{jes_syst}{var}_JERnom"] = ak.values_astype(
                                    j[f"pt_JEC{jes_syst}{var}"] * jer_smear_nom,
                                    np.float32,
                                )
                                j[f"mass_JEC{jes_syst}{var}_JERnom"] = ak.values_astype(
                                    j[f"mass_JEC{jes_syst}{var}"] * jer_smear_nom,
                                    np.float32,
                                )
                                t1j["pt"] = t1j[f"pt_JEC{jes_syst}{var}"]
                                jer_smear_nom_t1 = get_JER(
                                    correct_map, jername, t1j, "nom", jet_algo
                                )
                                t1j[f"pt_JEC{jes_syst}{var}_JERnom"] = ak.values_astype(
                                    t1j[f"pt_JEC{jes_syst}{var}"] * jer_smear_nom_t1,
                                    np.float32,
                                )
                                t1j["pt"] = t1j[f"pt_JEC{jes_syst}{var}_JERnom"]
                                _t1jets_jesvar = ak.unflatten(t1j, nt1j)
                                _t1jets_jesvar["pt_noMuL1"] = ak.unflatten(
                                    t1j["pt_noMuL1"], nt1j
                                )
                                _t1jets_jesvar["phi_noMuRaw"] = ak.unflatten(
                                    t1j["phi_noMuRaw"], nt1j
                                )
                                # JES syst factor = (1 + fac*jesunc), applied to
                                # both the nominal JEC and the muon-subtracted JEC
                                _t1jets_jesvar["pt_noMu_fullcorr"] = ak.unflatten(
                                    t1j["pt_noMu_JEC"]
                                    * (1 + fac * jesunc_t1)
                                    * jer_smear_nom_t1,
                                    nt1j,
                                )
                                (
                                    met_pt_JECvar_JERnom,
                                    met_phi_JECvar_JERnom,
                                ) = calc_T1_MET(
                                    met_raw,
                                    met_nano,
                                    _t1jets_jesvar,
                                    campaign,
                                )

                                ## JES/JEC uncertainty
                                unc_jets[f"JES_{jes_syst}{var}"] = copy.copy(nocorrjet)
                                unc_jets[f"JES_{jes_syst}{var}"]["pt"] = (
                                    ak.values_astype(
                                        ak.unflatten(
                                            j[f"pt_JEC{jes_syst}{var}_JERnom"], nj
                                        ),
                                        np.float32,
                                    )
                                )
                                unc_jets[f"JES_{jes_syst}{var}"]["mass"] = (
                                    ak.values_astype(
                                        ak.unflatten(
                                            j[f"mass_JEC{jes_syst}{var}_JERnom"], nj
                                        ),
                                        np.float32,
                                    )
                                )

                                unc_met[f"JES_{jes_syst}{var}"] = copy.copy(nocorrmet)
                                unc_met[f"JES_{jes_syst}{var}"]["pt"] = (
                                    ak.values_astype(
                                        met_pt_JECvar_JERnom,
                                        np.float32,
                                    )
                                )
                                unc_met[f"JES_{jes_syst}{var}"]["phi"] = (
                                    ak.values_astype(
                                        met_phi_JECvar_JERnom,
                                        np.float32,
                                    )
                                )

                    if jer_split_id in jer_split.keys():
                        for jer_bin_name in jer_split[jer_split_id]:
                            jets[f"JER_{jer_bin_name}"] = ak.zip(
                                {
                                    "up": unc_jets[f"JER{jer_bin_name}up"],
                                    "down": unc_jets[f"JER{jer_bin_name}down"],
                                }
                            )
                            met[f"JER_{jer_bin_name}"] = ak.zip(
                                {
                                    "up": unc_met[f"JER{jer_bin_name}up"],
                                    "down": unc_met[f"JER{jer_bin_name}down"],
                                }
                            )
                    met["MET_UnclusteredEnergy"] = ak.zip(
                        {
                            "up": unc_met["MET_UnclusteredEnergyup"],
                            "down": unc_met["MET_UnclusteredEnergydown"],
                        }
                    )
                    if jes_sources_id in jes_sources.keys():
                        for jes_syst in jes_sources[jes_sources_id]:
                            jets[f"JES_{jes_syst}"] = ak.zip(
                                {
                                    "up": unc_jets[f"JES_{jes_syst}up"],
                                    "down": unc_jets[f"JES_{jes_syst}down"],
                                }
                            )
                            met[f"JES_{jes_syst}"] = ak.zip(
                                {
                                    "up": unc_met[f"JES_{jes_syst}up"],
                                    "down": unc_met[f"JES_{jes_syst}down"],
                                }
                            )

        else:
            if isRealData:
                if "2016preVFP-UL" == campaign:
                    if "2016B" in dataset or "2016C" in dataset or "2016D" in dataset:
                        jecname = "BCD"
                    elif "2016E" in dataset or "2016F" in dataset:
                        jecname = "EF"
                elif "2016postVFP-UL" == campaign:
                    jecname = "FGH"
                elif campaign == "Rereco17_94X":
                    jecname = ""
                elif campaign == "Summer23":
                    if "v4" in dataset:
                        jecname = "Cv4"
                    else:
                        jecname = "Cv123"
                elif re.search(r"[Rr]un20\d{2}([A-Z])", dataset):
                    jecname = re.search(r"[Rr]un20\d{2}([A-Z])", dataset).group(1)
                else:
                    print("No valid jec name")
                    raise NameError
                jecname = "data" + jecname
            else:
                jecname = "MC"

            jets = correct_map["JME"]["jet_factory"][jecname].build(
                add_jec_variables(events.Jet, events.fixedGridRhoFastjetAll),
                lazy_cache=events.caches[0],
            )
            met = correct_map["JME"]["met_factory"].build(events.PuppiMET, jets, {})

        # Sort the jets by pt
        new_jet_idx = ak.argsort(jets.pt, ascending=False)
        jets = jets[new_jet_idx]

        # systematics
        if not isRealData and systematic != False:
            if "MET_UnclusteredEnergy" in met.fields:
                shifts += [
                    (
                        {
                            "Jet": jets,
                            "MET": met.MET_UnclusteredEnergy.up,
                        },
                        f"CMS_met_unclustered_{year[:4]}Up",
                    ),
                    (
                        {
                            "Jet": jets,
                            "MET": met.MET_UnclusteredEnergy.down,
                        },
                        f"CMS_met_unclustered_{year[:4]}Down",
                    ),
                ]
            if jer_split_id in jer_split.keys():
                for jer_bin_name in jer_split[jer_split_id]:
                    if f"JER_{jer_bin_name}" in jets.fields:
                        shifts += [
                            (
                                {
                                    "Jet": jets[f"JER_{jer_bin_name}"]["up"],
                                    "MET": met[f"JER_{jer_bin_name}"]["up"],
                                },
                                f"CMS_res_j_{jer_bin_name}_{year[:4]}Up",
                            ),
                            (
                                {
                                    "Jet": jets[f"JER_{jer_bin_name}"]["down"],
                                    "MET": met[f"JER_{jer_bin_name}"]["down"],
                                },
                                f"CMS_res_j_{jer_bin_name}_{year[:4]}Down",
                            ),
                        ]
            if jes_sources_id in jes_sources.keys():
                for jes_syst in jes_sources[jes_sources_id]:
                    if f"JES_{jes_syst}" in jets.fields:
                        shifts += [
                            (
                                {
                                    "Jet": jets[f"JES_{jes_syst}"]["up"],
                                    "MET": met[f"JES_{jes_syst}"]["up"],
                                },
                                f"CMS_scale_j_{jes_syst}_{year[:4]}Up",
                            ),
                            (
                                {
                                    "Jet": jets[f"JES_{jes_syst}"]["down"],
                                    "MET": met[f"JES_{jes_syst}"]["down"],
                                },
                                f"CMS_scale_j_{jes_syst}_{year[:4]}Down",
                            ),
                        ]

    else:
        met = events.PuppiMET
        jets = events.Jet

    shifts.insert(0, ({"Jet": jets, "MET": met}, None))
    return shifts
