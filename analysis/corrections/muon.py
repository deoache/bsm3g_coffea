import json
import correctionlib
import numpy as np
import awkward as ak
from typing import Type
from pathlib import Path
from .utils import unflat_sf
from coffea.analysis_tools import Weights
from analysis.corrections.utils import pog_years, get_pog_json, get_muon_hlt_json


# https://twiki.cern.ch/twiki/bin/view/CMS/MuonUL2016
# https://twiki.cern.ch/twiki/bin/view/CMS/MuonUL2017
# https://twiki.cern.ch/twiki/bin/view/CMS/MuonUL2018


class MuonCorrector:
    """
    Muon corrector class

    Parameters:
    -----------
    events:
        events collection
    weights:
        Weights object from coffea.analysis_tools
    year:
        Year of the dataset  {2016preVFP, 2016postVFP, 2017, 2018, 2022preEE, 2022postEE, 2023preBPix, 2023postBPix}
    variation:
        syst variation
    id_wp:
        ID working point {'loose', 'medium', 'tight'}
    iso_wp:
        Iso working point {'loose', 'medium', 'tight'}
    """

    def __init__(
        self,
        events,
        weights: Type[Weights],
        year: str = "2017",
        variation: str = "nominal",
        id_wp: str = "tight",
        iso_wp: str = "tight",
    ) -> None:
        self.events = events
        self.muons = events.selected_muons
        self.variation = variation
        self.id_wp = id_wp
        self.iso_wp = iso_wp

        # flat muon array
        self.flat_muons, self.muons_counts = ak.flatten(self.muons), ak.num(self.muons)

        # weights container
        self.weights = weights

        # define correction set
        self.cset = correctionlib.CorrectionSet.from_file(
            get_pog_json(json_name="muon", year=year)
        )
        self.year = year
        year_key_map = {
            "2016preVFP": "2016",
            "2016postVFP": "2016",
            "2022preEE": "2022",
            "2022postEE": "2022",
            "2023preBPix": "2023",
            "2023postBPix": "2023",
        }
        self.year_key = year_key_map.get(year, year)
        self.run_key = "Run3" if self.year_key in ["2022", "2023"] else "Run2"

    def add_reco_weight(self):
        """
        add muon RECO scale factors to weights container
        """
        if self.run_key == "Run2":
            # get muons within SF binning
            muon_pt_mask = self.flat_muons.pt >= 40.0
            muon_eta_mask = np.abs(self.flat_muons.eta) < 2.4
            in_muon_mask = muon_pt_mask & muon_eta_mask
            in_muons = self.flat_muons.mask[in_muon_mask]

            # get muons pT and abseta (replace None values with some 'in-limit' value)
            muon_pt = ak.fill_none(in_muons.pt, 40.0)
            muon_eta = np.abs(ak.fill_none(in_muons.eta, 0.0))

            # 'id' scale factors names
            reco_corrections = {
                "2016preVFP": "NUM_TrackerMuons_DEN_genTracks",
                "2016postVFP": "NUM_TrackerMuons_DEN_genTracks",
                "2017": "NUM_TrackerMuons_DEN_genTracks",
                "2018": "NUM_TrackerMuons_DEN_genTracks",
            }
            # get nominal scale factors
            nominal_sf = unflat_sf(
                self.cset[reco_corrections[self.year]].evaluate(
                    muon_eta, muon_pt, "nominal"
                ),
                in_muon_mask,
                self.muons_counts,
            )
            if self.variation == "nominal":
                # get 'up' and 'down' scale factors
                up_sf = unflat_sf(
                    self.cset[reco_corrections[self.year]].evaluate(
                        muon_eta, muon_pt, "systup"
                    ),
                    in_muon_mask,
                    self.muons_counts,
                )
                down_sf = unflat_sf(
                    self.cset[reco_corrections[self.year]].evaluate(
                        muon_eta, muon_pt, "systdown"
                    ),
                    in_muon_mask,
                    self.muons_counts,
                )
                # add scale factors to weights container
                self.weights.add(
                    name=f"CMS_eff_m_reco_{self.year_key}",
                    weight=nominal_sf,
                    weightUp=up_sf,
                    weightDown=down_sf,
                )
            else:
                self.weights.add(
                    name=f"CMS_eff_m_reco_{self.year_key}",
                    weight=nominal_sf,
                )

    def add_id_weight(self):
        """
        add muon ID scale factors to weights container
        """
        # get muons within SF binning
        muon_pt_mask = (self.flat_muons.pt > 15.0) & (self.flat_muons.pt < 199.999)
        muon_eta_mask = np.abs(self.flat_muons.eta) < 2.39
        in_muon_mask = muon_pt_mask & muon_eta_mask
        in_muons = self.flat_muons.mask[in_muon_mask]

        # get muons pT and abseta (replace None values with some 'in-limit' value)
        muon_pt = ak.fill_none(in_muons.pt, 15.0)
        muon_eta = np.abs(ak.fill_none(in_muons.eta, 0.0))

        # 'id' scale factors names
        id_corrections = {
            "loose": "NUM_LooseID_DEN_TrackerMuons",
            "medium": "NUM_MediumID_DEN_TrackerMuons",
            "tight": "NUM_TightID_DEN_TrackerMuons",
        }
        # get nominal scale factors
        nominal_sf = unflat_sf(
            self.cset[id_corrections[self.id_wp]].evaluate(
                muon_eta, muon_pt, "nominal"
            ),
            in_muon_mask,
            self.muons_counts,
        )
        if self.variation == "nominal":
            # get 'up' and 'down' scale factors
            up_sf = unflat_sf(
                self.cset[id_corrections[self.id_wp]].evaluate(
                    muon_eta, muon_pt, "systup"
                ),
                in_muon_mask,
                self.muons_counts,
            )
            down_sf = unflat_sf(
                self.cset[id_corrections[self.id_wp]].evaluate(
                    muon_eta, muon_pt, "systdown"
                ),
                in_muon_mask,
                self.muons_counts,
            )
            # add scale factors to weights container
            self.weights.add(
                name=f"CMS_eff_m_id_{self.year_key}",
                weight=nominal_sf,
                weightUp=up_sf,
                weightDown=down_sf,
            )
        else:
            self.weights.add(
                name=f"CMS_eff_m_id_{self.year_key}",
                weight=nominal_sf,
            )

    def add_iso_weight(self):
        """
        add muon Iso (LooseRelIso with mediumID) scale factors to weights container
        """
        iso_corrections = {
            "Run2": {
                "loose": {
                    "loose": "NUM_LooseRelIso_DEN_LooseID",
                    "medium": None,
                    "tight": None,
                },
                "medium": {
                    "loose": "NUM_LooseRelIso_DEN_MediumID",
                    "medium": None,
                    "tight": "NUM_TightRelIso_DEN_MediumID",
                },
                "tight": {
                    "loose": "NUM_LooseRelIso_DEN_TightIDandIPCut",
                    "medium": None,
                    "tight": "NUM_TightRelIso_DEN_TightIDandIPCut",
                },
            },
            "Run3": {
                "loose": {
                    "loose": "NUM_LoosePFIso_DEN_LooseID",
                    "medium": "NUM_LoosePFIso_DEN_MediumID",
                    "tight": "NUM_LoosePFIso_DEN_TightID",
                },
                "medium": {
                    "loose": None,
                    "medium": None,
                    "tight": None,
                },
                "tight": {
                    "loose": None,
                    "medium": "NUM_TightPFIso_DEN_MediumID",
                    "tight": "NUM_TightPFIso_DEN_TightID",
                },
            },
        }
        correction_name = iso_corrections[self.run_key][self.id_wp][self.iso_wp]
        assert correction_name, "No Iso SF's available"

        # get 'in-limits' muons
        muon_pt_mask = self.flat_muons.pt > (29.0 if self.run_key == "Run2" else 15.0)
        muon_eta_mask = np.abs(self.flat_muons.eta) < 2.39
        in_muon_mask = muon_pt_mask & muon_eta_mask
        in_muons = self.flat_muons.mask[in_muon_mask]

        # get muons pT and abseta (replace None values with some 'in-limit' value)
        muon_pt = ak.fill_none(in_muons.pt, 29.0 if self.run_key == "Run2" else 15.0)
        muon_eta = np.abs(ak.fill_none(in_muons.eta, 0.0))

        # get nominal scale factors
        nominal_sf = unflat_sf(
            self.cset[correction_name].evaluate(muon_eta, muon_pt, "nominal"),
            in_muon_mask,
            self.muons_counts,
        )
        if self.variation == "nominal":
            # get 'up' and 'down' scale factors
            up_sf = unflat_sf(
                self.cset[correction_name].evaluate(muon_eta, muon_pt, "systup"),
                in_muon_mask,
                self.muons_counts,
            )
            down_sf = unflat_sf(
                self.cset[correction_name].evaluate(muon_eta, muon_pt, "systdown"),
                in_muon_mask,
                self.muons_counts,
            )
            # add scale factors to weights container
            self.weights.add(
                name=f"CMS_eff_m_iso_{self.year_key}",
                weight=nominal_sf,
                weightUp=up_sf,
                weightDown=down_sf,
            )
        else:
            self.weights.add(
                name=f"CMS_eff_m_iso_{self.year_key}",
                weight=nominal_sf,
            )

    def add_triggeriso_weight(self) -> None:
        """
        add muon Trigger Iso weights for single/double muon events
        """
        assert (
            self.id_wp == "tight" and self.iso_wp == "tight"
        ), "there's only available muon trigger SF for 'tight' ID and Iso"

        kind = "single" if ak.all(ak.num(self.muons) == 1) else "double"

        # get 'in-limits' muons
        muon_pt_mask = self.flat_muons.pt > (29.0 if self.run_key == "Run2" else 26.0)
        if kind == "double":
            upper_limit = 199.99
            if self.year == "2022preEE":
                upper_limit = 499.99
            muon_pt_mask = muon_pt_mask & (self.flat_muons.pt < upper_limit)

        muon_eta_mask = np.abs(self.flat_muons.eta) < 2.399

        in_muon_mask = muon_pt_mask & muon_eta_mask
        in_muons = self.flat_muons.mask[in_muon_mask]

        # get muons transverse momentum and abs pseudorapidity (replace None values with some 'in-limit' value)
        muon_pt = ak.fill_none(in_muons.pt, 29.0 if self.run_key == "Run2" else 26.0)
        muon_eta = np.abs(ak.fill_none(in_muons.eta, 0.0))

        # scale factors keys
        sfs_keys = {
            "2016preVFP": "NUM_IsoMu24_or_IsoTkMu24_DEN_CutBasedIdTight_and_PFIsoTight",
            "2016postVFP": "NUM_IsoMu24_or_IsoTkMu24_DEN_CutBasedIdTight_and_PFIsoTight",
            "2017": "NUM_IsoMu27_DEN_CutBasedIdTight_and_PFIsoTight",
            "2018": "NUM_IsoMu24_DEN_CutBasedIdTight_and_PFIsoTight",
            "2022preEE": "NUM_IsoMu24_DEN_CutBasedIdTight_and_PFIsoTight",
            "2022postEE": "NUM_IsoMu24_DEN_CutBasedIdTight_and_PFIsoTight",
            "2023preBPix": "NUM_IsoMu24_DEN_CutBasedIdTight_and_PFIsoTight",
            "2023postBPix": "NUM_IsoMu24_DEN_CutBasedIdTight_and_PFIsoTight",
        }
        if self.variation == "nominal":
            if kind == "single":
                # for single muon events, compute SF from POG SF
                single_sf = self.cset[sfs_keys[self.year]].evaluate(
                    muon_eta, muon_pt, "nominal"
                )
                nominal_sf = unflat_sf(
                    single_sf,
                    in_muon_mask,
                    self.muons_counts,
                )
            if kind == "double":
                # for double muon events, compute SF from muons' efficiencies
                double_cset = correctionlib.CorrectionSet.from_file(
                    get_muon_hlt_json(year=self.year)
                )
    
                data_eff = double_cset["Muon-HLT-DataEff"].evaluate(
                    self.variation,
                    sfs_keys[self.year],
                    muon_eta,
                    muon_pt,
                )
                data_eff = ak.where(in_muon_mask, data_eff, ak.ones_like(data_eff))
                data_eff = ak.unflatten(data_eff, self.muons_counts)
                data_eff_1 = ak.firsts(data_eff)
                data_eff_2 = ak.pad_none(data_eff, target=2)[:, 1]
                full_data_eff = data_eff_1 + data_eff_2 - data_eff_1 * data_eff_2
    
                mc_eff = double_cset["Muon-HLT-McEff"].evaluate(
                    self.variation,
                    sfs_keys[self.year],
                    muon_eta,
                    muon_pt,
                )
                mc_eff = ak.where(in_muon_mask, mc_eff, ak.ones_like(mc_eff))
                mc_eff = ak.unflatten(mc_eff, self.muons_counts)
                mc_eff_1 = ak.firsts(mc_eff)
                mc_eff_2 = ak.pad_none(mc_eff, target=2)[:, 1]
                full_mc_eff = mc_eff_1 + mc_eff_2 - mc_eff_1 * mc_eff_2
    
                nominal_sf = full_data_eff / full_mc_eff
    
            if self.variation == "nominal":
                # get 'up' and 'down' scale factors
                if kind == "single":
                    up_sf = self.cset[sfs_keys[self.year]].evaluate(
                        muon_eta, muon_pt, "systup"
                    )
                    up_sf = unflat_sf(
                        up_sf,
                        in_muon_mask,
                        self.muons_counts,
                    )
                    down_sf = self.cset[sfs_keys[self.year]].evaluate(
                        muon_eta, muon_pt, "systdown"
                    )
                    down_sf = unflat_sf(
                        down_sf,
                        in_muon_mask,
                        self.muons_counts,
                    )
                    self.weights.add(
                        name=f"CMS_eff_m_trigger_{self.year_key}",
                        weight=nominal_sf,
                        weightUp=up_sf,
                        weightDown=down_sf,
                    )
                elif kind == "double":
                    self.weights.add(
                        name=f"CMS_eff_m_trigger_{self.year_key}",
                        weight=nominal_sf,
                        weightUp=np.ones_like(nominal_sf),
                        weightDown=np.ones_like(nominal_sf),
                    )
            else:
                self.weights.add(
                    name=f"CMS_eff_m_trigger_{self.year_key}",
                    weight=nominal_sf,
                )
