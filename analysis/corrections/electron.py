import json
import copy
import correctionlib
import numpy as np
import awkward as ak
import importlib.resources
from typing import Type
from pathlib import Path
from .utils import unflat_sf
from coffea.analysis_tools import Weights
from analysis.corrections.utils import pog_years, get_pog_json, get_electron_hlt_json


class ElectronCorrector:
    """
    Electron corrector class

    Parameters:
    -----------
    events:
        events collection
    weights:
        Weights object from coffea.analysis_tools
    year:
        Year of the dataset {2016preVFP, 2016postVFP, 2017, 2018, 2022preEE, 2022postEE, 2023preBPix, 2023postBPix}
    variation:
        if 'nominal' (default) add 'nominal', 'up' and 'down'
        variations to weights container. else, add only 'nominal' weights.
    """

    def __init__(
        self,
        events: ak.Array,
        weights: Type[Weights],
        year: str = "2017",
        variation: str = "nominal",
    ) -> None:
        self.electrons = events.selected_electrons
        self.variation = variation

        # flat electrons array
        self.flat_electrons, self.electrons_counts = ak.flatten(
            events.selected_electrons
        ), ak.num(events.selected_electrons)
        # weights container
        self.weights = weights

        # define correction set
        self.cset = correctionlib.CorrectionSet.from_file(
            get_pog_json(json_name="electron", year=year)
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

        # set id working points map
        self.id_map = {
            "wp80iso": "wp80iso",
            "wp90iso": "wp90iso",
            "wp80noiso": "wp80noiso",
            "wp90noiso": "wp90noiso",
            "loose": "Loose",
            "medium": "Medium",
            "tight": "Tight",
            "veto": "Veto",
        }

        self.year_map = {
            "2016preVFP": "2016preVFP",
            "2016postVFP": "2016postVFP",
            "2017": "2017",
            "2018": "2018",
            "2022postEE": "2022Re-recoE+PromptFG",
            "2022preEE": "2022Re-recoBCD",
            "2023preBPix": "2023PromptC",
            "2023postBPix": "2023PromptD",
        }

    def add_id_weight(self, id_working_point: str) -> None:
        """
        add electron identification scale factors to weights container

        Parameters:
        -----------
            id_working_point:
                Working point {'Loose', 'Medium', 'Tight', 'wp80iso', 'wp80noiso', 'wp90iso', 'wp90noiso'}
        """
        # get 'in-limits' electrons
        electron_pt_mask = (self.flat_electrons.pt > 10.0) & (
            self.flat_electrons.pt < 499.999
        )  # potential problems with pt > 500 GeV
        in_electrons_mask = electron_pt_mask
        in_electrons = self.flat_electrons.mask[in_electrons_mask]

        # get electrons transverse momentum and pseudorapidity (replace None values with some 'in-limit' value)
        electron_pt = ak.fill_none(in_electrons.pt, 10.0)
        electron_eta = ak.fill_none(in_electrons.eta + in_electrons.deltaEtaSC, 0.0)
        electron_phi = ak.fill_none(in_electrons.phi, 0)

        # get nominal scale factors
        cset_args = [
            self.year_map[self.year],
            "sf",
            self.id_map[id_working_point],
            electron_eta,
            electron_pt,
        ]
        if self.year.startswith("2023"):
            cset_args += [electron_phi]

        id_key = "UL-Electron-ID-SF" if self.run_key == "Run2" else "Electron-ID-SF"
        nominal_sf = unflat_sf(
            self.cset[id_key].evaluate(*cset_args),
            in_electrons_mask,
            self.electrons_counts,
        )
        if self.variation == "nominal":
            # get 'up' and 'down' scale factors
            cset_args_up = [
                self.year_map[self.year],
                "sfup",
                self.id_map[id_working_point],
                electron_eta,
                electron_pt,
            ]
            cset_args_down = [
                self.year_map[self.year],
                "sfdown",
                self.id_map[id_working_point],
                electron_eta,
                electron_pt,
            ]
            if self.year.startswith("2023"):
                cset_args_up += [electron_phi]
                cset_args_down += [electron_phi]

            up_sf = unflat_sf(
                self.cset[id_key].evaluate(*cset_args_up),
                in_electrons_mask,
                self.electrons_counts,
            )
            down_sf = unflat_sf(
                self.cset[id_key].evaluate(*cset_args_down),
                in_electrons_mask,
                self.electrons_counts,
            )
            # add scale factors to weights container
            self.weights.add(
                name=f"CMS_eff_e_id_{self.year_key}",
                weight=nominal_sf,
                weightUp=up_sf,
                weightDown=down_sf,
            )
        else:
            self.weights.add(
                name=f"CMS_eff_e_id_{self.year_key}",
                weight=nominal_sf,
            )

    def add_reco_weight(self, reco: str) -> None:
        """
        add electron reconstruction scale factors to weights container

        reco: {RecoAbove20, RecoBelow20, Reco20to75, RecoAbove75}
        """
        electron_pt_mask = {
            "Run2": {
                "RecoAbove20": (self.flat_electrons.pt > 20)
                & (self.flat_electrons.pt < 499.999),
                "RecoBelow20": (self.flat_electrons.pt > 10)
                & (self.flat_electrons.pt < 20),
            },
            "Run3": {
                "RecoBelow20": (self.flat_electrons.pt > 10.0)
                & (self.flat_electrons.pt < 20.0),
                "Reco20to75": (self.flat_electrons.pt > 20.0)
                & (self.flat_electrons.pt < 75.0),
                "RecoAbove75": self.flat_electrons.pt > 75,
            },
        }
        var_naming_map = {
            "RecoBelow20": f"CMS_eff_e_reco_below20_{self.year_key}",
            "RecoAbove20": f"CMS_eff_e_reco_above20_{self.year_key}",
            "Reco20to75": f"CMS_eff_e_reco_20to75_{self.year_key}",
            "RecoAbove75": f"CMS_eff_e_reco_above75_{self.year_key}",
        }
        # get 'in-limits' electrons
        in_electrons_mask = electron_pt_mask[self.run_key][reco]
        in_electrons = self.flat_electrons.mask[in_electrons_mask]

        # get electrons transverse momentum and pseudorapidity (replace None values with some 'in-limit' value)
        electrons_pt_limits = {
            "RecoAbove20": 21,
            "RecoBelow20": 15,
            "Reco20to75": 30,
            "RecoAbove75": 80,
        }
        electron_pt = ak.fill_none(in_electrons.pt, electrons_pt_limits[reco])
        electron_eta = ak.fill_none(in_electrons.eta + in_electrons.deltaEtaSC, 0.0)
        electron_phi = ak.fill_none(in_electrons.phi, 0.0)

        # get nominal scale factors
        cset_args = [
            self.year_map[self.year],
            "sf",
            reco,
            electron_eta,
            electron_pt,
        ]
        if self.year.startswith("2023"):
            cset_args += [electron_phi]

        id_key = "UL-Electron-ID-SF" if self.run_key == "Run2" else "Electron-ID-SF"
        nominal_sf = unflat_sf(
            self.cset[id_key].evaluate(*cset_args),
            in_electrons_mask,
            self.electrons_counts,
        )
        if self.variation == "nominal":
            # get 'up' and 'down' scale factors
            cset_args_up = [
                self.year_map[self.year],
                "sfup",
                reco,
                electron_eta,
                electron_pt,
            ]
            cset_args_down = [
                self.year_map[self.year],
                "sfdown",
                reco,
                electron_eta,
                electron_pt,
            ]
            if self.year.startswith("2023"):
                cset_args_up += [electron_phi]
                cset_args_down += [electron_phi]

            up_sf = unflat_sf(
                self.cset[id_key].evaluate(*cset_args_up),
                in_electrons_mask,
                self.electrons_counts,
            )
            down_sf = unflat_sf(
                self.cset[id_key].evaluate(*cset_args_down),
                in_electrons_mask,
                self.electrons_counts,
            )
            # add scale factors to weights container
            self.weights.add(
                name=var_naming_map[reco],
                weight=nominal_sf,
                weightUp=up_sf,
                weightDown=down_sf,
            )
        else:
            self.weights.add(
                name=var_naming_map[reco],
                weight=nominal_sf,
            )

    def add_hlt_weights(self, id_wp):
        """
        Compute electron HLT weights
        """
        # get electron correction set
        assert (
            id_wp == "wp80iso"
        ), "there's only available muon trigger SF for 'wp80iso' ID"

        run2_hlt_paths = {
            "2016preVFP": "Ele27_WPTight_Gsf_OR_Photon175",
            "2016postVFP": "Ele27_WPTight_Gsf_OR_Photon175",
            "2017": "Ele35_WPTight_Gsf_OR_Photon200",
            "2018": "Ele32_WPTight_Gsf",
        }
        hlt_path_id_map = {
            "wp80iso": "HLT_SF_Ele30_MVAiso80ID",
            "wp90iso": "HLT_SF_Ele30_MVAiso90ID",
        }
        # get electrons within SF binning
        if self.run_key == "Run3":
            in_electrons_mask = self.flat_electrons.pt > 25.0
            in_electrons = self.flat_electrons.mask[in_electrons_mask]
            # get electrons pT and abseta (replace None values with some 'in-limit' value)
            electron_pt = ak.fill_none(in_electrons.pt, 25.0)
            electron_eta = ak.fill_none(in_electrons.eta, 0)

        elif self.run_key == "Run2":
            pt_limits = {
                "2016preVFP": 27.0,
                "2016postVFP": 27.0,
                "2017": 35.0,
                "2018": 32.0,
            }
            electron_pt_mask = (self.flat_electrons.pt > pt_limits[self.year]) & (
                self.flat_electrons.pt < 499.9
            )
            eta_mask = (
                np.abs(self.flat_electrons.eta + self.flat_electrons.deltaEtaSC) < 2.5
            )
            in_electrons_mask = electron_pt_mask & eta_mask
            in_electrons = self.flat_electrons.mask[in_electrons_mask]

            # get electrons pT and abseta (replace None values with some 'in-limit' value)
            electron_pt = ak.fill_none(in_electrons.pt, 40)
            electron_eta = ak.fill_none(
                self.flat_electrons.eta + self.flat_electrons.deltaEtaSC, 0
            )

        # check whether there are single or double electron events
        kind = "single" if ak.all(ak.num(self.electrons) == 1) else "double"
        if kind == "single":

            if self.run_key == "Run3":
                cset = correctionlib.CorrectionSet.from_file(
                    get_pog_json(json_name="electron_hlt", year=self.year)
                )
                sf = cset["Electron-HLT-SF"].evaluate(
                    self.year_map[self.year],
                    "sf",
                    hlt_path_id_map[id_wp],
                    electron_eta,
                    electron_pt,
                )
                sf = ak.where(in_electrons_mask, sf, ak.ones_like(sf))
                sf = ak.fill_none(ak.unflatten(sf, self.electrons_counts), value=1)
                nominal_sf = ak.firsts(nominal_sf)

            elif self.run_key == "Run2":
                cset = correctionlib.CorrectionSet.from_file(
                    get_electron_hlt_json("SF", self.year)
                )
                sf = cset[f"HLT_SF_{run2_hlt_paths[self.year]}_MVAiso80ID"].evaluate(
                    electron_eta, electron_pt
                )
                sf = ak.where(in_electrons_mask, sf, ak.ones_like(sf))
                sf = ak.fill_none(ak.unflatten(sf, self.electrons_counts), value=1)
                nominal_sf = ak.firsts(sf)

        elif kind == "double":

            if self.run_key == "Run3":
                # for double electron events, compute SF from electrons' efficiencies
                cset = correctionlib.CorrectionSet.from_file(
                    get_pog_json(json_name="electron_hlt", year=self.year)
                )
                data_eff = cset["Electron-HLT-DataEff"].evaluate(
                    self.year_map[self.year],
                    "nom",
                    hlt_path_id_map[id_wp],
                    electron_eta,
                    electron_pt,
                )
                data_eff = ak.where(in_electrons_mask, data_eff, ak.ones_like(data_eff))
                data_eff = ak.unflatten(data_eff, self.electrons_counts)
                data_eff_leading = ak.firsts(data_eff)
                data_eff_subleading = ak.pad_none(data_eff, target=2)[:, 1]
                full_data_eff = (
                    data_eff_leading
                    + data_eff_subleading
                    - data_eff_leading * data_eff_subleading
                )
                full_data_eff = ak.fill_none(full_data_eff, 1)

                mc_eff = cset["Electron-HLT-McEff"].evaluate(
                    self.year_map[self.year],
                    "nom",
                    hlt_path_id_map[id_wp],
                    electron_eta,
                    electron_pt,
                )
                mc_eff = ak.where(in_electrons_mask, mc_eff, ak.ones_like(mc_eff))
                mc_eff = ak.unflatten(mc_eff, self.electrons_counts)
                mc_eff_leading = ak.firsts(mc_eff)
                mc_eff_subleading = ak.pad_none(mc_eff, target=2)[:, 1]
                full_mc_eff = (
                    mc_eff_leading
                    + mc_eff_subleading
                    - mc_eff_leading * mc_eff_subleading
                )
                full_mc_eff = ak.fill_none(full_mc_eff, 1)

                nominal_sf = full_data_eff / full_mc_eff

            elif self.run_key == "Run2":
                cset = correctionlib.CorrectionSet.from_file(
                    get_electron_hlt_json("DataEff", self.year)
                )
                data_eff = cset[
                    f"HLT_DataEff_{run2_hlt_paths[self.year]}_MVAiso80ID"
                ].evaluate(electron_eta, electron_pt)
                data_eff = ak.where(in_electrons_mask, data_eff, ak.ones_like(data_eff))
                data_eff = ak.unflatten(data_eff, self.electrons_counts)
                data_eff_leading = ak.firsts(data_eff)
                data_eff_subleading = ak.pad_none(data_eff, target=2)[:, 1]
                full_data_eff = (
                    data_eff_leading
                    + data_eff_subleading
                    - data_eff_leading * data_eff_subleading
                )
                full_data_eff = ak.fill_none(full_data_eff, 1)

                cset = correctionlib.CorrectionSet.from_file(
                    get_electron_hlt_json("MCEff", self.year)
                )
                mc_eff = cset[
                    f"HLT_MCEff_{run2_hlt_paths[self.year]}_MVAiso80ID"
                ].evaluate(electron_eta, electron_pt)
                mc_eff = ak.where(in_electrons_mask, mc_eff, ak.ones_like(mc_eff))
                mc_eff = ak.unflatten(mc_eff, self.electrons_counts)
                mc_eff_leading = ak.firsts(mc_eff)
                mc_eff_subleading = ak.pad_none(mc_eff, target=2)[:, 1]
                full_mc_eff = (
                    mc_eff_leading
                    + mc_eff_subleading
                    - mc_eff_leading * mc_eff_subleading
                )
                full_mc_eff = ak.fill_none(full_mc_eff, 1)

                nominal_sf = full_data_eff / full_mc_eff

        if self.variation == "nominal":
            self.weights.add(
                name=f"CMS_eff_e_trigger_{self.year_key}",
                weight=nominal_sf,
            )
        else:
            self.weights.add(
                name=f"CMS_eff_e_trigger_{self.year_key}",
                weight=nominal_sf,
            )
