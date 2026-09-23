import re
import json
import glob
import pathlib
import correctionlib
import numpy as np
import awkward as ak
import importlib.resources
from coffea import util
from typing import Type
from coffea.analysis_tools import Weights
from analysis.corrections.utils import get_pog_json
from analysis.working_points.utils import get_btag_mask


class BTagCorrector:
    """
    BTag corrector class.

    https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation

    Parameters:
    -----------
        events:
            Events collection
        weights:
            Weights container from coffea.analysis_tools
        worging_point:
            worging point {loose, medium, tight}
        year:
            dataset year {2016preVFP, 2016postVFP, 2017, 2018, 2022preEE, 2022postEE, 2023preBPix, 2023postBPix}
        workflow:
            workflow name
        category:
            category name
        variation:
            if 'nominal' (default) add 'nominal', 'up' and 'down' variations to weights container. else, add only 'nominal' weights.
        full_run:
            False (default) if only one year is analized,
            True if the fullRunII data is analyzed.
            If False, the 'up' and 'down' systematics are be used.
            If True, 'up/down_correlated' and 'up/down_uncorrelated'
            systematics are used instead of the 'up/down' ones,
            which are supposed to be correlated/decorrelated
            between the different data years
    """

    def __init__(
        self,
        events,
        weights: Type[Weights],
        year: str,
        workflow: str,
        category: str,
        worging_point: str,
        variation: str,
        full_run: bool = False,
    ) -> None:
        self._year = year
        self._wp = worging_point
        self._weights = weights
        self._full_run = full_run
        self._variation = variation

        self._nano_version = "9" if year.startswith("201") else "12"
        self._tagger = "deepJet" if self._nano_version == "9" else "particleNet"

        # check available btag SFs
        self._working_point_map = {"tight": "T", "medium": "M", "loose": "L"}
        if self._wp not in self._working_point_map:
            raise ValueError(
                f"There are no available b-tag SFs for the working point. Please specify {list(self._working_point_map.keys())}"
            )
        # check available btag efficiencies (btag_eff_<tagger>_<wp>_<year>.coffea)
        btag_eff_name_map = {
            "2b1mu": "btag_eff_mu",
            "1b1mu": "btag_eff_mu",
            "2b1e": "btag_eff_ele",
            "1b1e": "btag_eff_ele",
        }
        btag_eff_name = (
            btag_eff_name_map.get(workflow, f"btag_eff_{workflow}")
            + f"_{self._wp}_{category}_{self._year}.coffea"
        )
        btag_eff_file = (
            pathlib.Path().cwd()
            / "analysis"
            / "data"
            / "btag_efficiencies"
            / btag_eff_name
        )
        if not btag_eff_file.exists():
            raise ValueError(f"There is no b-tagging efficiency file '{btag_eff_name}'")

        # load efficiency lookup table (only for deepJet)
        # efflookup(pt, |eta|, flavor)
        with importlib.resources.path(
            "analysis.data.btag_efficiencies", btag_eff_name
        ) as filename:
            self._efflookup = util.load(str(filename))

        # define correction set
        self._cset = correctionlib.CorrectionSet.from_file(
            get_pog_json(json_name="btag", year=year)
        )

        # select bc and light jets
        # hadron flavor definition: 5=b, 4=c, 0=udsg
        self._bc_jets = events.selected_jets[events.selected_jets.hadronFlavour >= 4]
        self._light_jets = events.selected_jets[events.selected_jets.hadronFlavour == 0]
        self._jet_map = {"bc": self._bc_jets, "light": self._light_jets}

        self._jet_pass_btag = {
            "bc": get_btag_mask(self._jet_map["bc"], self._year, self._wp),
            "light": get_btag_mask(self._jet_map["light"], self._year, self._wp),
        }
        self.var_naming_map = {
            "bc": "CMS_btag_heavy",
            "light": "CMS_btag_light",
        }

    def add_btag_weights(self, flavor: str) -> None:
        """
        Add b-tagging weights (nominal, up and down) to weights container for bc or light jets

        Parameters:
        -----------
            flavor:
                hadron flavor {'bc', 'light'}
        """
        # efficiencies
        eff = self.efficiency(flavor=flavor)

        # mask with events that pass the btag working point
        passbtag = self._jet_pass_btag[flavor]

        # nominal scale factors
        btag_sf = self.get_scale_factors(flavor=flavor, syst="central")

        # nominal weights
        btag_weight = self.get_btag_weight(eff, btag_sf, passbtag)

        if self._variation == "nominal":
            # systematics
            if not self._full_run:
                # up and down scale factors
                btag_sf_up = self.get_scale_factors(flavor=flavor, syst="up")
                btag_sf_down = self.get_scale_factors(flavor=flavor, syst="down")
                btag_weight_up = self.get_btag_weight(eff, btag_sf_up, passbtag)
                btag_weight_down = self.get_btag_weight(eff, btag_sf_down, passbtag)
                # add weights to Weights container
                self._weights.add(
                    name=self.var_naming_map[flavor],
                    weight=btag_weight,
                    weightUp=btag_weight_up,
                    weightDown=btag_weight_down,
                )
            else:
                # up and down correlated scale factors
                btag_sf_up_correlated = self.get_scale_factors(
                    flavor=flavor, syst="up_correlated"
                )
                btag_sf_down_correlated = self.get_scale_factors(
                    flavor=flavor, syst="down_correlated"
                )
                btag_weight_up_correlated = self.get_btag_weight(
                    eff, btag_sf_up_correlated, passbtag
                )
                btag_weight_down_correlated = self.get_btag_weight(
                    eff, btag_sf_down_correlated, passbtag
                )
                # up and down uncorrelated scale factors
                btag_sf_up_uncorrelated = self.get_scale_factors(
                    flavor=flavor, syst="up_uncorrelated"
                )
                btag_sf_down_uncorrelated = self.get_scale_factors(
                    flavor=flavor, syst="down_uncorrelated"
                )
                btag_weight_up_uncorrelated = self.get_btag_weight(
                    eff, btag_sf_up_uncorrelated, passbtag
                )
                btag_weight_down_uncorrelated = self.get_btag_weight(
                    eff, btag_sf_down_uncorrelated, passbtag
                )
                # add weights to Weights container
                self._weights.add(
                    name=f"{self.var_naming_map[flavor]}_correlated",
                    weight=btag_weight,
                    weightUp=btag_weight_up_correlated,
                    weightDown=btag_weight_down_correlated,
                )
                self._weights.add(
                    name=f"{self.var_naming_map[flavor]}_uncorrelated_{self._year[:4]}",
                    weight=ak.ones_like(btag_weight),
                    weightUp=btag_weight_up_uncorrelated,
                    weightDown=btag_weight_down_uncorrelated,
                )
        else:
            self._weights.add(
                name=self.var_naming_map[flavor],
                weight=btag_weight,
            )

    def efficiency(self, flavor: str, fill_value=1) -> ak.Array:
        """compute the btagging efficiency for 'njets' jets"""
        return self._efflookup(
            self._jet_map[flavor].pt,
            np.abs(self._jet_map[flavor].eta),
            self._jet_map[flavor].hadronFlavour,
        )

    def get_scale_factors(self, flavor: str, syst="central", fill_value=1) -> ak.Array:
        """
        compute jets scale factors
        """
        return self.get_sf(flavor=flavor, syst=syst)

    def get_sf(self, flavor: str, syst: str = "central") -> ak.Array:
        """
        compute the scale factors for bc or light jets

        Parameters:
        -----------
            flavor:
                hadron flavor {'bc', 'light'}
            syst:
                Name of the systematic {'central', 'down', 'down_correlated', 'down_uncorrelated', 'up', 'up_correlated'}
        """
        cset_keys = {
            "bc": f"{self._tagger}_comb",
            "light": (
                f"{self._tagger}_incl"
                if self._nano_version == "9"
                else f"{self._tagger}_light"
            ),
        }
        # until correctionlib handles jagged data natively we have to flatten and unflatten
        j, nj = ak.flatten(self._jet_map[flavor]), ak.num(self._jet_map[flavor])

        # get 'in-limits' jets
        jet_eta_mask = np.abs(j.eta) < 2.499
        in_jet_mask = jet_eta_mask
        in_jets = j.mask[in_jet_mask]

        # get jet transverse momentum, abs pseudorapidity and hadron flavour (replace None values with some 'in-limit' value)
        jets_pt = ak.fill_none(in_jets.pt, 0.0)
        jets_eta = ak.fill_none(np.abs(in_jets.eta), 0.0)
        jets_hadron_flavour = ak.fill_none(
            in_jets.hadronFlavour, 5 if flavor == "bc" else 0
        )

        sf = self._cset[cset_keys[flavor]].evaluate(
            syst,
            self._working_point_map[self._wp],
            np.array(jets_hadron_flavour),
            np.array(jets_eta),
            np.array(jets_pt),
        )
        sf = ak.where(in_jet_mask, sf, ak.ones_like(sf))
        return ak.unflatten(sf, nj)

    @staticmethod
    def get_btag_weight(eff: ak.Array, sf: ak.Array, passbtag: ak.Array) -> ak.Array:
        """
        compute b-tagging weights

        see: https://twiki.cern.ch/twiki/bin/viewauth/CMS/BTagSFMethods

        Parameters:
        -----------
            eff:
                btagging efficiencies
            sf:
                jets scale factors
            passbtag:
                mask with jets that pass the b-tagging working point
        """
        # tagged SF = SF * eff / eff = SF
        tagged_sf = ak.prod(sf.mask[passbtag], axis=-1)

        # untagged SF = (1 - SF * eff) / (1 - eff)
        # untagged_sf = ak.prod(((1 - sf * eff) / (1 - eff)).mask[~passbtag], axis=-1)
        ratio = (1 - sf * eff) / (1 - eff)
        ratio_untagged = ak.where(~passbtag, ratio, 1.0)
        untagged_sf = ak.prod(ratio_untagged, axis=-1)

        return ak.fill_none(tagged_sf * untagged_sf, 1.0)
