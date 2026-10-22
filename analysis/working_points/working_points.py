import numpy as np
import awkward as ak
from analysis.working_points.utils import get_btag_mask


class WorkingPoints:
    # -------------------------------------------
    # Muons
    # -------------------------------------------
    def muons_id(self, events, wp):
        wps = {
            "highpt": events.Muon.highPtId == 2,
            # cutbased ID working points
            "loose": events.Muon.looseId,
            "medium": events.Muon.mediumId,
            "tight": events.Muon.tightId,
        }
        if wp not in wps:
            raise ValueError(
                f"Invalid value for muon ID working point. Please specify {list(wps.keys())}"
            )
        return wps[wp]

    def muons_iso(self, events, wp):
        wps = {
            "loose": (
                events.Muon.pfRelIso04_all < 0.25
                if hasattr(events.Muon, "pfRelIso04_all")
                else events.Muon.pfRelIso03_all < 0.25
            ),
            "medium": (
                events.Muon.pfRelIso04_all < 0.20
                if hasattr(events.Muon, "pfRelIso04_all")
                else events.Muon.pfRelIso03_all < 0.20
            ),
            "tight": (
                events.Muon.pfRelIso04_all < 0.15
                if hasattr(events.Muon, "pfRelIso04_all")
                else events.Muon.pfRelIso03_all < 0.15
            ),
        }
        if wp not in wps:
            raise ValueError(
                f"Invalid value {wp} for muon ISO working point. Please specify {list(wps.keys())}"
            )
        return wps[wp]

    # -------------------------------------------
    # Electrons
    # -------------------------------------------
    def electrons_id(self, events, wp):
        wps = {
            # mva ID working points https://twiki.cern.ch/twiki/bin/view/CMS/MultivariateElectronIdentificationRun2
            "wp80iso": (
                events.Electron.mvaFall17V2Iso_WP80
                if hasattr(events.Electron, "mvaFall17V2Iso_WP80")
                else events.Electron.mvaIso_WP80
            ),
            "wp90iso": (
                events.Electron.mvaFall17V2Iso_WP90
                if hasattr(events.Electron, "mvaFall17V2Iso_WP90")
                else events.Electron.mvaIso_WP90
            ),
            # cutbased ID working points https://twiki.cern.ch/twiki/bin/view/CMS/CutBasedElectronIdentificationRun2
            "fail": events.Electron.cutBased == 0,
            "veto": events.Electron.cutBased == 1,
            "loose": events.Electron.cutBased == 2,
            "medium": events.Electron.cutBased == 3,
            "tight": events.Electron.cutBased == 4,
        }
        if hasattr(events.Electron, "mvaFall17V2Iso_WPL"):
            wps["wpLiso"] = events.Electron.mvaFall17V2Iso_WPL
        if wp not in wps:
            raise ValueError(
                f"Invalid value {wp} for electron ID working point. Please specify {list(wps.keys())}"
            )
        return wps[wp]

    def electrons_iso(self, events, wp):
        wps = {
            # https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideMuonSelection
            "loose": (
                events.Electron.pfRelIso04_all < 0.25
                if hasattr(events.Electron, "pfRelIso04_all")
                else events.Electron.pfRelIso03_all < 0.25
            ),
            "medium": (
                events.Electron.pfRelIso04_all < 0.20
                if hasattr(events.Electron, "pfRelIso04_all")
                else events.Electron.pfRelIso03_all < 0.20
            ),
            "tight": (
                events.Electron.pfRelIso04_all < 0.15
                if hasattr(events.Electron, "pfRelIso04_all")
                else events.Electron.pfRelIso03_all < 0.15
            ),
        }
        if wp not in wps:
            raise ValueError(
                f"Invalid value {wp} for electron ISO working point. Please specify {list(wps.keys())}"
            )
        return wps[wp]

    # -------------------------------------------
    # Taus
    # -------------------------------------------
    def taus_vs_jet(self, events, wp):
        tau_vs_jet_branch = (
            events.Tau.idDeepTau2017v2p1VSjet
            if hasattr(events.Tau, "idDeepTau2017v2p1VSjet")
            else events.Tau.idDeepTau2018v2p5VSjet
        )
        wps = {
            "vvvloose": tau_vs_jet_branch > 1,
            "vvloose": tau_vs_jet_branch > 2,
            "vloose": tau_vs_jet_branch > 4,
            "loose": tau_vs_jet_branch > 8,
            "medium": tau_vs_jet_branch > 16,
            "tight": tau_vs_jet_branch > 32,
            "vtight": tau_vs_jet_branch > 64,
            "vvtight": tau_vs_jet_branch > 128,
        }
        if wp not in wps:
            raise ValueError(
                f"Invalid value {wp} for DeepTauvsJet working point. Please specify {list(wps.keys())}"
            )
        return wps[wp]

    def taus_vs_ele(self, events, wp):
        tau_vs_ele_branch = (
            events.Tau.idDeepTau2017v2p1VSe
            if hasattr(events.Tau, "idDeepTau2017v2p1VSe")
            else events.Tau.idDeepTau2018v2p5VSe
        )
        wps = {
            "vvvloose": tau_vs_ele_branch > 1,
            "vvloose": tau_vs_ele_branch > 2,
            "vloose": tau_vs_ele_branch > 4,
            "loose": tau_vs_ele_branch > 8,
            "medium": tau_vs_ele_branch > 16,
            "tight": tau_vs_ele_branch > 32,
            "vtight": tau_vs_ele_branch > 64,
            "vvtight": tau_vs_ele_branch > 128,
        }
        if wp not in wps:
            raise ValueError(
                f"Invalid value {wp} for DeepTauvsElectron working point. Please specify {list(wps.keys())}"
            )
        return wps[wp]

    def taus_vs_mu(self, events, wp):
        tau_vs_mu_branch = (
            events.Tau.idDeepTau2017v2p1VSmu
            if hasattr(events.Tau, "idDeepTau2017v2p1VSmu")
            else events.Tau.idDeepTau2018v2p5VSmu
        )
        wps = {
            "vloose": tau_vs_mu_branch > 1,
            "loose": tau_vs_mu_branch > 2,
            "medium": tau_vs_mu_branch > 4,
            "tight": tau_vs_mu_branch > 8,
        }
        if wp not in wps:
            raise ValueError(
                f"Invalid value {wp} for DeepTauvsMuon working point. Please specify {list(wps.keys())}"
            )
        return wps[wp]

    def taus_decaymode(self, events, wp):
        # prong to mode map
        wps = {
            "1": [0, 1, 2],
            "2": [5, 6, 7],
            "3": [10, 11],
            "13": [0, 1, 2, 10, 11],
            "12": [0, 1, 2, 5, 6, 7],
            "23": [5, 6, 7, 10, 11],
        }
        if wp not in wps:
            raise ValueError(
                f"Invalid value {wp} for tau prong working point. Please specify {list(wps.keys())}"
            )
        tau_dm = events.Tau.decayMode
        decay_mode_mask = ak.zeros_like(tau_dm)
        for mode in wps[wp]:
            decay_mode_mask = np.logical_or(decay_mode_mask, tau_dm == mode)
        return decay_mode_mask

    # -------------------------------------------
    # Jets
    # -------------------------------------------
    def jets_id(self, events, year, wp):
        # Run 3 NanoAODs have a bug in jetId
        # Implement fix from:
        # https://twiki.cern.ch/twiki/bin/viewauth/CMS/JetID13p6TeV#nanoAOD_Flags
        if year == "2024":
            # NanoV15
            barrel = (
                (events.Jet.neHEF < 0.99)
                & (events.Jet.neEmEF < 0.9)
                & (events.Jet.chMultiplicity + events.Jet.neMultiplicity > 1)
                & (events.Jet.chHEF > 0.01)
                & (events.Jet.chMultiplicity > 0)
            )
            t1 = (events.Jet.neHEF < 0.9) & (events.Jet.neEmEF < 0.99)
            t2 = events.Jet.neHEF < 0.99
            endcap = (events.Jet.neMultiplicity >= 2) & (events.Jet.neEmEF < 0.4)

            jetid_tight = ak.where(
                abs(events.Jet.eta) <= 2.6,
                barrel,
                ak.where(
                    (abs(events.Jet.eta) > 2.6) & (abs(events.Jet.eta) <= 2.7),
                    t1,
                    ak.where(
                        (abs(events.Jet.eta) > 2.7) & (abs(events.Jet.eta) <= 3.0),
                        t2,
                        ak.where(
                            (abs(events.Jet.eta) > 3.0),
                            endcap,
                            ak.zeros_like(events.Jet.pt, dtype=bool),
                        ),
                    ),
                ),
            )
            jetid_tightlepveto = ak.where(
                np.abs(events.Jet.eta) <= 2.7,
                jetid_tight & (events.Jet.muEF < 0.8) & (events.Jet.chEmEF < 0.8),
                jetid_tight,
            )
        elif year.startswith("202"):
            # NanoV12
            jetid_tight = ak.where(
                np.abs(events.Jet.eta) <= 2.7,
                (events.Jet.jetId >= 2)
                & (events.Jet.muEF < 0.8)
                & (events.Jet.chEmEF < 0.8),
                ak.where(
                    (abs(events.Jet.eta) > 2.7) & (abs(events.Jet.eta) <= 3.0),
                    (events.Jet.jetId >= 2) & (events.Jet.neHEF < 0.99),
                    ak.where(
                        (abs(events.Jet.eta) > 3.0),
                        (events.Jet.jetId & (1 << 1)) & (events.Jet.neEmEF < 0.4),
                        ak.zeros_like(events.Jet.pt, dtype=bool),
                    ),
                ),
            )
            jetid_tightlepveto = ak.where(
                np.abs(events.Jet.eta) <= 2.7,
                jetid_tight & (events.Jet.muEF < 0.8) & (events.Jet.chEmEF < 0.8),
                jetid_tight,
            )
        else:
            # NanoV9
            jetid_tight = events.Jet.jetId == 2
            jetid_tightlepveto = events.Jet.jetId == 6

        wps = {
            "tight": ak.values_astype(jetid_tight, bool),
            "tightlepveto": ak.values_astype(jetid_tightlepveto, bool),
        }
        if wp not in wps:
            raise ValueError(
                f"Invalid value {wp} for jet ID working point. Please specify {list(wps.keys())}"
            )
        return wps[wp]

    def jets_pileup_id(self, events, wp, year):
        if year.startswith("201"):
            # Run2
            wps = {
                "2016preVFP": {
                    "loose": events.Jet.puId == 1,
                    "medium": events.Jet.puId == 3,
                    "tight": events.Jet.puId == 7,
                },
                "2016postVFP": {
                    "loose": events.Jet.puId == 1,
                    "medium": events.Jet.puId == 3,
                    "tight": events.Jet.puId == 7,
                },
                "2017": {
                    "loose": events.Jet.puId == 4,
                    "medium": events.Jet.puId == 6,
                    "tight": events.Jet.puId == 7,
                },
                "2018": {
                    "loose": events.Jet.puId == 4,
                    "medium": events.Jet.puId == 6,
                    "tight": events.Jet.puId == 7,
                },
            }
            if wp not in wps[year]:
                raise ValueError(
                    f"Invalid value {wp} for jet pileup ID working point. Please specify {list(wps[year].keys())}"
                )
            # break up selection for low and high pT jets
            # to apply jets_pileup only to jets with pT < 50 GeV
            return ak.where(
                events.Jet.pt < 50,
                wps[year][wp],
                events.Jet.pt >= 50,
            )
        else:
            # Run3
            return np.ones_like(events.Jet.pt, dtype=bool)

    def jets_btagging(self, events, wp, year):
        return get_btag_mask(events.Jet, year, wp)
