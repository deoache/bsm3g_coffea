import numpy as np
from coffea.analysis_tools import Weights
from analysis.corrections import (
    TauCorrector,
    BTagCorrector,
    MuonCorrector,
    ElectronCorrector,
    MuonHighPtCorrector,
    add_isr_weight,
    add_lhepdf_weight,
    add_pileup_weight,
    add_pujetid_weight,
    add_scalevar_weight,
    apply_jet_corrections,
    apply_jerc_corrections,
    add_l1prefiring_weight,
    add_partonshower_weight,
    add_muon_boost_weight,
    add_electron_boost_weight,
    apply_met_phi_corrections,
    apply_electron_ss_corrections,
    apply_rochester_corrections_run2,
    apply_rochester_corrections_run3,
    apply_tau_energy_scale_corrections,
)


def object_corrector_manager(events, year, run, dataset, workflow_config):
    """apply object level corrections"""
    
    objcorr_config = workflow_config.corrections_config["objects"]
    
    if "jets" in objcorr_config:
        if run == "2":
            apply_jet_corrections(events, year)
        elif run == "3":
            apply_jec = True
            apply_jer = False
            apply_junc = False
            if hasattr(events, "genWeight"):
                apply_jer = True
            apply_jerc_corrections(
                events,
                year=year,
                dataset=dataset,
                apply_jec=apply_jec,
                apply_jer=apply_jer,
                apply_junc=apply_junc,
            )
    if "muons" in objcorr_config:
        # apply rochester corretions to muons
        if run == "2":
            apply_rochester_corrections_run2(events, year)
        elif run == "3":
            apply_rochester_corrections_run3(events, year)
    if "electrons" in objcorr_config:
        if run == "3":
            apply_electron_ss_corrections(
                events=events,
                year=year,
            )
    if "taus" in objcorr_config:
        if hasattr(events, "genWeight"):
            if run == "2":
                # apply energy corrections to taus (only to MC)
                apply_tau_energy_scale_corrections(events, year)
    if "met" in objcorr_config:
        # apply MET phi modulation corrections
        apply_met_phi_corrections(events, year)


def weight_manager(pruned_ev, year, run, workflow_config, variation, dataset):
    """apply event level corrections (weights)"""
    nevents = len(pruned_ev)
    # get weights config info
    weights_config = workflow_config.corrections_config["event_weights"]
    # initialize weights container
    weights_container = Weights(len(pruned_ev), storeIndividual=True)
    # add weights
    if hasattr(pruned_ev, "genWeight"):
        if "genWeight" in weights_config:
            if weights_config["genWeight"]:
                weights_container.add("genweight", pruned_ev.genWeight)

        if "l1prefiringWeight" in weights_config:
            if weights_config["l1prefiringWeight"]:
                add_l1prefiring_weight(pruned_ev, weights_container, year, variation)

        if "pileupWeight" in weights_config:
            if weights_config["pileupWeight"]:
                add_pileup_weight(pruned_ev, weights_container, year, variation)

        if "partonshowerWeight" in weights_config:
            if weights_config["partonshowerWeight"]:
                if "PSWeight" in pruned_ev.fields:
                    add_partonshower_weight(
                        events=pruned_ev,
                        weights_container=weights_container,
                        variation=variation,
                    )
        if "lhepdfWeight" in weights_config:
            if weights_config["lhepdfWeight"]:
                if "LHEPdfWeight" in pruned_ev.fields:
                    add_lhepdf_weight(
                        events=pruned_ev,
                        weights_container=weights_container,
                        variation=variation,
                    )

        if "lhescaleWeight" in weights_config:
            if weights_config["lhescaleWeight"]:
                if "LHEScaleWeight" in pruned_ev.fields:
                    add_scalevar_weight(
                        events=pruned_ev,
                        weights_container=weights_container,
                        variation=variation,
                    )

        if "pujetid" in weights_config:
            if weights_config["pujetid"]:
                if run == "2":
                    add_pujetid_weight(
                        events=pruned_ev,
                        weights=weights_container,
                        year=year,
                        working_point=weights_config["pujetid"]["id"],
                        variation=variation,
                    )
        if "btagging" in weights_config:
            if weights_config["btagging"]:
                btag_corrector = BTagCorrector(
                    events=pruned_ev,
                    weights=weights_container,
                    sf_type="comb",
                    worging_point=weights_config["btagging"]["id"],
                    year=year,
                    full_run=weights_config["btagging"]["full_run"],
                    variation=variation,
                )
                if weights_config["btagging"]["bc"]:
                    btag_corrector.add_btag_weights(flavor="bc")
                if weights_config["btagging"]["light"]:
                    btag_corrector.add_btag_weights(flavor="light")

        if "EleBoostWeight" in weights_config:
            if weights_config["EleBoostWeight"]:
                add_electron_boost_weight(
                    events=pruned_ev,
                    weights=weights_container,
                    year=year,
                    variation=variation,
                    dataset=dataset,
                )

        if "MuBoostWeight" in weights_config:
            if weights_config["MuBoostWeight"]:
                add_muon_boost_weight(
                    events=pruned_ev,
                    weights=weights_container,
                    year=year,
                    variation=variation,
                    dataset=dataset,
                )

        if "ISRWeight2D" in weights_config:
            if weights_config["ISRWeight2D"]:
                add_isr_weight(
                    events=pruned_ev,
                    weights=weights_container,
                    year=year,
                    variation=variation,
                    dataset=dataset,
                    fit=False,
                    one_dim=False
                )
        if "ISRWeightFit2D" in weights_config:
            if weights_config["ISRWeightFit2D"]:
                add_isr_weight(
                    events=pruned_ev,
                    weights=weights_container,
                    year=year,
                    variation=variation,
                    dataset=dataset,
                    fit=True,
                    one_dim=False
                )
        if "ISRWeight1D" in weights_config:
            if weights_config["ISRWeight1D"]:
                add_isr_weight(
                    events=pruned_ev,
                    weights=weights_container,
                    year=year,
                    variation=variation,
                    dataset=dataset,
                    fit=False,
                    one_dim=True
                )

        if "electron" in weights_config:
            if weights_config["electron"]:
                if "selected_electrons" in pruned_ev.fields:
                    electron_corrector = ElectronCorrector(
                        events=pruned_ev,
                        weights=weights_container,
                        year=year,
                        variation=variation,
                    )
                    if "id" in weights_config["electron"]:
                        if weights_config["electron"]["id"]:
                            electron_corrector.add_id_weight(
                                id_working_point=weights_config["electron"]["id"]
                            )
                    if "reco" in weights_config["electron"]:
                        if weights_config["electron"]["reco"]:
                            if run == "2":
                                electron_corrector.add_reco_weight("RecoAbove20")
                                electron_corrector.add_reco_weight("RecoBelow20")
                            elif run == "3":
                                electron_corrector.add_reco_weight("RecoBelow20")
                                electron_corrector.add_reco_weight("Reco20to75")
                                electron_corrector.add_reco_weight("RecoAbove75")
                    if "trigger" in weights_config["electron"]:
                        if weights_config["electron"]["trigger"]:
                            electron_corrector.add_hlt_weights(
                                id_wp=weights_config["electron"]["id"],
                            )

        if "muon" in weights_config:
            if weights_config["muon"]:
                if "selected_muons" in pruned_ev.fields:
                    muon_corrector = MuonCorrector(
                        events=pruned_ev,
                        weights=weights_container,
                        year=year,
                        variation=variation,
                        id_wp=weights_config["muon"]["id"],
                        iso_wp=weights_config["muon"]["iso"],
                    )
                    if "id" in weights_config["muon"]:
                        if weights_config["muon"]["id"]:
                            muon_corrector.add_id_weight()

                    if "reco" in weights_config["muon"]:
                        if weights_config["muon"]["reco"]:
                            muon_corrector.add_reco_weight()

                    if "iso" in weights_config["muon"]:
                        if weights_config["muon"]["iso"]:
                            muon_corrector.add_iso_weight()

                    if "trigger" in weights_config["muon"]:
                        if weights_config["muon"]["trigger"]:
                            muon_corrector.add_triggeriso_weight()

        if "tau" in weights_config:
            if weights_config["tau"]:
                if "selected_taus" in pruned_ev.fields:
                    tau_corrector = TauCorrector(
                        events=pruned_ev,
                        weights=weights_container,
                        year=year,
                        tau_vs_jet=weights_config["tau"]["taus_vs_jet"],
                        tau_vs_ele=weights_config["tau"]["taus_vs_ele"],
                        tau_vs_mu=weights_config["tau"]["taus_vs_mu"],
                        variation=variation,
                    )
                    if "taus_vs_jet" in weights_config["tau"]:
                        if weights_config["tau"]["taus_vs_jet"]:
                            tau_corrector.add_id_weight_deeptauvsjet()
                    if "taus_vs_ele" in weights_config["tau"]:
                        if weights_config["tau"]["taus_vs_ele"]:
                            tau_corrector.add_id_weight_deeptauvse()
                    if "taus_vs_mu" in weights_config["tau"]:
                        if weights_config["tau"]["taus_vs_mu"]:
                            tau_corrector.add_id_weight_deeptauvsmu()

    else:
        weights_container.add("weight", np.ones(len(pruned_ev)))
    return weights_container
