import copy
import correctionlib
import numpy as np
import awkward as ak
from pathlib import Path
from analysis.corrections import correction_files
from analysis.corrections.met import corrected_polar_met


def apply_electron_ss_corrections(
    events: ak.Array, year: str, shifts: dict, corrections_config: dict
):
    """
    Apply electron energy scale and smearing corrections for Run 3.

    The nominal electron correction is propagated to all existing MET shifts.
    Electron scale and resolution variations are then constructed independently
    from the MET reference before applying the nominal electron correction.

    References:
        https://egammapog.docs.cern.ch/Run3/SaS/
        https://gitlab.cern.ch/cms-analysis-corrections/EGM/examples/-/blob/latest/egmScaleAndSmearingExample.py
    """
    # load correction set
    cset = correctionlib.CorrectionSet.from_file(correction_files["electron_ss"][year])
    # Preserve the original electron transverse momentum.
    # This is needed to propagate both the nominal correction and the
    # systematic variations to MET.
    events["Electron", "pt_raw"] = events.Electron.pt
    electrons = events.Electron

    # Extract the electron quantities required by the correction payload.
    counts = ak.num(electrons)
    flat_ele = ak.flatten(electrons)
    seedgain = flat_ele.seedGain
    sceta = (
        flat_ele.superclusterEta
        if year == "2024"
        else flat_ele.eta + flat_ele.deltaEtaSC
    )
    r9 = flat_ele.r9
    pt = flat_ele.pt
    run = np.repeat(events.run, counts)

    # Evaluate the nominal electron correction.
    # MC: apply the nominal energy smearing.
    # Data: apply the nominal energy scale correction.
    if hasattr(events, "genWeight"):
        smear = cset["SmearAndSyst"].evaluate("smear", pt, r9, sceta)
        # Generate one random number per electron. The same random numbers
        # must be used for the nominal and systematic smearing variations.
        rng = np.random.default_rng(seed=42)
        random_numbers = rng.normal(
            loc=0.0,
            scale=1.0,
            size=len(pt),
        )

        smearing = 1.0 + smear * random_numbers
        pt_corrected = pt * smearing

    else:
        scale = cset.compound["Scale"].evaluate(
            "scale",
            run,
            sceta,
            r9,
            pt,
            seedgain,
        )

        pt_corrected = pt * scale

    # Update the nominal electron collection in every existing shift.
    electrons = ak.with_field(
        electrons,
        ak.unflatten(pt_corrected, counts),
        "pt",
    )
    for collections, _ in shifts:
        collections["Electron"] = electrons

    # Propagate the nominal electron correction to all existing METs.
    # Save the nominal MET before applying the electron correction.
    # This MET already contains the nominal jet and muon corrections.
    # The saved reference will also be used to construct the electron
    # scale and resolution variations in the following steps.
    met = copy.copy(shifts[0][0]["MET"])

    # Nominal MET: MET_original + ΔJ + Δμ + Δe
    nominal_met = copy.copy(met)
    nominal_met["pt"], nominal_met["phi"] = corrected_polar_met(
        met_pt=nominal_met.pt,
        met_phi=nominal_met.phi,
        other_phi=electrons.phi,
        other_pt_old=electrons.pt_raw,
        other_pt_new=electrons.pt,
    )
    shifts[0][0]["MET"] = nominal_met

    # Apply the same nominal electrons correction to every previously existing
    # systematic shift. This ensures that jet and muon variations also contain
    # the nominal muon correction.

    # For example:
    # JESUp: MET_original + ΔJ_JESUp + Δμ  -> MET_original + ΔJ_JESUp + Δμ + Δe
    # JERUp: MET_original + ΔJ_JERUp + Δμ  -> MET_original + ΔJ_JERUp + Δμ + Δe
    for i in range(1, len(shifts)):
        current_met = copy.copy(shifts[i][0]["MET"])
        current_met["pt"], current_met["phi"] = corrected_polar_met(
            met_pt=current_met.pt,
            met_phi=current_met.phi,
            other_phi=electrons.phi,
            other_pt_old=electrons.pt_raw,
            other_pt_new=electrons.pt,
        )
        shifts[i][0]["MET"] = current_met

    # Evaluate electron scale and resolution variations.
    # These variations are only applied to MC.
    # The correction inputs (pt, r9, ScEta) are evaluated using the
    # original MC electron quantities, before nominal smearing.
    if hasattr(events, "genWeight") and corrections_config["apply_obj_syst"]:
        # Create independent electron collections for each variation.
        ele_smear_up = copy.copy(events.Electron)
        ele_smear_down = copy.copy(events.Electron)
        ele_scale_up = copy.copy(events.Electron)
        ele_scale_down = copy.copy(events.Electron)

        # Electron resolution variations.
        # Use the same random numbers as for the nominal smearing.
        # The correction payload provides the varied smearing widths.
        smear_up = cset["SmearAndSyst"].evaluate("smear_up", pt, r9, sceta)
        smear_down = cset["SmearAndSyst"].evaluate("smear_down", pt, r9, sceta)

        smearing_up = 1.0 + smear_up * random_numbers
        smearing_down = 1.0 + smear_down * random_numbers

        ele_smear_up = ak.with_field(
            ele_smear_up,
            ak.unflatten(pt * smearing_up, counts),
            "pt",
        )
        ele_smear_down = ak.with_field(
            ele_smear_down,
            ak.unflatten(pt * smearing_down, counts),
            "pt",
        )

        # Electron scale variations.
        # The scale variations are evaluated using the original MC
        # electron quantities and applied to the nominally smeared pt.
        scale_up = cset["SmearAndSyst"].evaluate("scale_up", pt, r9, sceta)
        scale_down = cset["SmearAndSyst"].evaluate("scale_down", pt, r9, sceta)

        ele_scale_up = ak.with_field(
            ele_scale_up,
            ak.unflatten(scale_up * pt_corrected, counts),
            "pt",
        )
        ele_scale_down = ak.with_field(
            ele_scale_down,
            ak.unflatten(scale_down * pt_corrected, counts),
            "pt",
        )

        # Propagate the electron scale and resolution variations.
        # All variations start from the same MET reference:
        # MET reference = MET + nominal jet corrections
        #                 + nominal muon corrections

        # The nominal electron correction is NOT included in this
        # reference, because each varied electron collection already
        # contains its corresponding electron momentum.
        met_scale_up, met_scale_down = copy.copy(met), copy.copy(met)
        met_scale_up["pt"], met_scale_up["phi"] = corrected_polar_met(
            met_pt=met_scale_up.pt,
            met_phi=met_scale_up.phi,
            other_phi=ele_scale_up.phi,
            other_pt_old=ele_scale_up.pt_raw,
            other_pt_new=ele_scale_up.pt,
        )
        met_scale_down["pt"], met_scale_down["phi"] = corrected_polar_met(
            met_pt=met_scale_down.pt,
            met_phi=met_scale_down.phi,
            other_phi=ele_scale_down.phi,
            other_pt_old=ele_scale_down.pt_raw,
            other_pt_new=ele_scale_down.pt,
        )
        # Propagate smear variations to MET
        met_smear_up, met_smear_down = copy.copy(met), copy.copy(met)
        met_smear_up["pt"], met_smear_up["phi"] = corrected_polar_met(
            met_pt=met_smear_up.pt,
            met_phi=met_smear_up.phi,
            other_phi=ele_smear_up.phi,
            other_pt_old=ele_smear_up.pt_raw,
            other_pt_new=ele_smear_up.pt,
        )
        met_smear_down["pt"], met_smear_down["phi"] = corrected_polar_met(
            met_pt=met_smear_down.pt,
            met_phi=met_smear_down.phi,
            other_phi=ele_smear_down.phi,
            other_pt_old=ele_smear_down.pt_raw,
            other_pt_new=ele_smear_down.pt,
        )

        # Append the new electron systematic shifts.
        #
        # Start from the nominal collection dictionary to preserve all
        # existing collections, including Tau and any future additions.
        # Only MET and Electron are replaced for each variation.
        shifts += [
            (
                {
                    "Jet": shifts[0][0]["Jet"],
                    "MET": met_scale_up,
                    "Muon": shifts[0][0]["Muon"],
                    "Electron": ele_scale_up,
                },
                f"CMS_scale_e_{year[:4]}Up",
            )
        ]
        shifts += [
            (
                {
                    "Jet": shifts[0][0]["Jet"],
                    "MET": met_scale_down,
                    "Muon": shifts[0][0]["Muon"],
                    "Electron": ele_scale_down,
                },
                f"CMS_scale_e_{year[:4]}Down",
            )
        ]
        shifts += [
            (
                {
                    "Jet": shifts[0][0]["Jet"],
                    "MET": met_smear_up,
                    "Muon": shifts[0][0]["Muon"],
                    "Electron": ele_smear_up,
                },
                f"CMS_res_e_{year[:4]}Up",
            )
        ]
        shifts += [
            (
                {
                    "Jet": shifts[0][0]["Jet"],
                    "MET": met_smear_down,
                    "Muon": shifts[0][0]["Muon"],
                    "Electron": ele_smear_down,
                },
                f"CMS_res_e_{year[:4]}Down",
            )
        ]
    return shifts
