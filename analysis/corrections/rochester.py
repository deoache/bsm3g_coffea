import numpy as np
import awkward as ak
import correctionlib
from pathlib import Path
from random import random
from scipy.special import erfinv, erf
from analysis.corrections.met import corrected_polar_met
from coffea.lookup_tools import txt_converters, rochester_lookup


# taken from: https://gitlab.cern.ch/cms-muonPOG/muonscarekit/-/blob/master/scripts/MuonScaRe.py?ref_type=heads
class CrystallBall:
    def __init__(self, m, s, a, n):
        self.pi = 3.14159
        self.sqrtPiOver2 = np.sqrt(self.pi / 2.0)
        self.sqrt2 = np.sqrt(2.0)
        self.m = ak.Array(m)
        self.s = ak.Array(s)
        self.a = ak.Array(a)
        self.n = ak.Array(n)
        self.fa = abs(self.a)
        self.ex = np.exp(-self.fa * self.fa / 2)
        self.A = (self.n / self.fa) ** self.n * self.ex
        self.C1 = self.n / self.fa / (self.n - 1) * self.ex
        self.D1 = 2 * self.sqrtPiOver2 * erf(self.fa / self.sqrt2)

        self.B = self.n / self.fa - self.fa
        self.C = (self.D1 + 2 * self.C1) / self.C1
        self.D = (self.D1 + 2 * self.C1) / 2

        self.N = 1.0 / self.s / (self.D1 + 2 * self.C1)
        self.k = 1.0 / (self.n - 1)

        self.NA = self.N * self.A
        self.Ns = self.N * self.s
        self.NC = self.Ns * self.C1
        self.F = 1 - self.fa * self.fa / self.n
        self.G = self.s * self.n / self.fa
        self.cdfMa = self.cdf(self.m - self.a * self.s)
        self.cdfPa = self.cdf(self.m + self.a * self.s)

    def cdf(self, x):
        x = ak.Array(x)
        d = (x - self.m) / self.s
        result = ak.full_like(d, 1.0)

        # define different conditions
        c1a = (d < -self.a) & (self.F - self.s * d / self.G > 0)
        c1b = (d < -self.a) & (self.F - self.s * d / self.G <= 0)
        c2a = (d > self.a) & (self.F + self.s * d / self.G > 0)
        c2b = (d > self.a) & (self.F + self.s * d / self.G <= 0)

        c3 = ~c1a & ~c1b & ~c2a & ~c2b
        # For d < -a
        result = ak.where(
            c1a, self.NC / np.power(self.F - self.s * d / self.G, self.n - 1), result
        )
        result = ak.where(c1b, self.NC, result)

        # For d > a
        result = ak.where(
            c2a,
            self.NC * (self.C - np.power(self.F + self.s * d / self.G, 1 - self.n)),
            result,
        )
        result = ak.where(c2b, self.NC * self.C, result)

        # For -a <= d <= a
        result = ak.where(
            c3, self.Ns * (self.D - self.sqrtPiOver2 * erf(-d / self.sqrt2)), result
        )

        return result

    def invcdf(self, u):
        u = ak.Array(u)
        result = ak.zeros_like(u)

        c1a = (u < self.cdfMa) & (self.NC / u > 0)
        c1b = (u < self.cdfMa) & (self.NC / u <= 0)
        c2a = (u > self.cdfPa) & (self.C - u / self.NC > 0)
        c2b = (u > self.cdfPa) & (self.C - u / self.NC <= 0)
        c3 = ~c1a & ~c1b & ~c2a & ~c2b

        # For u < cdfMa
        result = ak.where(
            c1a, self.m + self.G * (self.F - (self.NC / u) ** self.k), result
        )
        result = ak.where(c1b, self.m + self.G * self.F, result)

        # For u > cdfPa
        result = ak.where(
            c2a,
            self.m - self.G * (self.F - (self.C - u / self.NC) ** (-self.k)),
            result,
        )
        result = ak.where(c2b, self.m - self.G * self.F, result)

        # For cdfMa <= u <= cdfPa
        result = ak.where(
            c3,
            self.m
            - self.sqrt2 * self.s * erfinv((self.D - u / self.Ns) / self.sqrtPiOver2),
            result,
        )

        return result


def get_rndm(eta, nL, cset, nested=False):
    # obtain parameters from correctionlib
    if nested:
        eta_f, nL_f, nmuons = ak.flatten(eta), ak.flatten(nL), ak.num(nL)
    else:
        eta_f, nL_f, nmuons = eta, nL, np.ones_like(eta)

    mean_f = cset.get("cb_params").evaluate(abs(eta_f), nL_f, 0)
    sigma_f = cset.get("cb_params").evaluate(abs(eta_f), nL_f, 1)
    n_f = cset.get("cb_params").evaluate(abs(eta_f), nL_f, 2)
    alpha_f = cset.get("cb_params").evaluate(abs(eta_f), nL_f, 3)

    # get random number following the CB
    # print(nmuons)
    rndm_f = [random() for i in nmuons for j in range(int(i))]

    cb_f = CrystallBall(mean_f, sigma_f, alpha_f, n_f)

    result_f = cb_f.invcdf(rndm_f)

    if nested:
        result = ak.unflatten(result_f, nmuons)
    else:
        result = result_f

    return result


def get_std(pt, eta, nL, cset, nested=False):
    if nested:
        eta_f, nL_f, pt_f, nmuons = (
            ak.flatten(eta),
            ak.flatten(nL),
            ak.flatten(pt),
            ak.num(nL),
        )
    else:
        eta_f, nL_f, pt_f, nmuons = eta, nL, pt, 1

    # obtain parameters from correctionlib
    param0_f = cset.get("poly_params").evaluate(abs(eta_f), nL_f, 0)
    param1_f = cset.get("poly_params").evaluate(abs(eta_f), nL_f, 1)
    param2_f = cset.get("poly_params").evaluate(abs(eta_f), nL_f, 2)

    # calculate value and return max(0, val)
    sigma_f = param0_f + param1_f * pt_f + param2_f * pt_f * pt_f
    sigma_corrected_f = np.where(sigma_f < 0, 0, sigma_f)

    if nested:
        result = ak.unflatten(sigma_corrected_f, nmuons)
    else:
        result = sigma_corrected_f

    return result


def get_k(eta, var, cset, nested=False):
    if nested:
        eta_f, nmuons = ak.flatten(eta), ak.num(eta)
    else:
        eta_f = eta

    # obtain parameters from correctionlib
    k_data_f = cset.get("k_data").evaluate(abs(eta_f), var)
    k_mc_f = cset.get("k_mc").evaluate(abs(eta_f), var)

    # calculate residual smearing factor
    # return 0 if smearing in MC already larger than in data
    k_f = np.zeros_like(k_data_f)
    condition = k_mc_f < k_data_f
    # k_f[condition] = (k_data_f[condition] ** 2 - k_mc_f[condition] ** 2) ** 0.5
    k_f = ak.where(condition, (k_data_f**2 - k_mc_f**2) ** 0.5, k_f)
    if nested:
        result = ak.unflatten(k_f, nmuons)
    else:
        result = k_f

    return result


def filter_boundaries(pt_corr, pt, nested):
    if not nested:
        pt_corr = np.asarray(pt_corr)
        pt = np.asarray(pt)

    # Check for pt values outside the range of [26, 200]
    outside_bounds = (pt < 26) | (pt > 200)

    if nested:
        n_pt_outside = ak.sum(ak.any(outside_bounds, axis=-1))
    else:
        n_pt_outside = np.sum(outside_bounds)

    if n_pt_outside > 0:
        print(
            f"There are {n_pt_outside} events with muon pt outside of [26,200] GeV. "
            "Setting those entries to their initial value."
        )
        pt_corr = np.where(pt > 200, pt, pt_corr)
        pt_corr = np.where(pt < 26, pt, pt_corr)

    # Check for NaN entries in pt_corr
    nan_entries = np.isnan(pt_corr)

    if nested:
        n_nan = ak.sum(ak.any(nan_entries, axis=-1))
    else:
        n_nan = np.sum(nan_entries)

    if n_nan > 0:
        print(
            f"There are {n_nan} nan entries in the corrected pt. "
            "This might be due to the number of tracker layers hitting boundaries. "
            "Setting those entries to their initial value."
        )
        pt_corr = np.where(np.isnan(pt_corr), pt, pt_corr)

    return pt_corr


def pt_resol(pt, eta, nL, cset, nested=False):
    """ "
    Function for the calculation of the resolution correction
    Input:
    pt - muon transverse momentum
    eta - muon pseudorapidity
    nL - muon number of tracker layers
    cset - correctionlib object

    This function should only be applied to reco muons in MC!
    """
    rndm = get_rndm(eta, nL, cset, nested)
    std = get_std(pt, eta, nL, cset, nested)
    k = get_k(eta, "nom", cset, nested)

    pt_corr = pt * (1 + k * std * rndm)

    pt_corr = filter_boundaries(pt_corr, pt, nested)

    return pt_corr


def pt_resol_var(pt_woresol, pt_wresol, eta, updn, cset, nested=False):
    """
    Function for the calculation of the resolution uncertainty
    Input:
    pt_woresol - muon transverse momentum without resolution correction
    pt_wresol - muon transverse momentum with resolution correction
    eta - muon pseudorapidity
    updn - uncertainty variation (up or dn)
    cset - correctionlib object

    This function should only be applied to reco muons in MC!
    """

    if nested:
        eta_f, nmuons = ak.flatten(eta), ak.num(eta)
        pt_wresol_f, pt_woresol_f = ak.flatten(pt_wresol), ak.flatten(pt_woresol)
    else:
        eta_f, nmuons = eta, 1
        pt_wresol_f, pt_woresol_f = pt_wresol, pt_woresol

    k_unc_f = cset.get("k_mc").evaluate(abs(eta_f), "stat")
    k_f = cset.get("k_mc").evaluate(abs(eta_f), "nom")

    pt_var_f = pt_wresol_f

    # Define condition and standard correction
    condition = k_f > 0
    std_x_cb = (pt_wresol_f / pt_woresol_f - 1) / k_f

    # Apply up or down variation using ak.where
    if updn == "up":
        pt_var_f = ak.where(
            condition,
            pt_woresol_f * (1 + (k_f + k_unc_f) * std_x_cb),
            pt_var_f,
        )
    elif updn == "dn":
        pt_var_f = ak.where(
            condition,
            pt_woresol_f * (1 + (k_f - k_unc_f) * std_x_cb),
            pt_var_f,
        )
    else:
        print("ERROR: updn must be 'up' or 'dn'")

    if nested:
        pt_var = ak.unflatten(pt_var_f, nmuons)
    else:
        pt_var = pt_var_f

    return pt_var


def pt_scale(is_data, pt, eta, phi, charge, cset, nested=False):
    """
    Function for the calculation of the scale correction
    Input:
    is_data - flag that is True if dealing with data and False if MC
    pt - muon transverse momentum
    eta - muon pseudorapidity
    phi - muon angle
    charge - muon charge
    var - variation (standard is "nom")
    cset - correctionlib object

    This function should be applied to reco muons in data and MC
    """
    if is_data:
        dtmc = "data"
    else:
        dtmc = "mc"

    if nested:
        eta_f, phi_f, nmuons = ak.flatten(eta), ak.flatten(phi), ak.num(eta)
    else:
        eta_f, phi_f, nmuons = eta, phi, 1

    a_f = cset.get("a_" + dtmc).evaluate(eta_f, phi_f, "nom")
    m_f = cset.get("m_" + dtmc).evaluate(eta_f, phi_f, "nom")

    if nested:
        a, m = ak.unflatten(a_f, nmuons), ak.unflatten(m_f, nmuons)
    else:
        a, m = a_f, m_f

    pt_corr = 1.0 / (m / pt + charge * a)

    pt_corr = filter_boundaries(pt_corr, pt, nested)

    return pt_corr


def pt_scale_var(pt, eta, phi, charge, updn, cset, nested=False):
    """
    Function for the calculation of the scale uncertainty
    Input:
    pt - muon transverse momentum
    eta - muon pseudorapidity
    phi - muon angle
    charge - muon charge
    updn - uncertainty variation (up or dn)
    cset - correctionlib object

    This function should be applied to reco muons in MC!
    """

    if nested:
        eta_f, phi_f, nmuons = ak.flatten(eta), ak.flatten(phi), ak.num(eta)
    else:
        eta_f, phi_f, pt_f, nmuons = eta, phi, pt, 1

    stat_a_f = cset.get("a_mc").evaluate(eta_f, phi_f, "stat")
    stat_m_f = cset.get("m_mc").evaluate(eta_f, phi_f, "stat")
    stat_rho_f = cset.get("m_mc").evaluate(eta_f, phi_f, "rho_stat")

    if nested:
        stat_a, stat_m, stat_rho = (
            ak.unflatten(stat_a_f, nmuons),
            ak.unflatten(stat_m_f, nmuons),
            ak.unflatten(stat_rho_f, nmuons),
        )

    unc = (
        pt
        * pt
        * (
            stat_m * stat_m / (pt * pt)
            + stat_a * stat_a
            + 2 * charge * stat_rho * stat_m / pt * stat_a
        )
        ** 0.5
    )

    pt_var = pt

    if updn == "up":
        pt_var = pt_var + unc
    elif updn == "dn":
        pt_var = pt_var - unc
    else:
        print("ERROR: updn must be 'up' or 'dn'")

    return pt_var


def apply_rochester_corrections_run3(
    events: ak.Array,
    year: str,
    workflow_config: str,
    shifts: dict,
    corrections_config: dict,
):
    muon_ss_files = {
        "2022preEE": "/cvmfs/cms-griddata.cern.ch/cat/metadata/MUO/Run3-22CDSep23-Summer22-NanoAODv12/2025-08-14/muon_scalesmearing.json.gz",
        "2022postEE": "/cvmfs/cms-griddata.cern.ch/cat/metadata/MUO/Run3-22EFGSep23-Summer22EE-NanoAODv12/2025-08-14/muon_scalesmearing.json.gz",
        "2023preBPix": "/cvmfs/cms-griddata.cern.ch/cat/metadata/MUO/Run3-23CSep23-Summer23-NanoAODv12/2025-08-14/muon_scalesmearing.json.gz",
        "2023postBPix": "/cvmfs/cms-griddata.cern.ch/cat/metadata/MUO/Run3-23DSep23-Summer23BPix-NanoAODv12/2025-08-14/muon_scalesmearing.json.gz",
        "2024": "/cvmfs/cms-griddata.cern.ch/cat/metadata/MUO/Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15/2025-10-17/muon_scalesmearing.json.gz",
    }
    json_path = muon_ss_files[year]
    cset = correctionlib.CorrectionSet.from_file(str(json_path))

    muons = events.Muon
    if hasattr(events, "genWeight"):
        # MC: both scale correction to gen Z peak AND resolution correction to Z width in data
        muon_ptscalecorr = pt_scale(
            False,
            muons.pt,
            muons.eta,
            muons.phi,
            muons.charge,
            cset,
            nested=True,
        )
        muon_ptcorr = pt_resol(
            muon_ptscalecorr,
            muons.eta,
            muons.nTrackerLayers,
            cset,
            nested=True,
        )
    else:
        # Data: only scale correction to gen Z peak
        muon_ptcorr = pt_scale(
            True,
            muons.pt,
            muons.eta,
            muons.phi,
            muons.charge,
            cset,
            nested=True,
        )

    # add nominal scale & smearing correction to shifts
    muons["pt"] = muon_ptcorr
    for i in range(len(shifts)):
        shifts[i][0]["Muon"] = muons

    # systematics
    if hasattr(events, "genWeight") and corrections_config["apply_obj_syst"]:
        muons_ptscalecorr_up = pt_scale_var(
            muon_ptcorr, muons.eta, muons.phi, muons.charge, "up", cset, nested=True
        )
        muons_ptscalecorr_down = pt_scale_var(
            muon_ptcorr, muons.eta, muons.phi, muons.charge, "dn", cset, nested=True
        )
        muons_ptcorr_resol_up = pt_resol_var(
            muon_ptscalecorr, muon_ptcorr, muons.eta, "up", cset, nested=True
        )
        muons_ptcorr_resol_down = pt_resol_var(
            muon_ptscalecorr, muon_ptcorr, muons.eta, "dn", cset, nested=True
        )

        mu_scale_up, mu_scale_down = events.Muon, events.Muon
        mu_resol_up, mu_resol_down = events.Muon, events.Muon

        mu_scale_up["pt"] = muons_ptscalecorr_up
        mu_scale_down["pt"] = muons_ptscalecorr_down
        mu_resol_up["pt"] = muons_ptcorr_resol_up
        mu_resol_down["pt"] = muons_ptcorr_resol_down

        shifts += [
            (
                {
                    "Jet": shifts[0][0]["Jet"],
                    "MET": shifts[0][0]["MET"],
                    "Muon": mu_scale_up,
                },
                f"CMS_scale_m_{year[:4]}Up",
            )
        ]
        shifts += [
            (
                {
                    "Jet": shifts[0][0]["Jet"],
                    "MET": shifts[0][0]["MET"],
                    "Muon": mu_scale_down,
                },
                f"CMS_scale_m_{year[:4]}Down",
            )
        ]
        shifts += [
            (
                {
                    "Jet": shifts[0][0]["Jet"],
                    "MET": shifts[0][0]["MET"],
                    "Muon": mu_resol_up,
                },
                f"CMS_res_m_{year[:4]}Up",
            )
        ]
        shifts += [
            (
                {
                    "Jet": shifts[0][0]["Jet"],
                    "MET": shifts[0][0]["MET"],
                    "Muon": mu_resol_down,
                },
                f"CMS_res_m_{year[:4]}Down",
            )
        ]

    return shifts


def apply_rochester_corrections_run2(
    events, year, workflow_config: str, shifts: dict, corrections_config: dict
):
    """apply rochester corrections for Run2"""
    # https://twiki.cern.ch/twiki/bin/viewauth/CMS/RochcorMuon
    rochester_data = txt_converters.convert_rochester_file(
        f"analysis/data/RoccoR{year}UL.txt", loaduncs=True
    )
    rochester = rochester_lookup.rochester_lookup(rochester_data)

    if hasattr(events, "genWeight"):
        hasgen = ~np.isnan(ak.fill_none(events.Muon.matched_gen.pt, np.nan))
        mc_rand = np.random.rand(*ak.to_numpy(ak.flatten(events.Muon.pt)).shape)
        mc_rand = ak.unflatten(mc_rand, ak.num(events.Muon.pt, axis=1))
        corrections = np.array(ak.flatten(ak.ones_like(events.Muon.pt)))
        mc_kspread = rochester.kSpreadMC(
            events.Muon.charge[hasgen],
            events.Muon.pt[hasgen],
            events.Muon.eta[hasgen],
            events.Muon.phi[hasgen],
            events.Muon.matched_gen.pt[hasgen],
        )
        mc_ksmear = rochester.kSmearMC(
            events.Muon.charge[~hasgen],
            events.Muon.pt[~hasgen],
            events.Muon.eta[~hasgen],
            events.Muon.phi[~hasgen],
            events.Muon.nTrackerLayers[~hasgen],
            mc_rand[~hasgen],
        )
        hasgen_flat = np.array(ak.flatten(hasgen))
        corrections[hasgen_flat] = np.array(ak.flatten(mc_kspread))
        corrections[~hasgen_flat] = np.array(ak.flatten(mc_ksmear))
        corrections = ak.unflatten(corrections, ak.num(events.Muon.pt, axis=1))

        if workflow_config in ["zplusjets", "wplusjets"]:
            corrections = ak.where(
                events.Muon.mediumId, corrections, ak.ones_like(events.Muon.pt)
            )

        errors = np.array(ak.flatten(ak.ones_like(events.Muon.pt)))
        errspread = rochester.kSpreadMCerror(
            events.Muon.charge[hasgen],
            events.Muon.pt[hasgen],
            events.Muon.eta[hasgen],
            events.Muon.phi[hasgen],
            events.Muon.matched_gen.pt[hasgen],
        )
        errsmear = rochester.kSmearMCerror(
            events.Muon.charge[~hasgen],
            events.Muon.pt[~hasgen],
            events.Muon.eta[~hasgen],
            events.Muon.phi[~hasgen],
            events.Muon.nTrackerLayers[~hasgen],
            mc_rand[~hasgen],
        )
        errors[hasgen_flat] = np.array(ak.flatten(errspread))
        errors[~hasgen_flat] = np.array(ak.flatten(errsmear))
        errors = ak.unflatten(errors, ak.num(events.Muon.pt, axis=1))
    else:
        corrections = rochester.kScaleDT(
            events.Muon.charge, events.Muon.pt, events.Muon.eta, events.Muon.phi
        )
        errors = rochester.kScaleDTerror(
            events.Muon.charge, events.Muon.pt, events.Muon.eta, events.Muon.phi
        )

    events["Muon", "pt_raw"] = events.Muon.pt
    muons = events.Muon
    counts = ak.num(events.Muon)

    flat_muons = ak.flatten(muons)
    corrected_pt = flat_muons.pt_raw * ak.flatten(corrections)

    # add nominal scale & smearing correction to shifts
    pt_nom = ak.where(flat_muons.pt <= 200, corrected_pt, flat_muons.pt)
    muons["pt"] = ak.unflatten(pt_nom, counts)
    for i in range(len(shifts)):
        shifts[i][0]["Muon"] = muons

    # Propagate muon pT changes to MET
    events["MET", "pt_raw"] = events.MET.pt
    events["MET", "phi_raw"] = events.MET.phi
    met = events.MET
    met["pt"], met["phi"] = corrected_polar_met(
        events.MET.pt_raw,
        events.MET.phi_raw,
        events.Muon.phi,
        events.Muon.pt_raw,
        events.Muon.pt,
    )
    for i in range(len(shifts)):
        shifts[i][0]["MET"] = met

    # systematics
    if hasattr(events, "genWeight") and corrections_config["apply_obj_syst"]:
        pt_error = ak.where(
            flat_muons.pt <= 200, ak.flatten(errors), np.zeros_like(flat_muons.pt)
        )
        pt_delta = flat_muons.pt * pt_error
        pt_up = pt_nom + pt_delta
        pt_down = pt_nom - pt_delta

        mu_up, mu_down = events.Muon, events.Muon

        mu_up["pt"] = ak.unflatten(pt_up, counts)
        mu_down["pt"] = ak.unflatten(pt_down, counts)

        # Propagate muon pT changes to MET
        met_up, met_down = events.MET, events.MET
        met_up["pt"], met_up["phi"] = corrected_polar_met(
            events.MET.pt_raw,
            events.MET.phi_raw,
            mu_up.phi,
            mu_up.pt_raw,
            mu_up.pt,
        )
        met_down["pt"], met_down["phi"] = corrected_polar_met(
            events.MET.pt_raw,
            events.MET.phi_raw,
            mu_down.phi,
            mu_down.pt_raw,
            mu_down.pt,
        )

        shifts += [
            (
                {
                    "Jet": shifts[0][0]["Jet"],
                    "MET": met_up,
                    "Muon": mu_up,
                },
                f"CMS_rochester_{year[:4]}Up",
            )
        ]
        shifts += [
            (
                {
                    "Jet": shifts[0][0]["Jet"],
                    "MET": met_down,
                    "Muon": mu_down,
                },
                f"CMS_rochester_{year[:4]}Down",
            )
        ]

    return shifts
