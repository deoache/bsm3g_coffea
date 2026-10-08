# BSM3G Coffea

[![Codestyle](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


Python package that uses a columnar framework to process input tree-based [NanoAOD](https://twiki.cern.ch/twiki/bin/view/CMSPublic/WorkBookNanoAOD) files using the [coffea](https://coffeateam.github.io/coffea/) and [scikit-hep](https://scikit-hep.org) Python libraries.

- [Setting up the environment](#Setting-up-the-environment)
- [Input filesets](#Input-filesets)
- [Workflows](#Workflows)
- [Local test](#Local-test)
- [Submit Condor jobs](#Submit-Condor-jobs)
- [Postprocessing](#Postprocessing)

## Setting up the environment

**1. Log in to lxplus and clone the repository**
   
If you haven't done so already:
```bash
ssh <your_username>@lxplus.cern.ch

git clone -b coffea-2026 https://github.com/deoache/bsm3g_coffea.git
cd bsm3g_coffea
```

**2. Initialize a valid CMS grid proxy**

To access remote datasets via xrootd, you need a valid grid proxy.

To generate the proxy:
```bash
voms-proxy-init --voms cms
```
If you're not registered in the CMS VO, follow these [instructions](https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideLcgAccess) to request access.


**3. Initialize the environment**

Run the following command
```bash
source setup.sh
```
This loads the CERN LCG_110 environment, which provides the required software stack, including Coffea 2026.5.0.

The setup script also ensures that the rucio-clients package is available. If it is not already installed, it is automatically installed in a local directory.

The environment must be initialized in each new shell session before running the analysis.


## Input filesets
 

For each data-taking year or simulation campaign, the corresponding configuration file in 
[`analysis/filesets/<year>_<nano version>.yaml`](https://github.com/deoache/bsm3g_coffea/tree/main/analysis/filesets) 
defines the datasets to be processed.

```yaml
SingleMuonF:
  era: F
  process: Data
  key: muon
  query: SingleMuon/Run2017F-UL2017_MiniAODv2_NanoAODv9-v1/NANOAOD
  xsec: null
DYJetsToLL_inclusive_50:
  era: MC
  process: DYJetsToLL
  key: dy_inclusive
  query: DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v2/NANOAODSIM
  xsec: 6025.6
signal_Wpb_MuNu_M300:
  era: MC
  process: Wpb_MuNu_M300
  key: Wpb_MuNu_M300
  query: Wpb_MuNu_M300/crendonh-RunIISummer20UL17_NanoAODv9_Wpb_MuNu_M300-00000000000000000000000000000000/USER
  xsec: 1.0
```

- **`era`**: The dataset era. For real data, this is the run period letter (e.g. B, F, etc.). For Monte Carlo, use "MC"
- **`process`**: Name of the process, used in postprocessing to group related datasets into the same physical category (e.g. DYJetsToLL across multiple HT bins).
- **`key`**: Category identifier linking the dataset to workflow definitions under `datasets` (see next section)
- **`query`**: CMS DAS path used to locate and fetch the dataset
- **`xsec`**: Cross section in picobarns (pb). Must be null for real data, and specified for Monte Carlo samples so that event weights can be normalized to luminosity.

These fileset definitions are the entry point for the analysis: they control which datasets are included, how they are grouped, and how they are weighted, ensuring a transparent and reproducible workflow across different campaigns.

## Workflows

A workflow defines the full configuration of an analysis: which datasets are used, how objects and events are selected, which corrections and weights are applied, and which histograms are produced. Workflows are stored as YAML files in the [`analysis/workflows`](https://github.com/deoache/bsm3g_coffea/tree/main/analysis/workflows) directory.

Each workflow YAML is organized into the following main sections:

- **`datasets`**: which samples to run over (referencing dataset *keys* defined in the filesets)
- **`object_selection`**: how to build and filter physics objects (muons, electrons, jets, etc.)
- **`event_selection`**: event-level cuts, triggers, and category definitions
- **`corrections`**: physics corrections and event weights  
- **`histogram_config`**: definition of output histograms

Together, these sections control the full behavior of the analysis pipeline.

The available workflows are:

* [ztomumu](https://github.com/deoache/bsm3g_coffea/blob/main/analysis/workflows/ztomumu.yaml)
* [ztoee](https://github.com/deoache/bsm3g_coffea/blob/main/analysis/workflows/ztoee.yaml)
* W'+b
    * $t\bar{t}$ estimation
        * [2b1mu](https://github.com/deoache/bsm3g_coffea/blob/main/analysis/workflows/2b1mu.yaml)
        * [2b1e](https://github.com/deoache/bsm3g_coffea/blob/main/analysis/workflows/2b1e.yaml)
        * [1b1mu1e](https://github.com/deoache/bsm3g_coffea/blob/main/analysis/workflows/1b1mu1e.yaml)
        * [1b1e1mu](https://github.com/deoache/bsm3g_coffea/blob/main/analysis/workflows/1b1e1mu.yaml)
    * QCD estimation
        * [qcd_mu](https://github.com/deoache/bsm3g_coffea/blob/main/analysis/workflows/qcd_mu.yaml)
        * [qcd_cr1T_mu](https://github.com/deoache/bsm3g_coffea/blob/main/analysis/workflows/qcd_cr1T_mu.yaml)
        * [qcd_cr2T_mu](https://github.com/deoache/bsm3g_coffea/blob/main/analysis/workflows/qcd_cr2T_mu.yaml)
        * [qcd_ele](https://github.com/deoache/bsm3g_coffea/blob/main/analysis/workflows/qcd_ele.yaml)
        * [qcd_cr1T_ele](https://github.com/deoache/bsm3g_coffea/blob/main/analysis/workflows/qcd_cr1T_ele.yaml)
        * [qcd_cr2T_ele](https://github.com/deoache/bsm3g_coffea/blob/main/analysis/workflows/qcd_cr2T_ele.yaml)
* VBF SUSY
    * [zplusjets_central](https://github.com/deoache/bsm3g_coffea/blob/main/analysis/workflows/zplusjets_central.yaml)
    * [zplusjets_vbf](https://github.com/deoache/bsm3g_coffea/blob/main/analysis/workflows/zplusjets_vbf.yaml)
    * [wplusjets_central](https://github.com/deoache/bsm3g_coffea/blob/main/analysis/workflows/wplusjets_central.yaml)
    * [wplusjets_vbf](https://github.com/deoache/bsm3g_coffea/blob/main/analysis/workflows/wplusjets_vbf.yaml)


<details>
  <summary><b>Click here to display Workflow documentation</b></summary>
    
#### `datasets`

This section specifies which datasets will be processed by the workflow. Instead of repeating full dataset definitions, it simply lists the **keys** defined in the fileset YAML files.  

```yaml
datasets:
  data:               # Keys of real data samples
    - electron
  mc:                 # Keys of background samples
    - dy_inclusive
    - singletop
    - tt
    - wjets_ht
    - diboson
  signal:             # Keys of signal samples
    - Wpb_MuNu_M300
    - Wpb_MuNu_M400
    - Wpb_MuNu_M600
```
- Each entry corresponds to the `key` field defined in the fileset configuration
- The job submission script `runner.py` reads this list to determine which samples to run for the workflow


#### `object_selection`

This section defines how physics objects (muons, electrons, jets, dileptons, etc.) are selected within the workflow.  
Each object is built starting from a **NanoAOD collection** or from a **custom selection function**, and then filtered with a set of cuts.

```yaml
object_selection:
  muons:
    field: events.Muon
    cuts:
      - events.Muon.pt > 10
      - np.abs(events.Muon.eta) < 2.4
      - working_points.muon_iso(events, 'tight')
      - working_points.muon_id(events, 'tight')
  electrons:
    field: events.Electron
    cuts:
      - events.Electron.pt > 10
      - np.abs(events.Electron.eta) < 2.5
      - working_points.electron_id(events, 'wp80iso')
      - delta_r_higher(events.Electron, objects['muons'], 0.4)
  dimuons:
    field: select_dimuons
    cuts:
      - objects['dimuons'].l1.delta_r(objects['dimuons'].l2) > 0.02
      - objects['dimuons'].l1.charge * objects['dimuons'].l2.charge < 0
      - (objects['dimuons'].p4.mass > 60.0) & (objects['dimuons'].p4.mass < 120.0)
```
* `field`: Defines the source of the object
    * A NanoAOD field such as `events.Muon`
    * A custom function such as `select_dimuons`, implemented as a method of the [`ObjectSelector` class](https://github.com/deoache/bsm3g_coffea/blob/main/analysis/selections/object_selections.py)
Each selected object is stored sequentially in the dictionary `objects`, which allows referencing them later (e.g. `objects['dimuons']`)

* `cuts`: A list of object-level requirements. These can be:
    * Direct expressions using NanoAOD fields (`events.Muon.pt > 24`)
    * Conditions based on other selected objects (`objects['dimuons'].z.mass < 120.0`)
    * A working point function (`working_points.muon_iso(events, 'tight')`), defined in [analysis/working_points/working_points.py](https://github.com/deoache/bsm3g_coffea/blob/main/analysis/working_points/working_points.py)

* Defining reusable masks with `add_cut`

    Instead of applying all cuts at once, you can define named masks that tag objects with different levels of quality. These masks can then be used later in the workflow.
    ```yaml
    muons:
        field: events.Muon
        add_cut:
          is_loose:
            - events.Muon.pt > 5
            - np.abs(events.Muon.eta) < 2.4
            - events.Muon.isGlobal | (events.Muon.isTracker & (events.Muon.nStations > 0))
          is_relaxed:
            - objects['muons'].is_loose
            - np.abs(events.Muon.sip3d) < 4
          is_tight:
            - objects['muons'].is_loose
            - objects['muons'].is_relaxed
            - events.Muon.isPFcand | ((events.Muon.highPtId > 0) & (events.Muon.pt > 200))
    zcandidates:
        field: select_zcandidates 
        add_cut:
          is_ossf:
            - objects['zcandidates'].l1.pdgId == -objects['zcandidates'].l2.pdgId
          is_ss:
            - objects['zcandidates'].l1.pdgId == objects['zcandidates'].l2.pdgId
          is_sr:
            - objects['zcandidates'].is_ossf
            - (1*objects['zcandidates'].l1.is_tight + 1*objects['zcandidates'].l2.is_tight) == 2
          is_sscr:
            - objects['zcandidates'].is_ss
            - objects['zcandidates'].l1.is_relaxed
            - objects['zcandidates'].l2.is_relaxed
    ```



#### `event_selection`

This section defines **event-level requirements** and how they are grouped into **analysis regions (categories)**.  
While `object_selection` filters individual particles, here we decide which *events* are kept for histogramming and further analysis.


```yaml
event_selection:
  hlt_paths:
    muon:                        # Dataset key (as defined in the fileset YAML)
      - SingleMu                 # Trigger flag (defined in analysis/selections/trigger_flags.yaml)
  selections:                    # Event-level selections
    trigger: get_trigger_mask(events, hlt_paths, dataset, year)
    trigger_match: get_trigger_match_mask(events, hlt_paths, year, events.Muon)
    lumi: get_lumi_mask(events, year)
    goodvertex: events.PV.npvsGood > 0
    two_muons: ak.num(objects['muons']) == 2
    one_dimuon: ak.num(objects['dimuons']) == 1
    leading_muon_pt: ak.firsts(objects['muons'].pt) > 30
    subleading_muon_pt: ak.pad_none(objects['muons'], target=2)[:, 1].pt > 15
  categories:
    base:                        # Named selection region
      - goodvertex
      - lumi
      - trigger
      - trigger_match
      - two_muons
      - leading_muon_pt
      - subleading_muon_pt
      - one_dimuon
```
- `hlt_paths`: Maps each dataset key (e.g. muon, electron) to the list of HLT trigger flags relevant for that dataset
    - Trigger flags are defined in [analysis/selections/trigger_flags.yaml](https://github.com/deoache/bsm3g_coffea/blob/main/analysis/selections/trigger_flags.yaml)
    - For data: only the triggers listed under the corresponding dataset key are applied.
    - For MC: all triggers across datasets are combined with a logical OR (via [`get_trigger_mask()`](https://github.com/deoache/bsm3g_coffea/blob/main/analysis/selections/trigger.py#L30))
    - In addition, lepton–trigger object matching is enforced (via [`get_trigger_match_mask()`](https://github.com/deoache/bsm3g_coffea/blob/main/analysis/selections/trigger.py#L63-L186)) to ensure selected leptons are consistent with the fired triggers.

- `selections`: Defines event-level cuts. Similarly to object selection, you can use any valid expression from a NanoAOD field or a custom event-selection function defined at [`analysis/selections/event_selections.py`](https://github.com/deoache/bsm3g_coffea/blob/main/analysis/selections/event_selections.py)

- `categories`: Named groups of selections that define analysis regions
    - Each category is a list of selection keys.
    - These regions are used to fill histograms and run postprocessing.



#### `corrections`

This section specifies which **object-level corrections** and **event-level weights** should be applied.  

```yaml
corrections:
  objects:
    - jets                 # JEC/JER
    - jets_veto            # Jets veto maps
    - muons                # Muon scale and resolution
    - electrons            # Electron scale and resolution 
    - taus                 # Tau energy scale
    - met                  # MET phi modulation
  apply_obj_syst: true     # Apply object-level systematic variations
  event_weights:
    genWeight: true
    pileupWeight: true
    l1prefiringWeight: true
    partonshowerWeight: true
    muon:
      - id: tight
      - iso: tight
      - trigger: true
    electron:
      - id: false
      - reco: false
      - trigger: false
```
<br>

- [`apply_obj_syst`](https://github.com/deoache/bsm3g_coffea/blob/main/analysis/processors/base.py#L64): If true, object-level systematic variations (e.g., JEC/JER, MET, tau energy scale, muon rochester corrections) are automatically applied and propagated to downstream event-level calculations.
- Corrections are implemented through a set of utilities managed by two functions:
    - [`object_corrector_manager`](https://github.com/deoache/bsm3g_coffea/blob/main/analysis/corrections/corrections_manager.py#L28): applies object-level corrections (jets scale and smearing corrections, rochester correctiones etc.).  
    - [`weight_manager`](https://github.com/deoache/bsm3g_coffea/blob/main/analysis/corrections/corrections_manager.py#L72): applies event-level weights (pileup, lepton identification efficiencies, etc.).

**Note**: Ensure that the working points used for object selection and event-level corrections (ID, isolation, etc.) are consistent!

<br>

<details>
  <summary><b>Click here to display Corrections documentation</b></summary>
    
**Object-level corrections**
Particle kinematics are not perfectly measured, so we apply corrections to bring reconstructed objects closer to their true values
- **JEC/JER corrections**:
    - **JEC**:  CMS applies a **factorized sequence of corrections**, where each level addresses a specific effect. Each step rescales the jet four-momentum by a correction factor that depends on variables such as jet $p_T$, $\eta$, pileup density $\rho$, and flavor. The corrections are **applied sequentially in a fixed order**, with the output of one step becoming the input to the next.
    <p align="left">
      <img width="600" src="https://i.imgur.com/CkBRbTa.png" />
    </p>

    
    In **data**, the mandatory Jet Energy Corrections are L1 + MC-truth + L2L3Residuals. The official CMS corrections are already applied in NanoAOD.

    In **MC**, the mandatory Jet Energy Corrections are L1 + MC-truth. We must apply the JEC ourselves so that jets are calibrated consistently with data.

    **Note**: Starting Run3, PUPPI jets are the primary jet collection. **PUPPI jets do not need the L1 Pileup corrections**

    - **JER**: Measurements show that the jet energy resolution (spread of reconstructed $p_T$ around the true value) is **worse in data than in simulation**. To account for this, jets in simulation are **smeared** so that their resolution matches the one observed in data.
    Two main methods are recommended:
        - **Scaling method**:  If a particle-level (generator) jet can be matched to the reconstructed jet, the corrected jet $p_T$ is rescaled by a factor  $$c_\text{JER} = 1 + (s_\text{JER} - 1)\frac{p_T - p_T^{\text{ptcl}}}{p_T}$$ where $s_\text{JER}$ is the data-to-simulation scale factor.
        - **Stochastic smearing**:  If no matching generator jet is available, the $p_T$ is randomly fluctuated according to the measured resolution in simulation and the scale factor $s_\text{JER}$.  

        - The **hybrid method** is the CMS recommendation:  
            - Use the scaling method when a good gen–reco match exists.  
            - Fall back to stochastic smearing otherwise.

    Since JEC/JER modifies the kinematics of jets, the **MET must be recomputed** accordingly. In our framework the JEC/JER corrections are handled by Coffea’s [`jetmet_tools`]

    ##### Run2 Workflow

    1. Build Jet/MET Factories
    
        - Use [`coffea.jetmet_tools`](https://coffea-hep.readthedocs.io/en/v0.7.30/modules/coffea.jetmet_tools.html) (JECStack, CorrectedJetsFactory, CorrectedMETFactory) with text-based JEC/JER files
    
        - MC and data have separate sets of files (.jec.txt for JEC, .jr.txt and .jersf.txt for JER). See [here](https://github.com/deoache/bsm3g_coffea/tree/main/analysis/data/JEC)
    
        - Factories are serialized using cloudpickle for later application. See [here](https://github.com/deoache/bsm3g_coffea/blob/main/analysis/data/scripts/build_jec.py)
    
    2. Apply Corrections
    
        - Apply JEC/JER corrections with the pre-built factories. See [here](https://github.com/deoache/bsm3g_coffea/blob/main/analysis/corrections/jec.py)
    
        - MET is updated to propagate the effect of corrected jets (Propagation to MET is included automatically by CorrectedMETFactory)
      

    ##### Run3 Workflow
  - [`jerc_params.yaml`](https://github.com/deoache/bsm3g_coffea/blob/main/analysis/corrections/jerc_params.yaml) defines the parameters needed to build the factories (runs per year, jet algorithms, JEC levels for MC and Data, JEC/JER tags and sources of uncertainties)
  - Apply JEC/JER corrections with the built factories. See [here](https://github.com/deoache/bsm3g_coffea/blob/main/analysis/corrections/jerc.py)
  - **PuppiMET is not updated**: Since the unclustered energy uncertainties in the MET collection switched from cartesian coordinates to ($p_T$, $\phi$) coordinates, the CorrectedMETFactory interface in coffea no longer works 
    
        
    ##### Summary
    
    - **JEC**: rescales jet four-momentum to correct detector response.  
    - **JER**: smears MC jets to match the worse resolution observed in data.  
    - Both corrections must be applied **before event selection**, and MET must be recomputed after applying them.  
    - Our workflow follows the [official CMS recommendations from the JME group](https://cms-jerc.web.cern.ch/JEC/)



<br>

- **[Jet veto maps](https://github.com/deoache/bsm3g_coffea/blob/main/analysis/corrections/jetvetomaps.py#L7)**: These are used to identify regions of the detector with anomalous jet activity, based on the **$\phi$-symmetry** of CMS. These regions include:
    - **Hot zones**: regions with an excess of reconstructed jets.
    - **Cold zones**: regions with a lack of reconstructed jets.
  
    These anomalies can arise from **detector inefficiencies, miscalibrations, or problematic regions** of the calorimeter.
    
    **How the correction works:**
    
    1. **Loading the veto map**  
       - Veto maps are provided as JSON files via `correctionlib` and indexed by year.  
       - Each jet in the event is checked against the map using its $(\eta, \phi)$ coordinates.  
       - Non-zero map values indicate that the jet falls into a vetoed region.
    
    2. **Applying the veto**  
       - Jets located in vetoed regions are **removed** from the event collection.  
       - Only jets in "safe" regions (map value = 0) are kept.
    
    3. **Propagating to MET**  
       - Since removing jets changes the momentum balance, the **MET vector** is recalculated.  
       - The $x$ and $y$ components of MET are adjusted by subtracting the vetoed jets’ momentum contributions.  
       - MET $p_T$ and $\phi$ are updated accordingly.
    
    **Notes:**
    
    - The correction is applied **before event selections**, ensuring that jet-related kinematics (jet counts, MET, etc.) are consistent.  
    - The MET key depends on the year:
      - Run2 & Run3 early years: `"MET"`
      - 2022/2023 later years: `"PuppiMET"`

<br>

- **[MET phi modulation](https://github.com/deoache/bsm3g_coffea/blob/main/analysis/corrections/met.py#L8)**: The distribution of true MET is independent of $\phi$ because of the rotational symmetry of the collisions around the beam axis. However, we observe that the reconstructed MET does depend on $\phi$. The MET $\phi$ distribution has roughly a sinusoidal curve with the period of $2\pi$. The possible causes of the modulation include anisotropic detector responses, inactive calorimeter cells, the detector misalignment, the displacement of the beam spot. The amplitude of the modulation increases roughly linearly with the number of the pile-up interactions.

    This correction reduces the MET $\phi$ modulation. It is also a mitigation for the pile-up effects. (taken from https://twiki.cern.ch/twiki/bin/view/CMSPublic/WorkBookMetAnalysis#7_7_6_MET_Corrections)
  
<br>

- **[Muon corrections (Rochester Corrections)](https://github.com/deoache/bsm3g_coffea/blob/main/analysis/corrections/rochester.py)**: The Rochester corrections are applied to reconstructed muons to improve the agreement between **data and simulation** for the muon transverse momentum ($p_T$). They address two main effects:
    1. **Momentum scale corrections**  
       - Adjust the measured muon $p_T$ to match the known $Z \to \mu\mu$ mass peak in **data** or simulation.  
       - Applied to both data and MC.  
       - Depends on muon properties: $p_T$, $\eta$, $\phi$, and charge.  
       - Scale corrections are handled via **correctionlib** lookups, which provide parameters for the Rochester scale functions.
    
    2. **Momentum resolution (smearing) corrections**  
       - In MC, the reconstructed muon resolution is generally **better than in data**.  
       - A smearing is applied to MC muons to reproduce the observed width of the $Z$ peak.  
       - Two types of smearing are used:
         - **kSpreadMC**: for MC muons matched to a generator-level particle.  
         - **kSmearMC**: for unmatched muons, using the number of tracker layers and a random Gaussian factor.
       - The correction accounts for the number of tracker layers and uses a **Crystall Ball function** to generate random fluctuations consistent with the resolution.
    
    3. **Uncertainty propagation**  
       - Both scale and resolution uncertainties are propagated.  
       - In Run2, separate **up/down variations** are stored for systematic studies.  
       - For Run3, uncertainties are applied as variations using the correctionlib parameters.
    
    4. **MET propagation**  
       - Any change in the muon $p_T$ is propagated to the **MET**.  
       - MET corrections are handled using `corrected_polar_met`, which adjusts the MET vector to account for muon $p_T$ shifts.
    
    
    **Implementation details**:
    
    - Run2 uses `RoccoR` text files and the `coffea.lookup_tools.rochester_lookup` class.  
    - Run3 uses JSON files and `correctionlib`.  
    - Corrections are applied **before event selection** to ensure kinematic quantities are consistent.  
    - Original muon and MET $p_T$ values are stored in `Muon.pt_raw` and `MET.pt_raw` for reference.

<br>

- **[Electron Energy Scale and Smearing Corrections (Run 3 only)](https://github.com/deoache/bsm3g_coffea/blob/main/analysis/corrections/electron_ss.py)**:

    Electron corrections address the difference between the reconstructed electron energy and the true particle energy.
    
    - **Applicability:**  
      - **Run 3:** Corrections must be explicitly applied.  
      - **Run 2:** Already included in the NanoAOD; no additional corrections needed.

    - **Problem:**  
      Detector effects, such as calorimeter response, material interactions, and electronics gain variations, cause reconstructed electron energies to deviate from the true electron energy. These effects depend on the electron’s:
      - Transverse momentum (pt)
      - Pseudorapidity (η)
      - Shower shape (R9)
      - Supercluster gain (for certain years/runs)
    
    - **Correction procedure:**  
      1. **Scale correction (data):**  
         - Adjusts the electron energy so that the reconstructed Z boson mass in data matches the known value.
         - Depends on run number, η, R9, and gain.
      2. **Smearing (MC):**  
         - Adds a stochastic component to MC electrons to reproduce the resolution observed in data.
         - Uses a Gaussian random number multiplied by a pt-dependent smearing factor.
      3. **Boundary checks:**  
         - pt values outside `[20, 250] GeV` or NaNs are reset to the original value.
      4. **MET propagation:**  
         - Changes in electron pt are propagated to the PuppiMET, ensuring consistent event kinematics.
    
    - **Implementation:**  
      - Corrections are applied using `correctionlib` JSON maps specific to each year and dataset.
      - Separate evaluators exist for scale (`EGMScale_Compound`) and smearing (`EGMSmearAndSyst`) corrections.


<br>

- **[Tau corrections](https://github.com/deoache/bsm3g_coffea/blob/main/analysis/corrections/tau_energy.py)**

    Tau energy scale (TES) corrections are applied to genuine tau candidates in simulation to correct their reconstructed energy, momentum, and mass. These corrections ensure that tau kinematics match generator-level expectations and propagate consistently to event-level quantities such as MET. The corrections are applied only to relevant tau candidates:

    - Genuine taus (`genmatch = 5`)
    - Electrons faking taus (`genmatch = 1`)
    - Muons faking taus (`genmatch = 2`)
    - Unmatched jets faking taus (`genmatch = 6`)

    ##### Selection criteria
    TES corrections are applied only to tau candidates that satisfy the following:
    - Decay modes:
        - Single-prong: 0, 1, 2, 10
        - Three-prong: 11
    - Optional pseudorapidity range: 0 ≤ |η| < 2.5
  
    This ensures that only physically meaningful tau candidates are corrected.

    ##### Application
    - Preparation of raw tau quantities: Each tau candidate’s original transverse momentum and mass are stored before corrections.
    - Evaluation of scale factors: scale factors are obtained from the CMS Tau POG JSON files using the tau’s transverse momentum, pseudorapidity, decay mode, and generator-level match. Separate scale factors are available for nominal, up, and down variations to evaluate systematic uncertainties.
    - Application to tau kinematics: The tau candidate’s pt and mass are updated by multiplying by the scale factor. Both nominal values and systematic variations are stored for later analysis.

    Since the corrections modify tau transverse momentum, the missing transverse energy of the event must also be updated to maintain consistent kinematics.


<br>

**Event-level corrections**

We use the common json format for scale factors (SFs), hence the requirement to install [correctionlib](https://github.com/cms-nanoAOD/correctionlib). Most of the SFs can be found in the central [POG repository](https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration), synced once a day with CVMFS: `/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration`. A summary of their content can be found [here](https://cms-nanoaod-integration.web.cern.ch/commonJSONSFs/). The SF implemented are:

* [Pileup SFs](https://github.com/deoache/bsm3g_coffea/blob/main/analysis/corrections/pileup.py)
* [Partonshower SFs](https://github.com/deoache/bsm3g_coffea/blob/main/analysis/corrections/partonshower.py)
* [L1Prefiring SFs](https://github.com/deoache/bsm3g_coffea/blob/main/analysis/corrections/l1prefiring.py) 
* [Electron ID, Reconstruction and Trigger* SFs](https://github.com/deoache/bsm3g_coffea/blob/main/analysis/corrections/electron.py)
* [Muon ID, Iso and TriggerIso Sfs](https://github.com/deoache/bsm3g_coffea/blob/main/analysis/corrections/muon.py)
* [Tau ID Sfs](https://github.com/deoache/bsm3g_coffea/blob/main/analysis/corrections/tau.py)
* [PileupJetId SF](https://github.com/deoache/bsm3g_coffea/blob/main/analysis/corrections/pujetid.py)
* [B-tagging](https://github.com/deoache/bsm3g_coffea/blob/main/analysis/corrections/btag.py):
  b-tagging weights are computed as (see https://twiki.cern.ch/twiki/bin/viewauth/CMS/BTagSFMethods)

  $$w = \prod_{i=\text{tagged}} \frac{SF_{i} \cdot \varepsilon_i}{\varepsilon_i} \prod_{j=\text{not tagged}} \frac{1 - SF_{j} \cdot \varepsilon_j}{1-\varepsilon_j} $$
  
  where $\varepsilon_i$ is the MC b-tagging efficiency and $\text{SF}$ are the b-tagging scale factors. $\text{SF}_i$ and $\varepsilon_i$ are functions of the jet flavor, jet $p_T$, and jet $\eta$. It's important to notice that the two products are 1. over jets tagged at the respective working point, and 2. over jets not tagged at the respective working point. **This is not to be confused with the flavor of the jets**.
  
  We can see, then, that the calculation of these weights require the knowledge of the MC b-tagging efficiencies, which depend on the event kinematics. It's important to emphasize that **the BTV POG only provides the scale factors and it is the analyst responsibility to compute the MC b-tagging efficiencies for each jet flavor in their signal and background MC samples before applying the scale factors**. The calculation of the MC b-tagging efficiencies is describe [here](https://github.com/deoache/bsm3g_coffea/blob/main/notebooks/btag_eff.ipynb), using the outputs from the [btag_eff workflow](https://github.com/deoache/bsm3g_coffea/blob/main/analysis/workflows/btag_eff.yaml)


</details>

<br>

### `histogram_config`

The `histogram_config` defines the histograms that the processor will produce. It specifies which variables to histogram, their binning, labels, and how axes are grouped. It also handles systematic uncertainties and weighted events.

```yaml
histogram_config:
  add_syst_axis: true        # Add a "variation" axis to include systematic uncertainties
  add_weight: true           # Use weighted histograms (hist.storage.Weight)
  flow: true                 # Whether to include underflow/overflow bins
  axes:                      # Dictionary of axis definitions (one per variable)
    <axis_name>:
      type: <AxisType>       # One of: Regular, Variable, Integer, IntCategory, StrCategory
      ...                    # Parameters depending on the type
      label: <str>           # Axis label for plots
      expression: <str>      # Python expression evaluated using objects or events
  layout:                    # Defines how axes are grouped into histograms
    <histogram_name>:
      - <axis_name_1>
      - <axis_name_2>
```
<br>

The configuration consists of several sections:
| Option          | Type | Description                                                                                                           |
| --------------- | ---- | --------------------------------------------------------------------------------------------------------------------- |
| `add_syst_axis` | bool | Add a `StrCategory` axis called `"variation"` to store systematics (e.g., `"nominal"`, `"pileupUp"`, `"pileupDown"`). |
| `add_weight`    | bool | Use weighted histograms via `hist.storage.Weight()` to propagate statistical uncertainties from weighted events.      |
| `flow`          | bool | underflow and overflow events are placed in the first/last bin.                                                                    |

<br>

**Axes definitions**: Each axis corresponds to one variable and defines how it is binned and labeled. Supported axis types are:

| Type          | Description                                                                                     | Required Fields                                      |
|---------------|-------------------------------------------------------------------------------------------------|------------------------------------------------------|
| `Regular`     | Fixed-width bins over a continuous range                                                        | `bins`, `start`, `stop`                              |
| `Variable`    | Custom bin edges over a continuous range                                                        | `edges`                                              |
| `Integer`     | Integer range for discrete binning (e.g. multiplicities)                                        | `start`, `stop`                                      |
| `IntCategory` | Discrete categories (e.g. 0, 4, 5 for jet flavor), optionally with growing capability            | `categories`, `growth` (optional, default `false`)   |
| `StrCategory` | String-labeled categories, typically for systematics or selections                              | `categories`, `growth` (optional, default `false`)   |


- **Expression**: Each axis must define an `expression` fiekd, which is a Python expression evaluated using `events` or `objects`.
- **Layout**: The layout defines how axes are combined into histograms
    - If set to `"individual"`, each axis is turned into a separate 1D histogram.
      - If a dictionary is provided, e.g:
        ```yaml
        layout:
          muon:
            - muon_pt
            - muon_eta
            - muon_phi
        ```
        a multi-dimensional histogram is built (make sure the axes have the same dimensionality)

More info on Hist histograms [here](https://hist.readthedocs.io/en/latest/)

</details>
<br>

### Local test

Local tests can be run using the `tester.ipynb`. There, you can set year, dataset, and executor (`iterative` or `futures`). Feel free to add more datasets in case you need to run a particular workflow. 

To ensure that the notebook uses the same software environment as the analysis, it should be executed with the LCG_110 environment which can be added as a Jupyter kernel by running the following commands in a terminal:

```bash
mkdir -p ~/.local/share/jupyter/kernels/lcg_110
cat << 'EOF' > ~/.local/share/jupyter/kernels/lcg_110/kernel.json
{
 "argv": [
  "/bin/bash",
  "-c",
  "source /cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc13-opt/setup.sh && exec python3 -m ipykernel_launcher -f {connection_file}"
 ],
 "display_name": "LCG 110 (CVMFS)",
 "language": "python"
}
EOF
```
This way, you can check that the workflow is running without issues before submitting batch jobs. It also allows you to interact with the output to check that it makes sense and contains the expected information. After creating the kernel, refresh the available Jupyter kernels in your IDE. If the new kernel is not detected, restart the IDE. The LCG 110 (CVMFS) kernel should then be available for selection in the notebook.


### Submit Condor jobs

**Submit all datasets from a workflow**

Use [runner.py](https://github.com/deoache/higgscharm/blob/lxplus/runner.py) to submit jobs for a specific workflow and campaign/year. 
```bash
python3 runner.py --workflow <workflow> --year <campaign> --submit --eos
``` 

You could use [submit_condor.py](https://github.com/deoache/bsm3g_coffea/blob/main/submit_condor.py) to submit jobs for a specific dataset:
```bash
python3 submit_condor.py --workflow <workflow> --dataset <dataset> --year <campaign> --submit --eos
```

**Note**: It's recommended to add the `--eos` flag to save the outputs to your `/eos` area

**Monitor job status**

To continuously monitor your Condor jobs:
```bash
watch condor_q
```
To get a summary of missing, failed, or incomplete jobs, and optionally resubmit them, use:
```bash
python3 jobs_status.py --workflow <workflow> --year <campaign> --eos
```
It can also regenerate filesets and resubmit jobs if needed.


### Postprocessing

Once the Condor jobs are completed and all outputs are saved under the `outputs/` directory, you can run `run_postprocess.py` to aggregate results, compute cutflows, and generate plots
```bash
python3 run_postprocess.py --workflow <workflow> --year <year> --postprocess --plot --log --eos
``` 

After running post-processing for the two campaigns of a particular year, you can use the same command (e.g. `--year 2016`) to automatically combine both campaigns and compute joint results and plots.

Results will be saved to the same directory as the output files.