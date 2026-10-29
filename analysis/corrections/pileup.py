import correctionlib
import awkward as ak
from typing import Type
from coffea.analysis_tools import Weights
from analysis.corrections import correction_files


def add_pileup_weight(
    events,
    weights_container: Type[Weights],
    year: str,
    variation: str = "nominal",
) -> None:
    """
    add pileup scale factor

    Parameters:
    -----------
        events:
            Events array
        weights_container:
            Weight object from coffea.analysis_tools
        year:
            dataset year {2016preVFP, 2016postVFP, 2017, 2018, 2022preEE, 2022postEE, 2023preBPix, 2023postBPix}
        variation:
            if 'nominal' (default) add 'nominal', 'up' and 'down'
            variations to weights container. else, add only 'nominal' weights.

    https://cms-nanoaod-integration.web.cern.ch/commonJSONSFs/summaries/LUM_2017_UL_puWeights.html
    """
    # define correction set and goldenJSON file names
    cset = correctionlib.CorrectionSet.from_file(correction_files["pileup"][year])
    year_to_corr = {
        "2016preVFP": "Collisions16_UltraLegacy_goldenJSON",
        "2016postVFP": "Collisions16_UltraLegacy_goldenJSON",
        "2017": "Collisions17_UltraLegacy_goldenJSON",
        "2018": "Collisions18_UltraLegacy_goldenJSON",
        "2022preEE": "Collisions2022_355100_357900_eraBCD_GoldenJson",
        "2022postEE": "Collisions2022_359022_362760_eraEFG_GoldenJson",
        "2023preBPix": "Collisions2023_366403_369802_eraBC_GoldenJson",
        "2023postBPix": "Collisions2023_369803_370790_eraD_GoldenJson",
        "2024": "Collisions24_CDEFGHI_goldenJSON",
    }
    # get number of true interactions
    nti = events.Pileup.nTrueInt
    # get nominal scale factors
    nominal_sf = cset[year_to_corr[year]].evaluate(ak.to_numpy(nti), "nominal")
    if variation == "nominal":
        # get up and down variations
        up_sf = cset[year_to_corr[year]].evaluate(ak.to_numpy(nti), "up")
        down_sf = cset[year_to_corr[year]].evaluate(ak.to_numpy(nti), "down")
        # add pileup scale factors to weights container
        weights_container.add(
            name=f"CMS_pileup_{year[:4]}",
            weight=nominal_sf,
            weightUp=up_sf,
            weightDown=down_sf,
        )
    else:
        weights_container.add(
            name=f"CMS_pileup_{year[:4]}",
            weight=nominal_sf,
        )
