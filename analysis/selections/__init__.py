from analysis.selections.utils import (
    delta_r_mask,
    select_dileptons,
    select_dileptons_qcd,
    get_genparticles,
)
from analysis.selections.object_selections import ObjectSelector
import analysis.selections.event_selections as event_selections

get_lumi_mask = event_selections.get_lumi_mask
get_trigger_mask = event_selections.get_trigger_mask
get_trigger_match_mask = event_selections.get_trigger_match_mask
get_metfilters_mask = event_selections.get_metfilters_mask
get_stitching_mask = event_selections.get_stitching_mask
get_hemcleaning_mask = event_selections.get_hemcleaning_mask
