import copy
import warnings
import numpy as np
import awkward as ak
from coffea import processor
from coffea.analysis_tools import PackedSelection, Weights
from analysis.workflows.config import WorkflowConfigBuilder
from analysis.histograms import HistBuilder, fill_histograms
from analysis.corrections import (
    object_corrector_manager,
    weight_manager,
)
from analysis.selections import (
    ObjectSelector,
    get_lumi_mask,
    get_trigger_mask,
    get_trigger_match_mask,
    get_metfilters_mask,
    get_stitching_mask,
    get_hemcleaning_mask,
)


def update(events, collections):
    """Return a copy of the events array with the specified collections replaced"""
    out = events
    for name, value in collections.items():
        out = ak.with_field(out, value, name)
    return out


class BaseProcessor(processor.ProcessorABC):
    def __init__(self, workflow: str, year: str = "2017", mode="virtual"):
        assert mode in ["eager", "virtual", "dask"]
        self._mode = mode
        self.year = year
        self.workflow = workflow
        self.year_key = year[:4]
        self.run = "2" if year.startswith("201") else "3"

        config_builder = WorkflowConfigBuilder(workflow=workflow)
        self.workflow_config = config_builder.build_workflow_config()
        self.histogram_config = self.workflow_config.histogram_config
        self.histograms = HistBuilder(self.workflow_config).build_histogram()

    def add_cutflow(self, events, output, selection_manager):
        for category, category_cuts in self.workflow_config.event_selection[
            "categories"
        ].items():
            output["metadata"].update({category: {"cutflow": {"initial": self.sumw}}})
            selections = []
            for cut_name in category_cuts:
                selections.append(cut_name)
                current_selection = selection_manager.all(*selections)
                if ak.sum(current_selection) != 0:
                    sumw_cutflow = (
                        ak.sum(self.genweight[current_selection])
                        if self.is_mc
                        else len(events[current_selection])
                    )
                    output["metadata"][category]["cutflow"][cut_name] = sumw_cutflow

                else:
                    output["metadata"][category]["cutflow"][cut_name] = 0

    def process(self, events):
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        np.seterr(divide="ignore", invalid="ignore")

        # identify data and simulation
        self.is_mc = hasattr(events, "genWeight")

        # compute sumw before jet veto for normalization
        self.sumw = ak.sum(events.genWeight) if self.is_mc else len(events)
        if self.is_mc:
            self.genweight = events.genWeight

        # apply object corrections, jet veto and prepare systematic variations
        vetoed_events, shifts = object_corrector_manager(
            events=events,
            year=self.year,
            run=self.run,
            workflow_config=self.workflow_config,
            dataset=events.metadata["dataset"],
        )
        return processor.accumulate(
            self.process_shift(update(vetoed_events, collections), shift)
            for collections, shift in shifts
        )

    def process_shift(self, events, shift_name):
        # define variables required by workflow expressions evaluated with eval()
        year = self.year
        dataset = events.metadata["dataset"]
        hlt_paths = self.workflow_config.event_selection["hlt_paths"]

        # initialize output metadata
        output = {}
        output["metadata"] = {}
        if shift_name is None:
            # store the initial sum of weights before object selection
            output["metadata"].update({"sumw": self.sumw})

        # object selection
        object_selector = ObjectSelector(
            self.workflow_config.object_selection, self.year, self.run
        )
        objects = object_selector.select_objects(events)

        # event selection
        selection_manager = PackedSelection()
        event_selection = self.workflow_config.event_selection
        for selection, mask in event_selection["selections"].items():
            selection_manager.add(selection, eval(mask))

        self.add_cutflow(events, output, selection_manager)
        # histogram filling
        histograms = copy.deepcopy(self.histograms)
        for category, category_cuts in event_selection["categories"].items():
            category_mask = selection_manager.all(*category_cuts)
            nevents_after = ak.sum(category_mask)
            if nevents_after > 0:
                # select events passing the category requirements
                # add the selected object collections to events
                pruned_ev = events[category_mask]
                for obj in objects:
                    pruned_ev[f"selected_{obj}"] = objects[obj][category_mask]

                # compute event weights
                weights_container = weight_manager(
                    pruned_ev=pruned_ev,
                    year=year,
                    workflow=self.workflow,
                    category=category,
                    run=self.run,
                    workflow_config=self.workflow_config,
                    variation="nominal" if shift_name is None else shift_name,
                    dataset=dataset,
                )
                if shift_name is None:
                    # save weighted event yield after selection
                    weighted_final_nevents = ak.sum(weights_container.weight())
                    output["metadata"][category].update(
                        {"weighted_final_nevents": weighted_final_nevents}
                    )

                # evaluate variables required by the histogram layout
                variables_map = {}
                layout_axes = self.histogram_config.layout.values()
                for variable, axis in self.histogram_config.axes.items():
                    if variable in ak.flatten(layout_axes):
                        variables_map[variable] = eval(axis.expression)[category_mask]

                # fill histograms for the current category and systematic variation
                fill_histograms(
                    histogram_config=self.histogram_config,
                    weights_container=weights_container,
                    variables_map=variables_map,
                    histograms=histograms,
                    shift_name=shift_name,
                    category=category,
                    is_mc=self.is_mc,
                    flow=self.histogram_config.flow,
                )

        output["histograms"] = histograms
        return output

    def postprocess(self, accumulator):
        return accumulator
