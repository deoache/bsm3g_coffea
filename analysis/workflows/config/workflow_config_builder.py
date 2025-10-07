import yaml
import importlib.resources
from analysis.histograms import HistogramConfig
from .workflow_config import WorkflowConfig


class WorkflowConfigBuilder:

    def __init__(self, workflow: str):
        with importlib.resources.open_text(
            f"analysis.workflows", f"{workflow}.yaml"
        ) as file:
            self.config = yaml.safe_load(file)

    def build_workflow_config(self):
        return WorkflowConfig(
            object_selection=self.parse_object_selection(),
            event_selection=self.parse_event_selection(),
            corrections_config=self.parse_corrections_config(),
            histogram_config=self.parse_histogram_config(),
            datasets=self.parse_datasets_config(),
        )

    def parse_object_selection(self):
        object_selection = {}
        for object_name in self.config["object_selection"]:
            object_selection[object_name] = {
                "field": self.config["object_selection"][object_name]["field"]
            }
            if "cuts" in self.config["object_selection"][object_name]:
                object_selection[object_name]["cuts"] = self.config["object_selection"][
                    object_name
                ]["cuts"]
            if "add_cut" in self.config["object_selection"][object_name]:
                cuts_to_add = self.config["object_selection"][object_name]["add_cut"]
                object_selection[object_name]["add_cut"] = {}
                for cut_name, cuts in cuts_to_add.items():
                    object_selection[object_name]["add_cut"][cut_name] = cuts
        return object_selection

    def parse_event_selection(self):
        event_selection = {}
        for cut_name, cut in self.config["event_selection"].items():
            event_selection[cut_name] = cut
        return event_selection

    def parse_histogram_config(self):
        hist_config = HistogramConfig(**self.config["histogram_config"])
        hist_config.categories = list(self.parse_event_selection()["categories"].keys())
        return hist_config

    def parse_corrections_config(self):
        corrections = {}
        corrections["objects"] = self.config["corrections"]["objects"]
        corrections["apply_obj_syst"] = self.config["corrections"]["apply_obj_syst"]
        corrections["event_weights"] = {}
        for name, vals in self.config["corrections"]["event_weights"].items():
            if isinstance(vals, bool):
                corrections["event_weights"][name] = vals
            elif isinstance(vals, list):
                corrections["event_weights"][name] = {}
                for val in vals:
                    for corr, wp in val.items():
                        corrections["event_weights"][name][corr] = wp
        return corrections

    def parse_datasets_config(self):
        return {key: self.config["datasets"][key] for key in self.config["datasets"]}