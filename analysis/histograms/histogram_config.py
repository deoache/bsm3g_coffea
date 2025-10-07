from dataclasses import dataclass, field
from typing import Union, Dict, Any, List


@dataclass
class VariableAxis:
    name: str
    edges: list
    label: str
    expression: str
    type_: str = field(default="Variable", metadata={"alias": "type"})

    def __post_init__(self):
        self.__dict__["type"] = self.type_
        self.build_args = {"name": self.name, "label": self.label, "edges": self.edges}


@dataclass
class RegularAxis:
    name: str
    bins: list
    start: Union[int, float]
    stop: Union[int, float]
    label: str
    expression: str
    type_: str = field(default="Regular", metadata={"alias": "type"})

    def __post_init__(self):
        self.__dict__["type"] = self.type_
        self.build_args = {
            "name": self.name,
            "label": self.label,
            "bins": self.bins,
            "start": self.start,
            "stop": self.stop,
        }


@dataclass
class IntCategoryAxis:
    name: str
    categories: list
    label: str
    expression: str
    growth: bool = False
    type_: str = field(default="IntCategory", metadata={"alias": "type"})

    def __post_init__(self):
        self.__dict__["type"] = self.type_
        self.build_args = {
            "name": self.name,
            "label": self.label,
            "categories": self.categories,
            "growth": self.growth,
        }


@dataclass
class IntegerAxis:
    name: str
    start: Union[int, float]
    stop: Union[int, float]
    label: str
    expression: str
    type_: str = field(default="Integer", metadata={"alias": "type"})

    def __post_init__(self):
        self.__dict__["type"] = self.type_
        self.build_args = {
            "name": self.name,
            "label": self.label,
            "start": self.start,
            "stop": self.stop,
        }


@dataclass
class StrCategoryAxis:
    name: str
    categories: list
    label: str
    expression: str
    growth: bool = False
    type_: str = field(default="StrCategory", metadata={"alias": "type"})

    def __post_init__(self):
        self.__dict__["type"] = self.type_
        self.build_args = {
            "name": self.name,
            "label": self.label,
            "categories": self.categories,
            "growth": self.growth,
        }


@dataclass
class BooleanAxis:
    name: str
    label: str
    expression: str
    type_: str = field(default="Boolean", metadata={"alias": "type"})

    def __post_init__(self):
        self.__dict__["type"] = self.type_
        self.build_args = {"name": self.name, "label": self.label}


@dataclass
class HistogramConfig:
    """
    config class for a Histogram

    Attributes:
    -----------
        axes:
                dictionary with axis info like {"type": <type>, **args}
                <type> is a hist.axis type: Regular, IntCategory, StrCategory, Variable

                Example:
                    axes = {
                        "jet_pt": {
                            "type": "Variable",
                            "edges ": [30, 100, 1000],
                            "label": r"Jet $p_T$ [GeV]",
                            "expresison": "objects['jet_pt'].pt"
                        },
                        "jet_eta": {
                            "type": "Regular",
                            "bins": 50,
                            "start": -2.5,
                            "stop": 2.5,
                            "label": "Jet $\eta$",
                            "expresison": "objects['jet_pt'].eta"
                        },
                        "njets": {
                            "type": "IntCategory",
                            "categories": [],
                            "gowth": True,
                            "label": "$N_{jets}$",
                            "expresison": "ak.num(objects['jet_pt'])"
                        }
                    }
        layout:
            if "individual", when building the histogram, each axis will be an individual histogram:
                {'jet_pt': hist.Hist(jet_pt_axis), 'jet_eta': hist.Hist(jet_eta_axis), 'njets': hist.Hist(njets__axis)}
            if a dict like {'jet': ['jet_pt', 'jet_eta'], 'njets': ['njets']}, when building the histogram, each key will contain its values as axes:
                {'jet': hist.Hist(jet_pt_axis, jet_eta_axis), 'njets': hist.Hist(njets_axis)}
        add_syst_axis:
            if True histograms will include a StrCategory axis for systematics
        add_weight:
            if True hist.storage.Weight() will be added to the histograms
        flow:
            wether to add the underflow/overflow bins or not
    """

    axes: Dict[str, Any]
    layout: Union[str, Dict[str, List[str]]]
    add_weight: bool = True
    add_syst_axis: bool = True
    flow: bool = True

    def __post_init__(self):
        # set variables attribute
        if isinstance(self.layout, str):
            self.stack = False
            self.variables = list(self.axes.keys())
        elif isinstance(self.layout, dict):
            self.stack = True
            self.variables = []
            for variables in self.layout.values():
                for variable in variables:
                    self.variables.append(variable)

        # replace axes dictionary with instantiated axis objects
        self.dict_axes = self.axes.copy()
        axis_type_map = {
            "Variable": VariableAxis,
            "Regular": RegularAxis,
            "IntCategory": IntCategoryAxis,
            "StrCategory": StrCategoryAxis,
            "Integer": IntegerAxis,
            "Boolean": BooleanAxis,
        }
        for name, axis_dict in self.axes.items():
            axis_type = axis_dict.pop("type")
            axis_dict.update({"name": name})
            hist_axis = axis_type_map[axis_type](**axis_dict)
            self.axes[name] = hist_axis

    def to_dict(self):
        """Convert HistogramConfig to a dictionary."""
        return {
            "add_syst_axis": self.add_syst_axis,
            "add_weight": self.add_weight,
            "axes": self.dict_axes,
            "layout": self.layout,
            "flow": self.flow,
        }
