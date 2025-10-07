import hist


class HistBuilder:
    def __init__(self, workflow_config):
        self.workflow_config = workflow_config
        self.histogram_config = workflow_config.histogram_config
        self.axis_opt = {
            "StrCategory": hist.axis.StrCategory,
            "IntCategory": hist.axis.IntCategory,
            "Regular": hist.axis.Regular,
            "Variable": hist.axis.Variable,
            "Integer": hist.axis.Integer,
            "Boolean": hist.axis.Boolean,
        }
        self.int_cat = False
        if all(isinstance(cat, int) for cat in self.histogram_config.categories):
            self.int_cat = True
            self.int_cat_axis = hist.axis.IntCategory(
                name="category",
                categories=[int(cat) for cat in self.histogram_config.categories],
            )
        else:
            self.cat_axis = hist.axis.StrCategory(
                name="category", categories=self.histogram_config.categories
            )

    def build_histogram(self):
        if self.histogram_config.stack:
            histograms = {}
            for hist_name, axes_names in self.histogram_config.layout.items():
                histograms[hist_name] = self.build_stacked_histogram(axes_names)
        else:
            histograms = self.build_individual_histogram()
        return histograms

    def build_individual_histogram(self):
        histograms = {}
        for axis in self.histogram_config.axes:
            axes = [self.build_axis(axis)]
            if len(self.histogram_config.categories) > 1:
                if self.int_cat:
                    axes.append(self.int_cat_axis)
                else:
                    axes.append(self.cat_axis)
            if self.histogram_config.add_syst_axis:
                axes.append(self.get_syst_axis())
            if self.histogram_config.add_weight:
                axes.append(hist.storage.Weight())
            histograms[axis] = hist.Hist(*axes)
        return histograms

    def build_stacked_histogram(self, axes_names):
        axes = []
        for axis in axes_names:
            axes.append(self.build_axis(axis))
        if len(self.histogram_config.categories) > 1:
            if self.int_cat:
                axes.append(self.int_cat_axis)
            else:
                axes.append(self.cat_axis)
        if self.histogram_config.add_syst_axis:
            axes.append(self.get_syst_axis())
        if self.histogram_config.add_weight:
            axes.append(hist.storage.Weight())
        return hist.Hist(*axes)

    def build_axis(self, axis_name: dict):
        """build a hist axis object from an axis config"""
        hist_type = self.histogram_config.axes[axis_name].type
        axis_args = self.histogram_config.axes[axis_name].build_args
        return self.axis_opt[hist_type](**axis_args)

    def get_syst_axis(self):
        return hist.axis.StrCategory(name="variation", categories=[], growth=True)
