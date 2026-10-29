import numpy as np


def add_scalevar_weight(events, weights_container, variation):
    """
    Twiki: https://twiki.cern.ch/twiki/bin/viewauth/CMS/TopSystematics#Factorization_and_renormalizatio

    __doc__:
    ['LHE scale variation weights (w_var / w_nominal)',
    ' [0] is renscfact=0.5d0 facscfact=0.5d0 ',
    ' [1] is renscfact=0.5d0 facscfact=1d0 ',
    ' [2] is renscfact=0.5d0 facscfact=2d0 ',
    ' [3] is renscfact=1d0 facscfact=0.5d0 ',
    ' [4] is renscfact=1d0 facscfact=1d0 ',
    ' [5] is renscfact=1d0 facscfact=2d0 ',
    ' [6] is renscfact=2d0 facscfact=0.5d0 ',
    ' [7] is renscfact=2d0 facscfact=1d0 ',
    ' [8] is renscfact=2d0 facscfact=2d0 ']
    """
    nom = np.ones(len(weights_container.weight()))
    if variation == "nominal":
        if "LHEScaleWeight" in events.fields:
            lhe_weights = events.LHEScaleWeight
            if len(lhe_weights) > 0:
                if len(lhe_weights[0]) == 9:
                    nom = lhe_weights[:, 4]
                    weights_container.add(
                        "scalevar_muR",
                        nom,
                        lhe_weights[:, 1] / nom,
                        lhe_weights[:, 7] / nom,
                    )
                    weights_container.add(
                        "scalevar_muF",
                        nom,
                        lhe_weights[:, 3] / nom,
                        lhe_weights[:, 5] / nom,
                    )
                    weights_container.add(
                        "scalevar_muR_muF", nom, lhe_weights[:, 0], lhe_weights[:, 8]
                    )
                elif len(lhe_weights[0]) > 1:
                    print("Scale variation vector has length ", len(lhe_weights[0]))
            else:
                warnings.warn(
                    "LHE scale variation weights are not available, put nominal weights"
                )
                weights_container.add("scalevar_muR", nom, nom, nom)
                weights_container.add("scalevar_muF", nom, nom, nom)
                weights_container.add("scalevar_muR_muF", nom, nom, nom)
    else:
        weights_container.add("scalevar_3pt", nom)
