import os
import pandas as pd
from ema_workbench.analysis import parcoords
from dmdu.utils_dmdu import get_lever_names, get_outcome_names


def parcoords_levers(exp_dict, bufn, model_colors=None):
    """Plots a parallel axis plot for the passed belief update function (bufn)"""

    # Get limits
    exp_path = os.path.join(os.getcwd(), "data", "paretosort", "input", f"experiments_{bufn.name}.csv")
    all_exp = pd.read_csv(exp_path)
    only_levers = all_exp[get_lever_names()]
    exp_limits = parcoords.get_limits(only_levers)

    # Parallel axis plot
    axes = parcoords.ParallelAxes(exp_limits)
    data = exp_dict[bufn.name][get_lever_names()]
    if model_colors:
        color, alpha = model_colors[bufn.name]
        axes.plot(data, color=color, alpha=alpha)
    else:
        axes.plot(data)


def parcoords_outcomes(out_dict, bufns_list, model_colors=None, other_out_dict_for_limits=None):
    """
    Plots parallel axis coords of the outcomes.

    @param out_dict: dict with the outcomes, {BeliefUpdate : outcomes}
    @param bufns_list: list of BeliefUpdate objects, the bufns to be included in the plot
    @param other_out_dict_for_limits: a different out_dict that can be passed to have shared limits
                                      (e.g., w/ before reevaluation)
    @param model_colors: dict for colors {BeliefUpdate : (color, alpha)}
    """
    # Get limits
    if other_out_dict_for_limits:
        # outcomes_list = out_dict.values() + other_out_dict_for_limits.values()
        concat_out = pd.concat(list(out_dict.values()) + list(other_out_dict_for_limits.values()))
    else:
        concat_out = pd.concat(out_dict.values())
    out_limits = parcoords.get_limits(concat_out)

    # Set up axes
    axes = parcoords.ParallelAxes(out_limits)

    # Invert axes that should be minimized (best values always on top)
    minimize = ['polarization_variance', 'free_speech_constraint', 'avg_user_effort']
    for metric in minimize:
        axes.invert_axis(metric)

    # Plotting
    for bufn in bufns_list:
        data = out_dict[bufn.name][get_outcome_names()]  # to get rid of Unnamed, but causes error like that

        if model_colors:
            color, alpha = model_colors[bufn.name]

            if len(data) <= 50:  # to make sure that pic not to faint
                alpha = 0.9
            axes.plot(data, color=color, alpha=alpha)
        else:
            axes.plot(data)
