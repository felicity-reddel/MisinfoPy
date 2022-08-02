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
        color = model_colors[bufn.name]
        axes.plot(data, color=color)
    else:
        axes.plot(data)


def parcoords_outcomes(out_dict, bufn, model_colors=None):

    # Get limits
    concat_out = pd.concat(out_dict.values())
    out_limits = parcoords.get_limits(concat_out)

    axes = parcoords.ParallelAxes(out_limits)

    minimize = ['polarization_variance', 'free_speech_constraint', 'avg_user_effort']
    for metric in minimize:
        axes.invert_axis(metric)

    data = out_dict[bufn.name]  # [get_outcome_names()] to get rid of Unnamed, but causes error like that
    if model_colors:
        color = model_colors[bufn.name]
        axes.plot(data, color=color)
    else:
        axes.plot(data)
