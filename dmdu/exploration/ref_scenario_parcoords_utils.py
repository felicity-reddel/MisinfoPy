# General
import os
import pandas as pd
import numpy as np
import seaborn as sns

# ema_workbench
from ema_workbench import load_results
from ema_workbench.analysis import parcoords


def get_results(belief_update_fn, scenarios, replications):
    """
    Quick fn to load data from the exploration for picking the reference scenario.

    @param belief_update_fn: str (BeliefUpdate.name)
    @param scenarios: int, number of scenarios
    @param replications: int, number of replications

    @return: (pd.DataFrame, dict)
    """
    path = os.path.join(os.getcwd(), 'data', f'ref_scenario_{scenarios}_scenarios_{replications}_replications_{belief_update_fn}')
    results = load_results(path)
    experiments, outcomes = results

    return experiments, outcomes


def arraydict_to_df(data):
    """
    Change dict outcomes (from ReplicatorModel with ArrayOutcomes) into the pd.DataFrame form that is required for the parallel axis plots (ema_workbench's parcoords).
    @param data: dict, k: str (outcome name), v: nd_array
    @return: pd.DataFrame
    """
    series_list = []

    for k, v in data.items():
        # flatten (bc don't need seed-info, nor scenario-info
        if not isinstance(v, str):  # bc in end v=DEFFUANT (etc.) for some reason
            flat_data = v.flatten()
            # create series from data (-> 1 col in resulting DataFrame)
            s = pd.Series(flat_data, name=k)
            series_list.append(s)

    # add everything together to a DataFrame
    df = pd.concat(series_list, axis=1)

    return df


def parcoords_color_by(data, color_by):
    """
    Plots parallel axis plots (using ema_workbench's parcoords), colored by specific column in data.

    @param data: dict {str: pd.DataFrame}, str being the BeliefUpdate.name, e.g. 'DEFFUANT'
    @param color_by: str, needs to be a column name in data
    @return:
    """
    # Get unique policy names
    unique = data[str(color_by)].unique().tolist()

    # Assign each unique policy a color
    v_colors = {}
    for _, (v, color) in enumerate(zip(unique, sns.color_palette())):
        v_colors[v] = color

    # Plotting preparations
    sns.set(rc={'figure.figsize': (14, 10)})
    limits = parcoords.get_limits(data)
    axes = parcoords.ParallelAxes(limits)

    minimize = ['polarization_variance', 'free_speech_constraint', 'avg_user_effort']
    for metric in minimize:
        axes.invert_axis(metric)

    for v, color in v_colors.items():
        indices = list(data[data[color_by] == v].index)
        part = data.iloc[indices, :]
        axes.plot(part, color=color, label=v)
        # Bonus: exclude belief_update_fn axis (e.g., get limits via combined df, but do plotting via dict {'DEFFUANT': df} )

    axes.legend()
