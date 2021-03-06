from model.misinfo_model import MisinfoPy
import os
import pandas as pd
from ema_workbench import (
    Policy,
    ScalarOutcome,
    IntegerParameter,
    RealParameter,
    Constant,
    Model,
    ReplicatorModel,
    ArrayOutcome,
    Scenario
)


# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# Get main inputs for DMDU
# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––


def get_constants(steps, belief_update_fn, n_replications=None):
    """
    Returns the constants. In the fitting format for the ema_workbench.
    @param steps: int, number of model steps
    @param belief_update_fn: BeliefUpdateFn (Enum)
    @param n_replications: int, number of replications (model runs for 1 optimization-step)
    @return: list of ema_workbench Constants
    """
    constants = [
        Constant("steps", steps),
        Constant("belief_update_fn", belief_update_fn),
    ]

    if n_replications:
        constants += [Constant("n_replications", n_replications)]

    return constants


def get_uncertainties():
    """
    Returns the outcomes. In the fitting format for the ema_workbench.
    @return: list of ema_workbench Parameters
    """
    uncertainties = [
        IntegerParameter("belief_metric_threshold", 75, 80),
        IntegerParameter("n_edges", 2, 3),
        RealParameter(
            "ratio_normal_user", 0.98, 0.995
        ),  # other parts need this to stay RealParam
        IntegerParameter("mean_normal_user", 0, 2),
        IntegerParameter("mean_disinformer", 8, 12),
        RealParameter(
            "high_media_lit", 0.25, 0.35
        ),  # other parts need this to stay RealParam
        RealParameter("deffuant_mu", 0.01, 0.03),  # DEFFUANT-specific
        RealParameter("sampling_p_update", 0.01, 0.03),  # SAMPLING-specific
        IntegerParameter("n_posts_estimate_similarity", 5, 15),
    ]  # SIT-specific
    return uncertainties


def get_uncertainty_names():
    """Returns the uncertainty names"""

    names = [
        "belief_metric_threshold",
        "n_edges",
        "ratio_normal_user",
        "mean_normal_user",
        "mean_disinformer",
        "high_media_lit",
        "deffuant_mu",
        "sampling_p_update",
        "n_posts_estimate_similarity",
    ]

    return names


def get_outcomes():
    """
    Returns the outcomes. In the fitting format for the ema_workbench.
    @return: list of ema_workbench Outcomes
    """
    outcomes = [
        ScalarOutcome("n_agents_above_belief_threshold", ScalarOutcome.MAXIMIZE),
        ScalarOutcome("polarization_variance", ScalarOutcome.MINIMIZE),
        ScalarOutcome("engagement", ScalarOutcome.MAXIMIZE),
        ScalarOutcome("free_speech_constraint", ScalarOutcome.MINIMIZE),
        ScalarOutcome("avg_user_effort", ScalarOutcome.MINIMIZE),
    ]

    return outcomes


def get_outcome_names():
    """Returns the lever names"""
    names = [
        "n_agents_above_belief_threshold",
        "polarization_variance",
        "engagement",
        "free_speech_constraint",
        "avg_user_effort"
    ]

    return names


def get_replicator_outcomes():
    """
        Returns the outcomes. In the fitting format for the ema_workbench.
        @return: list of ema_workbench Outcomes
        """
    outcomes = [
        ArrayOutcome("n_agents_above_belief_threshold"),
        ArrayOutcome("polarization_variance"),
        ArrayOutcome("engagement"),
        ArrayOutcome("free_speech_constraint"),
        ArrayOutcome("avg_user_effort"),
    ]

    return outcomes


def get_epsilons():
    """
    Returns the epsilon values for all outcomes. In the same order as the outcomes list.
    These values stem from the analysis in epsilon_values.ipynb.

    In the following order:
    - n_agents_above_belief_threshold:  2       (2, 4.6, 4.8), before 1
    - polarization_variance:            2       (2, 18, 16), before 1
    - engagement:                       40      (48, 30, 39), before 10
    - free_speech_constraint:           0.02    (0.01, 0.04, 0.03), before 0.01
    - avg_user_effort:                  1.0     (0.65 , 1.05, 1.06), before 0.5

    @return: list of floats
    """
    # epsilons = [1.0, 1.0, 10.0, 0.01, 0.5]   # previous values
    epsilons = [2, 2, 40, 0.02, 1.0]

    return epsilons


def get_levers():
    """
    Returns the levers. In the fitting format for the ema_workbench.
    Currently downscaled:
     - for range
        0 -> 0
        1 -> 10
        2 -> 20
        ...
        10 -> 100

    @return: list of ema_workbench Parameters
    """

    levers = [
        IntegerParameter("mlit_select", lower_bound=0, upper_bound=10),
        IntegerParameter("del_t", lower_bound=0, upper_bound=5),
        IntegerParameter("rank_punish", lower_bound=0, upper_bound=10),
        IntegerParameter("rank_t", lower_bound=0, upper_bound=5),
        IntegerParameter("strikes_t", lower_bound=0, upper_bound=5),
    ]

    return levers


def get_test_levers():
    """
    Returns the levers for testing. In the fitting format for the ema_workbench.
    Currently downscaled:
     - for range
        0 -> 0
        1 -> 10
        2 -> 20
        ...
        10 -> 100

    @return: list of ema_workbench Parameters
    """

    levers = [
        IntegerParameter("mlit_select", lower_bound=0, upper_bound=1),
        IntegerParameter("del_t", lower_bound=0, upper_bound=1),
        IntegerParameter("rank_punish", lower_bound=0, upper_bound=1),
        IntegerParameter("rank_t", lower_bound=0, upper_bound=1),
        IntegerParameter("strikes_t", lower_bound=0, upper_bound=1),
    ]

    return levers


def get_lever_names():
    """Returns the lever names"""
    names = ["mlit_select", "del_t", "rank_punish", "rank_t", "strikes_t"]

    return names


def get_reference_scenario():
    """
    Provides the reference scenario (selected in ref_scenario_parcoords.ipynb).
    Can also be loaded from the 'ref_scenario.csv' file.
    @return: Scenario
    """
    params = {'belief_metric_threshold': 80.0,
              'deffuant_mu': 0.0185664008066397,
              'high_media_lit': 0.2851257267625852,
              'mean_disinformer': 8.0,
              'mean_normal_user': 0,
              'n_edges': 3,
              'n_posts_estimate_similarity': 5,
              'ratio_normal_user': 0.9930851855394072,
              'sampling_p_update': 0.027594598317021}
    ref_scenario = Scenario('reference', **params)

    return ref_scenario


def get_100_seeds():
    """Provide the list of 100 used seeds."""
    seeds_100 = [577747, 914425, 445063, 977049, 617127, 639676, 137294, 845058, 718814, 119679, 435223, 347541, 666852,
                 701324, 604437, 908374, 941595, 800210, 745388, 399447, 140918, 910967, 917428, 497096, 222919, 726572,
                 748497, 185669, 610661, 709441, 801330, 506120, 891889, 298223, 164318, 929955, 854094, 553307, 279254,
                 597549, 223105, 708080, 220244, 126086, 634792, 458729, 822070, 972244, 751076, 130675, 100289, 252061,
                 262114, 449996, 206219, 764775, 285626, 385767, 111989, 812234, 305433, 822474, 312966, 877990, 598853,
                 389796, 777981, 937667, 943990, 393412, 913947, 594493, 543410, 199872, 519301, 577412, 615253, 914266,
                 136560, 705707, 433804, 414487, 198043, 325188, 906659, 507433, 268008, 894819, 994630, 427593, 129353,
                 207160, 780566, 131963, 158586, 428856, 485180, 445734, 806806, 958623]

    return seeds_100


# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# Get different prepared sets of policies
# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––


def get_policies_all():
    """Returns list of 2 Policy objects: all-off and all-max."""

    policy_list = [
        Policy(
            "all off",
            **{
                "mlit_select": 0,
                "del_t": 0,
                "rank_punish": 0,
                "rank_t": 0,
                "strikes_t": 0,
            }
        ),
        Policy(
            "all max",
            **{
                "mlit_select": 10,
                "del_t": 5,
                "rank_punish": 10,
                "rank_t": 5,
                "strikes_t": 5,
            }
        ),
    ]

    return policy_list


def get_policies_indiv():
    """
    Returns list of 5 Policy objects:
    Each individual lever maximized while the other levers are on a very low value.
    """

    policy_list = [
        Policy(
            "mlit_select max",
            **{
                "mlit_select": 10,
                "del_t": 0,
                "rank_punish": 0,
                "rank_t": 0,
                "strikes_t": 0,
            }
        ),
        Policy(
            "del_t max",
            **{
                "mlit_select": 0,
                "del_t": 5,
                "rank_punish": 0,
                "rank_t": 0,
                "strikes_t": 0,
            }
        ),
        Policy(
            "rank_punish max",
            **{
                "mlit_select": 0,
                "del_t": 0,
                "rank_punish": 10,
                "rank_t": 1,
                "strikes_t": 0,
            }
        ),
        Policy(
            "rank_t max",
            **{
                "mlit_select": 0,
                "del_t": 0,
                "rank_punish": 1,
                "rank_t": 5,
                "strikes_t": 0,
            }
        ),
        Policy(
            "double rank max",
            **{
                "mlit_select": 0,
                "del_t": 0,
                "rank_punish": 10,
                "rank_t": 5,
                "strikes_t": 0,
            }
        ),
        Policy(
            "strikes_t max",
            **{
                "mlit_select": 0,
                "del_t": 0,
                "rank_punish": 0,
                "rank_t": 0,
                "strikes_t": 5,
            }
        ),
    ]

    return policy_list


def epsilon_helper(
        outcomes, bufn, metric, divide_by=10, best_quantile=0.25, minimize=None
):
    """
    Helps to explore which epsilon-values would be suitable.
    -   This is to explore the results from a random open exploration. As such, the results are likely not as good as
        those that would be in a set of Pareto-optimal policies (either resulting from an optimization or from a
        thorough evaluation of many policy candidates).
    -   Consequently, it will be explored how many policies are not in the whole range, but in a subset (best quantile).
        ("Best" depends on whether the metric is to be minimized or maximized.)
    -   The range of 1 quantile is sliced into a number of parts (divide_by).
    -   The result of the relevant range, divided by divide_by, leads to an epsilon value.
    -   The function returns that epsilon value and how many policies fall within the "first epsilon-sized slice".
    -   Like that it can be estimated what resolution might be useful, i.e., how big the epsilon value should be.
    This can be done for all three models, then a suitable value that might work for each of them can be chosen.
    -> Likely closer to the lower end of the three values (to make sure enough policies are returned).

    @param outcomes: dataframe
    @param bufn: BeliefUpdateFn
    @param metric: string
    @param divide_by: int
    @param best_quantile: float, in range [0.0, 1.0]
    @param minimize: list of strings (metric names)
    @return: tuple, (dataframe, float)
    """

    subset = outcomes[outcomes["belief_update_fn"] == bufn]

    if minimize is None:
        minimize = [
            "polarization_variance",
            "free_speech_constraint",
            "avg_user_effort",
        ]

    if metric in minimize:
        lower_bound = min(subset[metric])
        quantile_value = subset.quantile(q=best_quantile)[metric]
        relevant_range = lower_bound + quantile_value
        epsilon = relevant_range / float(divide_by)
        upper_bound = lower_bound + epsilon
        within_1_epsilon = subset.loc[subset[metric] <= upper_bound]
        within_1_epsilon = within_1_epsilon[metric]
    else:  # maximize metric
        upper_bound = max(subset[metric])
        quantile_value = subset.quantile(q=1 - best_quantile)[metric]
        relevant_range = float(upper_bound - quantile_value)
        epsilon = relevant_range / divide_by
        lower_bound = upper_bound - epsilon
        within_1_epsilon = subset.loc[subset[metric] >= lower_bound]
        within_1_epsilon = within_1_epsilon[metric]

    return within_1_epsilon, epsilon


def model_setup(belief_update_fn, steps):
    """
    Sets up a MisinfoPy model for the ema_workbench. Uses the standard (single) Model.
    @return: MisinfoPy
    """

    # Setting up the model
    model = MisinfoPy()
    model = Model("MisinfoPy", function=model)

    model.uncertainties = get_uncertainties()
    model.constants = get_constants(steps=steps, belief_update_fn=belief_update_fn)
    model.outcomes = get_outcomes()
    model.levers = get_levers()

    return model


def replicator_model_setup(belief_update_fn, steps, replications):
    """
    Sets up a MisinfoPy model for the ema_workbench. Uses the ReplicatorModel.
    @return: MisinfoPy
    """
    # Setting up the seeds
    seeds_100 = get_100_seeds()
    seeds = seeds_100[0:replications]
    seeds = [dict(seed=s) for s in seeds]

    # Setting up the model
    model = MisinfoPy()
    model = ReplicatorModel("MisinfoPy", function=model)
    model.replications = seeds
    model.uncertainties = get_uncertainties()
    model.constants = get_constants(steps=steps, belief_update_fn=belief_update_fn)
    model.outcomes = get_replicator_outcomes()
    model.levers = get_levers()
    # model.levers = get_test_levers()

    return model


def make_sure_path_exists(path):
    """
    Makes sure the directory exists. If the path doesn't exist yet, it is created.
    @param path: string
    """

    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError:
            raise OSError("Creation of the directory failed")


def calculate_quantiles(outcomes_data, outcome, quantiles):
    """
    Calculates the quantile data. Used by seed_analysis.ipynb (I think) # TODO: move to seed_analysis_utils.py
    @param outcomes_data: pd.Dataframe (with 'seed' column)
    @param outcome: str, name of the outcome
    @param quantiles: list of floats, each in range [0.0, 1.0]

    @return: pd.Dataframe,
        col-header: quantile values,
        row-header: quantile was calculated over this many seeds (first n)
    """
    # Preparation
    seeds = outcomes_data["seed"].unique().tolist()
    n_seeds_column = [s for s in range(len(seeds) + 1) if s != 0]
    data_dict = dict.fromkeys(n_seeds_column)

    # Actual calculation
    for n_seeds in n_seeds_column:
        considered_seeds = seeds[0:n_seeds]
        # subsetted dataframe: only the specific 'outcome' column and the 'seed' column
        subset = outcomes_data[[outcome, "seed"]]
        # subsetted dataframe: only the rows of the seeds that should currently be considered
        subset = subset.loc[subset["seed"].isin(considered_seeds)]

        q_data = []
        for q in quantiles:
            q_value = subset[outcome].quantile(q)
            q_value = round(q_value, 2)
            q_data.append(q_value)

        data_dict[n_seeds] = q_data

    quantile_data = pd.DataFrame(data_dict)

    # Transform dataframe (to be clearer and more user-friendly)
    # 1) transposing
    quantile_data = quantile_data.transpose()

    #  2) changing headers (quantiles, n_seeds)
    old_headers = list(quantile_data.columns)
    header_mapping = list(zip(old_headers, quantiles))
    renaming_dict = {}

    for mapping in header_mapping:
        old, new = mapping
        renaming_dict[old] = new

    quantile_data = quantile_data.rename(columns=renaming_dict)

    return quantile_data
