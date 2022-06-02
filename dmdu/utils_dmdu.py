from model.misinfo_model import MisinfoPy
import os
from ema_workbench import (
    Policy,
    ScalarOutcome,
    IntegerParameter,
    RealParameter,
    Constant,
    Model
)

# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# Get main inputs for DMDU
# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––


def get_constants(steps, belief_update_fn):
    """
    Returns the constants. In the fitting format for the ema_workbench.
    @param steps: int, number of model steps
    @param belief_update_fn: BeliefUpdateFn (Enum)
    @return: list of ema_workbench Constants
    """
    constants = [Constant('steps', steps),
                 Constant('belief_update_fn', belief_update_fn)]

    return constants


def get_uncertainties():
    """
    Returns the outcomes. In the fitting format for the ema_workbench.
    @return: list of ema_workbench Parameters
    """
    uncertainties = [IntegerParameter('belief_metric_threshold', 75, 80),
                     IntegerParameter('n_edges', 2, 3),
                     RealParameter('ratio_normal_user', 0.98, 0.995),  # other parts need this to stay RealParam
                     IntegerParameter('mean_normal_user', 0, 2),
                     IntegerParameter('mean_disinformer', 8, 12),
                     RealParameter('high_media_lit', 0.25, 0.35),   # other parts need this to stay RealParam

                     RealParameter('deffuant_mu', 0.01, 0.03),  # DEFFUANT-specific
                     RealParameter('sampling_p_update', 0.01, 0.03),  # SAMPLING-specific
                     IntegerParameter('n_posts_estimate_similarity', 5, 15)]  # SIT-specific
    return uncertainties


def get_outcomes():
    """
    Returns the outcomes. In the fitting format for the ema_workbench.
    @return: list of ema_workbench Outcomes
    """
    outcomes = [
        ScalarOutcome('n_agents_above_belief_threshold', ScalarOutcome.MAXIMIZE),
        ScalarOutcome('polarization_variance', ScalarOutcome.MINIMIZE),
        ScalarOutcome('engagement', ScalarOutcome.MAXIMIZE),
        ScalarOutcome('free_speech_constraint', ScalarOutcome.MINIMIZE),
        ScalarOutcome('avg_user_effort', ScalarOutcome.MINIMIZE)
    ]

    return outcomes


def get_epsilons():
    """
    Returns the epsilon values for all outcomes. In the same order as the outcomes list.
    These values stem from the analysis in epsilon_values.ipynb.

    @return: list of floats
    """
    epsilons = [1.0, 1.0, 10.0, 0.01, 0.5]

    return epsilons


def get_levers():
    """
    Returns the levers. In the fitting format for the ema_workbench.
    @return: list of ema_workbench Parameters
    """

    levers = [
        IntegerParameter('mlit_select', lower_bound=0, upper_bound=10),
        IntegerParameter('del_t', lower_bound=0, upper_bound=5),
        IntegerParameter('rank_punish', lower_bound=0, upper_bound=10),
        IntegerParameter('rank_t', lower_bound=0, upper_bound=5),
        IntegerParameter('strikes_t', lower_bound=0, upper_bound=5),
    ]

    return levers


# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# Get different prepared sets of policies
# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––


def get_policies_all():
    """Returns list of 2 Policy objects: all-off and all-max."""

    policy_list = [
        Policy('all off', **{'mlit_select': 0,
                             'del_t': 0,
                             'rank_punish': 0,
                             'rank_t': 0,
                             'strikes_t': 0}),
        Policy('all max', **{'mlit_select': 10,
                             'del_t': 5,
                             'rank_punish': 10,
                             'rank_t': 5,
                             'strikes_t': 5}),
    ]

    return policy_list


def get_policies_indiv():
    """
    Returns list of 5 Policy objects:
    Each individual lever maximized while the other levers are on a very low value.
    """

    policy_list = [
        Policy('mlit_select max', **{'mlit_select': 10,
                                     'del_t': 0,
                                     'rank_punish': 0,
                                     'rank_t': 0,
                                     'strikes_t': 0}),
        Policy('del_t max', **{'mlit_select': 0,
                               'del_t': 5,
                               'rank_punish': 0,
                               'rank_t': 0,
                               'strikes_t': 0}),
        Policy('rank_punish max', **{'mlit_select': 0,
                                     'del_t': 0,
                                     'rank_punish': 10,
                                     'rank_t': 1,
                                     'strikes_t': 0}),
        Policy('rank_t max', **{'mlit_select': 0,
                                'del_t': 0,
                                'rank_punish': 1,
                                'rank_t': 5,
                                'strikes_t': 0}),
        Policy('double rank max', **{'mlit_select': 0,
                                     'del_t': 0,
                                     'rank_punish': 10,
                                     'rank_t': 5,
                                     'strikes_t': 0}),
        Policy('strikes_t max', **{'mlit_select': 0,
                                   'del_t': 0,
                                   'rank_punish': 0,
                                   'rank_t': 0,
                                   'strikes_t': 5})
    ]

    return policy_list


def epsilon_helper(outcomes, bufn, metric, divide_by=10, best_quantile=0.25, minimize=None):
    """
    Helps to explore which epsilon-values would be suitable.

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
        minimize = ['polarization_variance', 'free_speech_constraint', 'avg_user_effort']

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
    Sets up a MisinfoPy model for the ema_workbench.
    @return: MisinfoPy
    """

    # Setting up the model
    model = MisinfoPy()
    model = Model('MisinfoPy', function=model)

    model.uncertainties = get_uncertainties()
    model.constants = get_constants(steps=steps, belief_update_fn=belief_update_fn)
    model.outcomes = get_outcomes()
    model.levers = get_levers()

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
