from ema_workbench import (
    Policy,
    ScalarOutcome,
    IntegerParameter,
    RealParameter,
    Constant,
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
                     RealParameter('high_media_lit', 0.25, 0.35)]  # other parts need this to stay RealParam
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
