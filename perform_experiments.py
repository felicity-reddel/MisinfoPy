import pandas as pd
from ema_workbench import (
    Model,
    Policy,
    ema_logging,
    MultiprocessingEvaluator,
    SequentialEvaluator,
    ScalarOutcome,
    IntegerParameter,
    RealParameter,
    Constant,
    save_results
)
from ema_workbench.em_framework.parameters import Category
from misinfo_model import MisinfoPy
from enums import BeliefUpdate

ema_logging.log_to_stderr(ema_logging.INFO)


def perform_my_experiments(policies, scenarios, belief_update_fn, steps=60, saving=False, file_name=None):
    """
    Sets up the model, performs experiments and returns the results.

    @param steps:
    @param belief_update_fn:
    @param scenarios: int or list of scenarios
    @param policies: int or list of policies
    @param saving:
    @param file_name:
    @return:
    """

    # Setting up the model
    model = MisinfoPy()
    model = Model('MisinfoPy', function=model)

    model.uncertainties = [IntegerParameter('belief_metric_threshold', 75, 80),
                           IntegerParameter('n_edges', 2, 3),
                           RealParameter('ratio_normal_user', 0.98, 0.995),  # other parts need this to stay RealParam
                           IntegerParameter('mean_normal_user', 0, 2),
                           IntegerParameter('mean_disinformer', 8, 12),
                           RealParameter('high_media_lit', 0.25, 0.35)]   # other parts need this to stay RealParam

    model.constants = [Constant('steps', steps),
                       Constant('belief_update_fn', belief_update_fn)]
    model.outcomes = get_outcomes()
    model.levers = get_levers()

    # experiments
    with MultiprocessingEvaluator(model) as evaluator:
        results = evaluator.perform_experiments(scenarios=scenarios, policies=policies)

    if saving:
        if file_name is None:
            file_name = f"exploration_{scenarios}_scenarios"
        save_results(results, file_name)

    return results


# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# Helper funs
# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––


def get_outcomes():
    """
    Returns the outcomes. In the fitting format for the ema_workbench.
    @return:
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
    @return: list of CategoricalParameter
    """

    levers = [
        IntegerParameter('mlit_select', lower_bound=0, upper_bound=10),
        IntegerParameter('del_t', lower_bound=0, upper_bound=5),
        IntegerParameter('rank_punish', lower_bound=0, upper_bound=10),
        IntegerParameter('rank_t', lower_bound=0, upper_bound=5),
        IntegerParameter('strikes_t', lower_bound=0, upper_bound=5),
    ]

    return levers


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


if __name__ == "__main__":
    # # (1) [all-off & all-max] Experiment
    # policies = get_policies_all()
    # exp_name = "all"

    # # (2) [indiv-max] Experiment
    # policies = get_policies_indiv()
    # exp_name = "indiv"

    # (3) [open exploration] Experiment
    policies = 50
    exp_name = "sampled"

    scenarios = 50
    beliefs = list(BeliefUpdate)
    steps = 60

    for belief in beliefs:
        perform_my_experiments(policies=policies,
                               scenarios=scenarios,
                               belief_update_fn=belief,
                               steps=steps,
                               saving=True,
                               file_name=f"open_exploration_{exp_name}_{belief.name}")
