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
    @return:
    """

    # Setting up the model
    model = MisinfoPy()
    model = Model('MisinfoPy', function=model)

    model.uncertainties = [RealParameter('belief_metric_threshold', 40.0, 60.0),
                           IntegerParameter('n_edges', 2, 3),
                           RealParameter('ratio_normal_user', 0.95, 1.0),
                           IntegerParameter('mean_normal_user', 0, 2),
                           IntegerParameter('mean_disinformer', 5, 15),
                           RealParameter('high_media_lit', 0.05, 0.4)]

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

    # Currently, as RealParameter
    # levers = [
    #     RealParameter('mlit_select', 0.0, 1.0),
    #     RealParameter('del_t', 0.0, 1.0),
    #     RealParameter('rank_punish', -1.0, -0.0),
    #     RealParameter('rank_t', 0.0, 1.0),
    #     RealParameter('strikes_t', 0.0, 1.0),
    # ]

    levers = [  # TODO: FIRST ADJUST MODEL TO BE ABLE TO USE INTEGERS! RESCALE.
        IntegerParameter('mlit_select', lower_bound=0, upper_bound=4),
        IntegerParameter('del_t', lower_bound=0, upper_bound=4),
        IntegerParameter('rank_punish', lower_bound=0, upper_bound=4),
        IntegerParameter('rank_t', lower_bound=0, upper_bound=4),
        IntegerParameter('strikes_t', lower_bound=0, upper_bound=4),

        # IntegerParameter('del_t', 0, 100, resolution=[5]),
        # IntegerParameter('rank_punish', -100, 0, resolution=[-5]),
        # IntegerParameter('rank_t', 0, 100, resolution=[5]),
        # IntegerParameter('strikes_t', 0, 100, resolution=[5])
    ]

    return levers


if __name__ == "__main__":

    # policy_list = [
    #     Policy('all off', **{'mlit_select': 0,
    #                          'del_t': 0,
    #                          'rank_punish': 0,
    #                          'rank_t': 0,
    #                          'strikes_t': 0}),
    #     Policy('all max', **{'mlit_select': 4,
    #                          'del_t': 2,
    #                          'rank_punish': 4,
    #                          'rank_t': 2,
    #                          'strikes_t': 2}),
    # ]

    # res = perform_my_experiments(policies=policy_list, scenarios=10, saving=True)
    policies = 50
    scenarios = 50
    steps = 60

    beliefs = list(BeliefUpdate)

    for belief in beliefs:
        perform_my_experiments(policies=policies,
                               scenarios=scenarios,
                               belief_update_fn=belief,
                               steps=steps,
                               saving=True,
                               file_name=f"open_exploration_{belief.name}")

