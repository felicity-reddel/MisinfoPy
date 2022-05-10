from ema_workbench import (
    Model,
    Policy,
    ema_logging,
    MultiprocessingEvaluator,
    ScalarOutcome,
    CategoricalParameter,
    RealParameter
)
from ema_workbench.em_framework.parameters import Category

from misinfo_model import MisinfoPy


def perform_my_experiments(n_scenarios=0, n_policies=10):
    """
    Sets up the model, performs experiments and returns the results.

    @param n_scenarios: int
    @param n_policies: int,  # TODO: SPECIFY FURTHER, how to make all policies?
    @return:
    """

    # Setting up the model
    model = MisinfoPy()
    model = Model('MisinfoPy', function=model)

    model.uncertainties = []
    model.constants = []  # TODO: FILL
    model.outcomes = get_outcomes()
    model.levers = get_levers()

    # experiments
    with MultiprocessingEvaluator(model) as evaluator:
        results = evaluator.perform_experiments(scenarios=n_scenarios, policies=n_policies)

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
    # standard_categories = [
    #     Category('0.0', 0.0),
    #     Category('0.2', 0.2),
    #     Category('0.4', 0.4),
    #     Category('0.6', 0.6),
    #     Category('0.8', 0.8),
    #     Category('1.0', 1.0),
    # ]
    #
    # negative_categories = [
    #     Category('-0.0', -0.0),
    #     Category('-0.2', -0.2),
    #     Category('-0.4', -0.4),
    #     Category('-0.6', -0.6),
    #     Category('-0.8', -0.8),
    #     Category('-1.0', -1.0),
    # ]
    #
    # levers = [
    #     CategoricalParameter('mlit_select', standard_categories),
    #     CategoricalParameter('del_t', standard_categories),
    #     CategoricalParameter('rank_punish', negative_categories),
    #     CategoricalParameter('rank_t', standard_categories),
    #     CategoricalParameter('strikes_t', standard_categories),
    # ]

    levers = [
        RealParameter('mlit_select', 0.0, 1.0),
        RealParameter('del_t', 0.0, 1.0),
        RealParameter('rank_punish', -1.0, -0.0),
        RealParameter('rank_t', 0.0, 1.0),
        RealParameter('strikes_t', 0.0, 1.0),
    ]

    return levers


if __name__ == "__main__":
    perform_my_experiments()
