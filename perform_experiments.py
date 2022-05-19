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

ema_logging.log_to_stderr(ema_logging.INFO)


def perform_my_experiments(policies, scenarios, saving=False):
    """
    Sets up the model, performs experiments and returns the results.

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

    model.constants = []
    model.outcomes = get_outcomes()
    model.levers = get_levers()

    # experiments
    with MultiprocessingEvaluator(model) as evaluator:
        results = evaluator.perform_experiments(scenarios=scenarios, policies=policies)

    if saving:
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

    # Currently, as RealParameter (to make sure that issue is not
    # with the potentially incorrect use of CategoricalParameter)
    levers = [
        RealParameter('mlit_select', 0.0, 1.0),
        RealParameter('del_t', 0.0, 1.0),
        RealParameter('rank_punish', -1.0, -0.0),
        RealParameter('rank_t', 0.0, 1.0),
        RealParameter('strikes_t', 0.0, 1.0),
    ]

    # # TODO: Move to IntegerParameter (to not do full exploration, but only specified values)
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

    return levers


if __name__ == "__main__":

    policy_list = [
        Policy('all off', **{'mlit_select': 0.0,
                             'del_t': 0.0,
                             'rank_punish': -0.0,
                             'rank_t': 0.0,
                             'strikes_t': 0.0}),
        Policy('all max', **{'mlit_select': 1.0,
                             'del_t': 0.5,
                             'rank_punish': -1.0,
                             'rank_t': 0.5,
                             'strikes_t': 0.5}),
    ]

    res = perform_my_experiments(policies=policy_list, scenarios=10, saving=True)
    # exp, out = res
    #
    # out = pd.DataFrame(out)
    #
    # for idx, row in out.iterrows():
    #     print(row)
    #     print()
