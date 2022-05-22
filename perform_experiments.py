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
from utils_dmdu import (
    get_constants,
    get_uncertainties,
    get_levers,
    get_outcomes,
    get_policies_all,
    get_policies_indiv
)

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

    model.uncertainties = get_uncertainties()
    model.constants = get_constants()
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
