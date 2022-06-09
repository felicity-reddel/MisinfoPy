from ema_workbench import Model, ema_logging, MultiprocessingEvaluator, save_results
from model.misinfo_model import MisinfoPy
from model.enums import BeliefUpdate
from dmdu.utils_dmdu import (
    get_constants,
    get_uncertainties,
    get_levers,
    get_outcomes,
    get_policies_all,
    model_setup,
)
import os

ema_logging.log_to_stderr(ema_logging.INFO)


def perform_my_experiments(
    policies,
    scenarios,
    belief_update_fn,
    steps=60,
    saving=False,
    dir_path=None,
    file_name=None,
):
    """
    Sets up the model, performs experiments and returns the results.

    @param steps:               int, number of model steps
    @param belief_update_fn:    BeliefUpdateFn (Enum)
    @param scenarios:           int or list of scenarios
    @param policies:            int or list of policies
    @param saving:              boolean
    @param dir_path:            string
    @param file_name:           string
    @return:
    """

    # Setting up the model
    model = model_setup(belief_update_fn, steps)

    # experiments
    with MultiprocessingEvaluator(model) as evaluator:
        results = evaluator.perform_experiments(scenarios=scenarios, policies=policies)

    if saving:
        if dir_path is None:
            dir_path = os.path.join(os.getcwd(), "dmdu", "exploration", "data")
        if file_name is None:
            file_name = f"exploration_{scenarios}_scenarios"
        path = os.path.join(dir_path, file_name)
        save_results(results, path)

    return results


if __name__ == "__main__":
    # # (1) [all-off & all-max] Experiment
    # policies = get_policies_all()
    # exp_name = "all"

    # # (2) [indiv-max] Experiment
    # policies = get_policies_indiv()
    # exp_name = "indiv"

    # # (3) [open exploration] Experiment
    # policies = 50
    # exp_name = f"{policies}_policies"

    # (4) no-policy Experiment (for picking ref_scenario)
    policies = get_policies_all()[0]
    exp_name = "no_policy"

    scenarios = 100
    beliefs = list(BeliefUpdate)
    steps = 60

    for belief in beliefs:
        perform_my_experiments(
            policies=policies,
            scenarios=scenarios,
            belief_update_fn=belief,
            steps=steps,
            saving=True,
            file_name=f"open_exploration_{exp_name}_{belief.name}",
        )
