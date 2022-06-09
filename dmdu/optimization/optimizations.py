# Project
from model.misinfo_model import MisinfoPy
from model.enums import BeliefUpdate
from dmdu.utils_dmdu import model_setup, get_epsilons, make_sure_path_exists

# General
import os

# ema_workbench
from ema_workbench.em_framework.optimization import EpsilonProgress, ArchiveLogger
from ema_workbench import Model, MultiprocessingEvaluator, ema_logging

ema_logging.log_to_stderr(ema_logging.INFO)


def run_my_optimization(bufn, nfe=100000, steps=60, saving=True, dir_path=None):

    # Setting up central components
    model = model_setup(belief_update_fn=bufn, steps=steps)
    epsilons = get_epsilons()
    epsilon_progress = [EpsilonProgress()]

    # Running the optimization
    with MultiprocessingEvaluator(model) as evaluator:
        results, epsilon_progress = evaluator.optimize(
            nfe=nfe, epsilons=epsilons, convergence=epsilon_progress
        )

    # Saving the data
    if saving:

        # Create path name
        if dir_path is None:
            dir_path = os.path.join(os.getcwd(), "data", bufn.name)
        make_sure_path_exists(dir_path)

        results_path = os.path.join(dir_path, f"results.csv")
        epsilon_progress_path = os.path.join(dir_path, f"epsilon_progress.csv")

        # Save data
        results.to_csv(results_path)
        epsilon_progress.to_csv(epsilon_progress_path)


if __name__ == "__main__":
    # Running only one model
    # -> save computation, but still get a rough feeling of the convergence
    only_one_model = True
    nfe = 150000

    models = [BeliefUpdate.SIT] if only_one_model else list(BeliefUpdate)

    for belief_update_fn in models:
        print(f"Starting optimization for {belief_update_fn.name}")
        run_my_optimization(bufn=belief_update_fn, nfe=nfe, saving=True)
