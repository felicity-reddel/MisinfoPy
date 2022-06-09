# Project
from model.misinfo_model import MisinfoPy
from dmdu.utils_dmdu import (
    get_uncertainties,
    get_constants,
    get_outcomes,
    get_levers,
    make_sure_path_exists,
    get_100_seeds,
)
from model.enums import BeliefUpdate

# General
import pandas as pd
import os

# ema_workbench
from ema_workbench import (
    Model,
    Constant,
    MultiprocessingEvaluator,
    save_results,
    SequentialEvaluator,
)


if __name__ == "__main__":

    test_version = True
    saving = True

    if test_version:
        n_experiments = 4
        models = [BeliefUpdate.SIT]
        seeds = [
            577747,
            914425,
            445063,
            977049,
            617127,
            639676,
            137294,
            845058,
            718814,
            119679,
        ]  # 10 seeds
    else:
        # Params for real version
        n_experiments = 400
        models = list(BeliefUpdate)
        seeds = get_100_seeds()

    # Gather data for each model individually
    for belief_update_fn in models:

        # Datastructure to gather dataframes
        all_experiments = []
        all_outcomes = []

        for seed in seeds:
            # Setting up the model
            model = MisinfoPy()
            model = Model("MisinfoPy", function=model)
            steps = 60

            # Combining uncertainties and levers (for better sampling coverage)
            model.uncertainties = get_uncertainties() + get_levers()
            model.constants = get_constants(
                steps=steps, belief_update_fn=belief_update_fn
            ) + [Constant("seed", seed)]
            model.outcomes = get_outcomes()
            model.levers = []

            # Experiments
            with MultiprocessingEvaluator(model) as evaluator:
                results = evaluator.perform_experiments(n_experiments)
            experiments, outcomes = results

            # Add seed column to outcomes
            outcomes = pd.DataFrame(outcomes)
            outcomes["seed"] = seed

            # Save results from this seed
            all_experiments.append(experiments)
            all_outcomes.append(outcomes)

        # Combine dataframes of different seeds
        experiments = pd.concat(all_experiments)
        outcomes = pd.concat(all_outcomes)

        # Save data
        if saving:
            # Path
            dir_path = os.path.join(
                os.getcwd(), "data", "seedanalysis", belief_update_fn.name
            )
            make_sure_path_exists(dir_path)
            experiments_path = os.path.join(dir_path, f"experiments.csv")
            outcomes_path = os.path.join(dir_path, f"outcomes.csv")

            # Save to csv
            experiments.to_csv(experiments_path)
            outcomes.to_csv(outcomes_path)
