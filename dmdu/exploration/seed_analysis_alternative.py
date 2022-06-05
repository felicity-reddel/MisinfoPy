# Project
from model.misinfo_model import MisinfoPy
from dmdu.utils_dmdu import (get_uncertainties, get_constants, get_outcomes,
                             get_levers, make_sure_path_exists)
from model.enums import BeliefUpdate

# General
import pandas as pd
import os

# ema_workbench
from ema_workbench import (Model, Constant, MultiprocessingEvaluator, ArrayOutcome,
                           save_results, SequentialEvaluator, ReplicatorModel)

if __name__ == "__main__":
    n_experiments = 400
    only_one_model = False
    saving = True

    # 100 seeds
    # seeds = [577747, 914425, 445063, 977049, 617127, 639676, 137294, 845058, 718814, 119679, 435223, 347541, 666852,
    #          701324, 604437, 908374, 941595, 800210, 745388, 399447, 140918, 910967, 917428, 497096, 222919, 726572,
    #          748497, 185669, 610661, 709441, 801330, 506120, 891889, 298223, 164318, 929955, 854094, 553307, 279254,
    #          597549, 223105, 708080, 220244, 126086, 634792, 458729, 822070, 972244, 751076, 130675, 100289, 252061,
    #          262114, 449996, 206219, 764775, 285626, 385767, 111989, 812234, 305433, 822474, 312966, 877990, 598853,
    #          389796, 777981, 937667, 943990, 393412, 913947, 594493, 543410, 199872, 519301, 577412, 615253, 914266,
    #          136560, 705707, 433804, 414487, 198043, 325188, 906659, 507433, 268008, 894819, 994630, 427593, 129353,
    #          207160, 780566, 131963, 158586, 428856, 485180, 445734, 806806, 958623]

    models = [BeliefUpdate.SIT] if only_one_model else list(BeliefUpdate)
    seeds = [577747, 914425]
    seeds = [dict(seed=entry) for entry in seeds]  # ReplicatorModel alternative

    # Gather data for each model individually
    for belief_update_fn in models:

        # Datastructure to gather dataframes
        all_experiments = []
        all_outcomes = []

        # Setting up the model
        model_function = MisinfoPy()
        model = ReplicatorModel('MisinfoPy', function=model_function)  # ReplicatorModel alternative
        model.replications = seeds  # ReplicatorModel alternative
        steps = 60

        # Combining uncertainties and levers (for better sampling coverage)
        model.uncertainties = get_uncertainties() + get_levers()
        model.constants = get_constants(steps=steps, belief_update_fn=belief_update_fn)
        model.outcomes = [ArrayOutcome('n_agents_above_belief_threshold'),  # ReplicatorModel alternative
                          ArrayOutcome('polarization_variance'),
                          ArrayOutcome('engagement'),
                          ArrayOutcome('free_speech_constraint'),
                          ArrayOutcome('avg_user_effort')]
        model.levers = []

        # Experiments
        with MultiprocessingEvaluator(model) as evaluator:
            results = evaluator.perform_experiments(n_experiments)
        experiments, outcomes = results

        # Add seed column to outcomes
        outcomes = pd.DataFrame(outcomes)
        # outcomes['seed'] = seed  # ReplicatorModel alternative

        # Save results from this seed
        all_experiments.append(experiments)
        all_outcomes.append(outcomes)

        # # Combine dataframes of different seeds
        # experiments = pd.concat(all_experiments)
        # outcomes = pd.concat(all_outcomes)

        # Save data
        if saving:
            # Path
            dir_path = os.path.join(os.getcwd(), 'data', 'seedanalysis', belief_update_fn.name)
            make_sure_path_exists(dir_path)
            experiments_path = os.path.join(dir_path, f"experiments.csv")
            outcomes_path = os.path.join(dir_path, f"outcomes.csv")

            # Save to csv
            experiments.to_csv(experiments_path)
            outcomes.to_csv(outcomes_path)

# Thoughts on ReplicatorModel alternative:
# Are the main benefits of it:
#   - that the seed doesn't need to be passed as a constant, and
#   - that the seed doesn't need to be explicitly saved to the outcomes?
