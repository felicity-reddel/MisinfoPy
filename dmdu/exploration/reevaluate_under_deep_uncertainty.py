from dmdu.utils_dmdu import get_uncertainty_names, get_lever_names
import pandas as pd
from ema_workbench import Scenario, Policy
from model.enums import BeliefUpdate
import os
from dmdu.exploration.perform_experiments_replicator_model import perform_my_experiments


def load_policies(bufn_list):
    """
    Loads policies for all bufns in bufn_list (e.g., pareto-optimal policies) from csv file. 
    Returns them in a dictionary.
    
    @param bufn_list: list of BeliefUpdate objects
    @return: policies_dict, dict with {BeliefUpdate: policies}
    """
    policies_dict = {}

    for bufn in bufn_list:
        path = os.path.join(
            os.getcwd(),
            "data",
            "paretosort",
            "output",
            f"experiments_{bufn.name}.csv"
        )
        experiments = pd.read_csv(path)
        experiments = experiments[get_lever_names()]

        # divide lever values by 10 (ema_workbench Integerparam, only use every 10%)
        experiments = experiments.div(10)
        experiments = experiments.astype('int32')

        policy_list = [Policy(f"{idx}", **row) for idx, row in experiments.iterrows()]

        policies_dict[bufn] = policy_list

    return policies_dict


if __name__ == "__main__":

    scenarios = []
    n_scenarios = 50

    beliefs = list(BeliefUpdate)
    policies = load_policies(beliefs)
    dir_path = os.path.join(
        os.getcwd(),
        "data",
        "reevaluation"
    )

    for belief in beliefs:
        experiments, _ = perform_my_experiments(
            policies=policies[belief],
            scenarios=n_scenarios if not scenarios else scenarios,
            belief_update_fn=belief,
            saving=True,
            dir_path=dir_path,
            file_name=f"results_{belief.name}",
        )

        # save scenarios for reuse for other bufns
        if not scenarios:
            x_names = get_uncertainty_names()
            x_cols = experiments[x_names]
            x_cols = x_cols.drop_duplicates()

            int_cols = [
                "belief_metric_threshold",
                "n_edges",
                "mean_normal_user",
                "mean_disinformer",
                "n_posts_estimate_similarity"
            ]

            # adjust Uncertainty params (some need to be ints for ema_workbench)
            for idx, row in x_cols.iterrows():

                scenario_dict = row.to_dict()
                for col_name in int_cols:
                    scenario_dict[col_name] = int(scenario_dict[col_name])

                scenario = Scenario(f"{idx}", **scenario_dict)
                scenarios.append(scenario)
