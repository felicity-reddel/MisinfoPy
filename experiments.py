import itertools
import os
import pandas as pd
import time
from misinfo_model import *


if __name__ == '__main__':

    n_agents = 50  # 1000
    n_edges = 3
    max_run_length = 60
    n_replications = 9  # 12

    # Scenarios are different agent_ratios
    scenarios = [{NormalUser.__name__: 0.99, Disinformer.__name__: 0.01},
                 # {NormalUser.__name__: 0.95, Disinformer.__name__: 0.05},
                 # {NormalUser.__name__: 0.8, Disinformer.__name__: 0.2},
                 # {NormalUser.__name__: 0.25, Disinformer.__name__: 0.75}
                 ]

    # Experiment-Conditions are a combination of: Policies + BeliefUpdateFn
    # (Policies themselves = combinations of intervention values)
    media_literacy_intervention_values = [(0.0, SelectAgentsBy.RANDOM),
                                          # (0.1, SelectAgentsBy.RANDOM),
                                          # (0.25, SelectAgentsBy.RANDOM)
                                          ]

    ranking_visibility_adjustment_values = [-0.0]  # by default no ranking adjustment
    p_true_threshold_deleting_values = [0.0]  # by default no deleting
    p_true_threshold_ranking_values = [0.0]  # by default no ranking
    p_true_threshold_strikes_values = [0.0]  # by default no strike system
    belief_update_fn_values = list(BeliefUpdate)

    policies = list(itertools.product(media_literacy_intervention_values,
                                      ranking_visibility_adjustment_values,
                                      p_true_threshold_deleting_values,
                                      p_true_threshold_ranking_values,
                                      p_true_threshold_strikes_values))

    print("Policies:")
    for policy in policies:
        print(f'– {policy}')

    # Printing
    start_time_seconds = time.time()
    start_time = time.localtime(time.time())
    human_understandable_time = time.strftime('%Y-%m-%d %H:%M:%S', start_time)
    print(f"\nStarting at time: {human_understandable_time}")

    # Run Experiments
    for belief_update_fn in belief_update_fn_values:
        for i, scenario in enumerate(scenarios):  # Each scenario is 1 ratio of agent types
            # Set up data structures
            data = pd.DataFrame({"Replication": list(range(0, n_replications))})
            engagement = {}

            for j, policy in enumerate(policies):
                # Unpack policy (over multiple lines -> into a list)
                [media_literacy_intervention,
                 ranking_visibility_adjustment,
                 p_true_threshold_deleting,
                 p_true_threshold_ranking,
                 p_true_threshold_strikes] = policy

                # Set up data structure (col: policy, row: replication)
                df_column = []

                replications_n_seen_posts = []
                for replication in range(n_replications):
                    # Set up the model
                    model = MisinfoPy(
                        # ––– Network –––
                        n_agents=n_agents,
                        n_edges=n_edges,
                        agent_ratio=scenario,

                        # # ––– Posting behavior –––
                        # sigma=0.7,
                        # mean_normal_user=1,
                        # mean_disinformer=10,
                        # adjustment_based_on_belief=2,

                        # ––– Levers –––
                        media_literacy_intervention=media_literacy_intervention,
                        media_literacy_intervention_durations=None,
                        ranking_visibility_adjustment=ranking_visibility_adjustment,
                        p_true_threshold_deleting=p_true_threshold_deleting,
                        p_true_threshold_ranking=p_true_threshold_ranking,
                        p_true_threshold_strikes=p_true_threshold_strikes,

                        # ––– Belief updating behavior –––
                        belief_update_fn=belief_update_fn,
                        # sampling_p_update=0.02,
                        # deffuant_mu=0.02,
                    )

                    # Save start data
                    agents_belief_before = [agent.beliefs[Topic.VAX] for agent in model.schedule.agents]

                    # Run the model
                    for tick in range(max_run_length):
                        model.step()

                    # Save end data
                    agents_belief_after = [agent.beliefs[Topic.VAX] for agent in model.schedule.agents]
                    # save data from this replication
                    replication_data = (agents_belief_before, agents_belief_after)
                    df_column.append(replication_data)

                    # replications_n_seen_posts.append(model.get_total_seen_posts())
                    # Printing
                    print(f"replication {replication} done")

                # Create policy columns
                policy_column = pd.Series(df_column, name=str(policy))
                # Save policy column into the dataframe
                data = data.join(policy_column)

                # engagement[policy] = replications_n_seen_posts

                # Save scenario data into a csv file
                directory = os.getcwd()
                path = directory + '/results/'

                file_name = "belief_distr_" + belief_update_fn.name + str(scenario) + ".csv"
                # noinspection PyTypeChecker
                data.to_csv(path + file_name)

                # Printing
                print(f"policy {j} done: {policy}")
            print(f"Belief Update Function {belief_update_fn.name} done")

        # for lever_combination, n_seen_posts_repl in engagement.items():
        #     print(f"Ranking: {lever_combination[1]}, "
        #           f"n's: {n_seen_posts_repl}, "
        #           f"avg: {sum(n_seen_posts_repl) / len(n_seen_posts_repl)}")

    # # Printing
    # end_time = time.localtime(time.time())
    # human_understandable_time = time.strftime('%Y-%m-%d %H:%M:%S', end_time)
    # print(f"Ending at time: {human_understandable_time}")
    # run_time = round(time.time() - start_time_seconds, 2)
    # print(
    #     f"\nWith {max_run_length} steps, runtime is {run_time} seconds "
    #     f"--> roughly {round(run_time / 60 / 60, 2)} hours")
