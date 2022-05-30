# –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# From when the ratio was ratio_normal_user dictionary of agent types:
# –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

# # misinfo_model.py // MisinfoPy init
# if ratio_normal_user is None:
#     ratio_normal_user = {NormalUser.__name__: 0.9, Disinformer.__name__: 0.1}
# # Making sure that the agent ratios add up to 1.0
# if sum(ratio_normal_user.values()) != 1.0:
#     raise ValueError(f"The agent ratios add up to {sum(ratio_normal_user.values())}, "
#                      f"while they should add up to 1.0.")

# # misinfo_model.py // init_agents
# def init_agents(self, ratio_normal_user):
#     """Initializes the agents.
#     @param ratio_normal_user: dictionary, {String: float}
#     """
#
#     # Saving scenario
#     types = []
#     percentages = []
#     for agent_type, percentage in ratio_normal_user.items():
#         types.append(agent_type)
#         percentages.append(percentage)
#
#     # Create agents & add them to the scheduler
#     for i in range(self.n_agents):
#
#         # Pick which type should be added
#         agent_type = random.choices(population=types, weights=percentages, k=1)[0]
#
#         # Add agent of that type
#         if agent_type is NormalUser.__name__:
#             a = NormalUser(i, self)
#             self.schedule.add(a)
#         elif agent_type is Disinformer.__name__:
#             a = Disinformer(i, self)
#             self.schedule.add(a)
#
#     # Place each agent in its node. (& save node_position into agent)
#     for node in self.G.nodes:  # each node is just an integer (i.e., a node_id)
#         agent = self.schedule.agents[node]
#
#         # save node_position into agent
#         self.grid.place_agent(agent, node)
#
#         # add agent to node
#         self.G.nodes[node]['agent'] = agent

# –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––


# –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# utils_belief_distribution_checks.py
# –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

# MULTIPLE EXPERIMENTS
# def run_experiments(mlit_select, rank_punish, del_t, rank_t, strikes_t, belief_update, n_agents=1000,
#                     ratio_normal_user=0.99, n_steps=60, n_repl=9):
#     """
#     Performs the requested experiments and returns the results wrt the belief distributions (before, after).
#
#     # Lever values (for policies)
#     @param mlit_select: list of floats (each in range [0,1], %of agents selected for media literacy intervention)
#     @param rank_punish: list of floats (each in range [-1,-0], visibility may be reduced by this %)
#     @param del_t: float in range [-1,-0], p_true_treshold: if below -> may be deleted
#     @param rank_t: float in range [-1,-0], p_true_treshold: if below -> may be down-ranked
#     @param strikes_t: float in range [-1,-0], p_true_treshold: if below -> may be punished with a strike
#
#     # Other params
#     @param belief_update: BeliefUpdate
#     @param n_agents: int
#     @param ratio_normal_user: float in range [0,1], ratio of NormalUser
#     @param n_steps: int, number of steps per model run
#     @param n_repl: int, number of replications
#
#     @return: DataFrame (col: policy, row: replication)
#     """
#
#     policies = list(itertools.product(mlit_select, rank_punish, del_t, rank_t, strikes_t))
#
#     for policy in policies:
#         # Unpack policy (over multiple lines -> into a list)
#         [mlit_select, rank_punish, del_t, rank_t, strikes_t] = policy
#
#         # Set up data structure (col: policy, row: replication)
#         df_column = []
#
#         for _ in range(n_repl):
#             # Set up the model
#             model = MisifoPy()
#
#             # Run experiment (1 replication)
#             before = [agent.beliefs[Topic.VAX] for agent in model.schedule.agents]
#             model(n_steps)
#             after = [agent.beliefs[Topic.VAX] for agent in model.schedule.agents]
#
#             # Save data from 1 replication
#             repl_data = (before, after)
#             df_column.append(repl_data)
#
#         return df_column  # Right indentation?
# –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
