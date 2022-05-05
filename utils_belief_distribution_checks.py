def run_experiment(medlit_select, rank_punish, del_t, rank_t, strikes_t, belief_update, n_agents=1000,
                   ratio_normal_user=0.99, n_steps=60, n_repl=9):
    """
    Performs the requested experiments and returns the results wrt the belief distributions (before, after).

    # Lever values (for policies)
    @param medlit_select: float (in range [0,1], %of agents selected for media literacy intervention)
    @param rank_punish: float (in range [-1,-0], visibility may be reduced by this %)
    @param del_t: float in range [-1,-0], p_true_treshold: if below -> may be deleted
    @param rank_t: float in range [-1,-0], p_true_treshold: if below -> may be down-ranked
    @param strikes_t: float in range [-1,-0], p_true_treshold: if below -> may be punished with a strike

    # Other params
    @param belief_update: BeliefUpdate
    @param n_agents: int
    @param ratio_normal_user: float in range [0,1], ratio of NormalUser
    @param n_steps: int, number of steps per model run
    @param n_repl: int, number of replications

    @return: DataFrame (col: policy, row: replication)
    """

    # Set up data structure (col: policy, row: replication)
    df_column = []

    for _ in range(n_repl):
        # Set up the model
        model = MisifoPy()  # TODO, all params, incl. lever values

        # Run experiment (1 replication)
        before = [agent.beliefs[Topic.VAX] for agent in model.schedule.agents]
        model()  # TODO: param n_steps
        after = [agent.beliefs[Topic.VAX] for agent in model.schedule.agents]

        # Save data from 1 replication
        repl_data = (before, after)
        df_column.append(repl_data)

    return df_column


