from misinfo_model import *

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from matplotlib.ticker import MaxNLocator


def run_experiment(mlit_select, rank_punish, del_t, rank_t, strikes_t, belief_update, n_agents=1000,
                   ratio_normal_user=0.99, n_steps=60, n_repl=9):
    """
    Performs the requested experiments and returns the results wrt the belief distributions (before, after).

    # Lever values (for policies)
    @param mlit_select: float (in range [0.0, 1.0], %of agents selected for media literacy intervention)
    @param rank_punish: float (in range [-0.0, -1.0], visibility may be reduced by this %)
    @param del_t: float in range [0.0, 1.0], p_true_treshold: if below -> may be deleted
    @param rank_t: float in range [0.0, 1.0], p_true_treshold: if below -> may be down-ranked
    @param strikes_t: float in range [0.0, 1.0], p_true_treshold: if below -> may be punished with a strike

    # Other params
    @param belief_update: BeliefUpdate
    @param n_agents: int
    @param ratio_normal_user: float in range [0,1], ratio of NormalUser
    @param n_steps: int, number of steps per model run
    @param n_repl: int, number of replications

    @return: DataFrame (col: policy, rows: replication)
    """

    # Set up data structure (1 col: policy, row: replication)
    df_column = []

    for _ in range(n_repl):
        # Set up the model
        model = MisinfoPy(mlit_select=mlit_select, rank_punish=rank_punish, del_t=del_t, rank_t=rank_t,
                          strikes_t=strikes_t, belief_update_fn=belief_update,
                          n_agents=n_agents, ratio_normal_user=ratio_normal_user)

        # Run experiment (1 replication)
        before = [agent.beliefs[Topic.VAX] for agent in model.schedule.agents]
        model(steps=n_steps)
        after = [agent.beliefs[Topic.VAX] for agent in model.schedule.agents]

        # Save data from 1 replication
        repl_data = (before, after)
        df_column.append(repl_data)

    return df_column


def create_subplot(data, n_rows=3, n_cols=3, saving=False, title="", x_ticks=5, y_ticks=5, y_max=None):
    """
    Plots and potentially save a subplot of the requested policy and number of subplots (via nrows & ncols).

    @param data: list,      [(replication1), (replication2), ...],   replication = (before, after)
    @param n_rows: int,     number of subplots on vertical axis-plot
    @param n_cols: int,     number of subplots on horizontal axis-plot
    @param saving: boolean, whether to save the plot
    @param title: String,   title of the whole plot
    @param x_ticks: int,    number of grid lines in a subplot
    @param y_ticks: int,    number of grid lines in a subplot
    @param y_max: int,      maximal y-axis value -> all plots have same y-axis scale
    """

    # Sample which runs to include in the plot
    n_runs = len(data)
    all_run_ids = [*range(n_runs)]
    plot_ids = random.sample(all_run_ids, k=n_rows*n_cols)  # outside of model performance, can use non-model random

    # Create subplots
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, sharex="all", sharey="all", figsize=(10.8, 8))

    for row in range(n_rows):
        for col in range(n_cols):

            # Plot 1 hist (incl. before & after)
            idx = plot_ids.pop()
            repl_data = data[idx]
            before, after = repl_data
            sns.histplot(data=before, color="skyblue", label="Before", kde=True, bins=25, binrange=(0, 100),
                         ax=axes[row, col])
            sns.histplot(data=after, color="red", label="After", kde=True, bins=25, binrange=(0, 100),
                         ax=axes[row, col])
            if y_max is not None:
                plt.ylim(0, y_max)

            # Depict information (labels etc)
            # - replication number
            anc = AnchoredText(f'Run {idx}', loc="upper center", frameon=False)
            axes[row, col].add_artist(anc)
            # - y axis
            if col == 0:
                axes[row, col].set_ylabel("Agent count")
                axes[row, col].yaxis.set_major_locator(MaxNLocator(y_ticks))
            # - x axis
            if row == n_rows - 1:
                axes[row, col].set_xlabel("Belief that vaccines are good")
                axes[row, col].xaxis.set_major_locator(MaxNLocator(x_ticks))
            # - legend
            if row == 0 and col == n_cols - 1:
                axes[row, col].legend(fontsize='large', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

    # Title for whole plot
    plt.suptitle(f"\n{title}", fontsize='x-large')

    # Potentially save plot
    if saving:
        directory = os.getcwd()
        root_directory = os.path.dirname(directory)
        images_folder = root_directory + '/results/images/'
        fig.savefig(images_folder + f"belief_distributions_{title}.png", dpi=200, bbox_inches="tight", pad_inches=0.1)
