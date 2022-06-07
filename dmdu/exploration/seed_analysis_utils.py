import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def get_data(belief_update_fn, metric):
    """
    Loading and preparing the requested data.
    @param belief_update_fn: BeliefUpdate
    @param metric: string, from ['avg_user_effort', 'engagement', 'free_speech_constraint', 'n_agents_above_belief_threshold', 'polarization_variance']
    @return: pd.Dataframe
    """
    path = os.path.join(os.getcwd(), 'data', 'seedanalysis', belief_update_fn.name, 'quantiles')
    file_name = '/' + metric + '.csv'
    metric_data = pd.read_csv(path + file_name)
    metric_data = metric_data.rename(columns={'Unnamed: 0': 'n_seeds'})

    # add belief_update_fn column?
    metric_data['belief_update_fn'] = belief_update_fn.name

    return metric_data


def plot_quartiles(data, metric, title=None, y_min=0, y_max=600, fig_size=(10, 7), x_tick_freq=5):
    """
    # TODO: FILL THIS
    @param data:
    @param metric:
    @param title:
    @param y_min:
    @param y_max:
    @param fig_size:
    @param x_tick_freq:
    @return:
    """
    # only quartile data
    data = data[['n_seeds','0.25', '0.5', '0.75', 'belief_update_fn']]

    # TODO: Decide wrt below attempt of making the y_max programmatic
    # set y-axis max
    # if y_max is None:
        # curr_max = 0
        # for col_name in data.columns:
        #     new_max = data[col_name].max()
        #     print(f'new_max: {type(new_max), new_max} \n '
        #           f'curr_max: {type(curr_max)}')
        #     if not isinstance(new_max, str):  # ugly workaround because at some point, new_max is 'SIT'?!
        #         if new_max > curr_max:
        #             curr_max = new_max
        # y_max = curr_max + (0.1 * curr_max)  # some padding to the top

    # plotting
    sns.set(rc={'figure.figsize': fig_size})
    plt.ylim(y_min, y_max)
    ticks = list(range(0, len(data) + 1, x_tick_freq))
    plt.xticks(ticks=ticks)
    plt.ylabel(metric)

    sns.lineplot(x='n_seeds', y='0.5', data=data, label='median')
    sns.lineplot(x='n_seeds', y='0.25', data=data, label='25% quantile')
    sns.lineplot(x='n_seeds', y='0.75', data=data, label='75% quantile')

    if title:
        plt.title(title, fontsize=20)

    plt.legend(loc=2, bbox_to_anchor=(1,1))


def plot_quartiles_metric_bufn_combo(metric, belief_update_fn, data_mapping, y_min=None, y_max=None):

    # title = f'{metric}: {belief_update_fn}'
    data = data_mapping[belief_update_fn]

    plot_quartiles(data=data, metric=metric, title=belief_update_fn, y_min=y_min, y_max=y_max)
