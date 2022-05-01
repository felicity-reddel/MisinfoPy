import networkx as nx
import math
import statistics as stats
from scipy.special import rel_entr


def calculate_extremeness(beliefs):
    """
    Calculates how extreme the post is.
    Here: the posts extremeness = the average extremeness of all its tweet_beliefs.
    :param beliefs: dictionary of tweet_beliefs {Topic: float}. This can be from an agent or from a post.
    :return: float: in domain [0,1)
    """
    extremeness_values = []
    for topic, value in beliefs.items():
        extremeness = abs(50 - value)
        extremeness_values.append(extremeness)

    avg_extremeness = sum(extremeness_values) / len(extremeness_values)

    # Scale to domain [0,1)
    avg_extremeness /= 50

    return avg_extremeness


# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
#   From/for misinfo_model.py
# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

def discretize(belief_list, n_bins=25):
    """
    Transform a list of (belief) values into a discretized distribution
    :param belief_list: list: list of belief values (floats)
    :param n_bins: int: number of bins
    :return: list: list of integers, representing number of agents in that "belief-bin"
    """
    discrete_distribution = []
    bin_size = math.ceil(100.0 / n_bins)
    for i in range(1, n_bins + 1):
        upper_bound = i * bin_size
        agents_in_bin = [x for x in belief_list if x <= upper_bound]
        # remove agents from current bin from the belief_list
        belief_list = [x for x in belief_list if x not in agents_in_bin]
        discrete_distribution.append(len(agents_in_bin))

    # Make more robust (to avoid cases of existing empty bins, which would make KL-div=infinity)
    if 0 in discrete_distribution:
        discrete_distribution = [x + 1 for x in discrete_distribution]

    return discrete_distribution


def create_polarized_pdf(epsilon=0.001, n_bins=25):
    """
    Creates most polarized probability distribtion function (pdf) as a comparison for the KL-divergence metric.
    :param epsilon: float: smallest n_seen_posts_repl,
                    needed because values should not be 0 because of ln(0)=inf in KL-div
    :param n_bins: int: number of bins of discrete belief_list
    :return: list: representing polarized pdf
    """
    pole_value = (1 - ((n_bins - 2) * epsilon)) / 2
    pdf = [epsilon] * n_bins
    pdf[0] = pole_value
    pdf[-1] = pole_value

    return pdf


def kl_divergence(belief_list=None, model=None, template_pdf=None, n_bins=25, n_digits=2):
    """
    Calculates the symmetric Kullback-Leibler divergence between the template of a polarized belief_list and
    the current belief belief_list of the agents (or the provided belief_list).
    If both, belief_list and model, are not specified, no KL divergence can be calculated and the function returns: None
    :param belief_list: list: listing the belief n_seen_posts_repl of each agent
    :param model: MisinfoPy model
    :param template_pdf: list: discrete probability density function as a list of length n_bins
    :param n_bins: int: number of bins for discretization of belief_list
    :param n_digits: int: to how many digits the n_seen_posts_repl should be rounded, for non-rounded use high value
    :return: float: symmetric kl-divergence
    """

    belief_list = get_belief_list(belief_list, model)

    # Discretize belief_list
    discrete_distribution = discretize(belief_list, n_bins)

    # Transform to probability belief_list (i.e., transform belief_list into a relative belief_list)
    relative_distribution = [x / sum(discrete_distribution) for x in discrete_distribution]

    # Define the template of a polarized pdf. (Values should not be 0 because of ln(0)=inf in KL-divergence.)
    if template_pdf is None:
        template_pdf = create_polarized_pdf(epsilon=0.001, n_bins=n_bins)

    # Actual KL-divergence calculation (which is the same as the sum over the relative entropy)
    direction1 = sum(rel_entr(relative_distribution, template_pdf))
    direction2 = sum(rel_entr(template_pdf, relative_distribution))
    symmetric_kl_div = (direction1 + direction2) / 2

    rounded_kl_div = round(symmetric_kl_div, n_digits)

    return rounded_kl_div


def variance(belief_list=None, model=None, n_digits=2):
    """
    Calculates the variance of
    the current belief belief_list of the agents (or the provided belief_list).
    :param belief_list: list: listing the belief n_seen_posts_repl of each agent
    :param model: MisinfoPy model
    :param n_digits: int: to how many digits the n_seen_posts_repl should be rounded, for non-rounded use high number
                            (e.g., 20)
    :return: float: variance
    """
    belief_list = get_belief_list(belief_list, model)
    var = round(stats.variance(belief_list), n_digits)

    return var


def get_belief_list(belief_list=None, model=None):
    """
    Makes sure there is a belief_list (either provided, or belief_list from the model's schedule),
    e.g., before calculating polarization metrics KL-divergence & variance, or belief metric
    :param belief_list: list: listing the belief n_seen_posts_repl of each agent
    :param model: MisinfoPy model
    :return: list
    """

    if belief_list is None:
        if model is None:
            raise ValueError("The model is not yet initialized and no belief-list has been provided.")
        else:
            belief_list = model.get_beliefs()

    return belief_list


# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
#   From/for agents.py
# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––


def rescale(old_value, new_domain=(-100, 100)):
    """
    Rescales a value from one range to another.
    By default, from range [-100ˆ3,100ˆ3] to [-100,100].

    :param old_value:   float
    :param new_domain:   tuple, (min_new_range, max_new_range)
    :return: new_value: float
    """
    old_min, old_max = (-1000000, 1000000)

    if old_value < 0:
        old_max = 0  # we already know that it should decrease, thus old_max = 0  and  new_max = 0
        new_max = 0
        new_min, _ = new_domain
        old_range = old_max - old_min
        new_range = new_max - new_min

        new_value = (((old_value - old_min) * new_range) / old_range) + new_min

    elif old_value > 0:
        old_min = 0  # we already know that it should increase, thus old_min = 0   and  new_min = 0
        new_min = 0
        _, new_max = new_domain
        old_range = old_max - old_min
        new_range = new_max - new_min

        new_value = (((old_value - old_min) * new_range) / old_range) + new_min
    else:
        new_value = 0

    return new_value


def get_update_strength(prev_belief, mean=50.0, std_dev=30.0):
    """
    Uses a normal distribution (with the provided parameters)
    to return the update_strength that corresponds to the provided belief_strength.
    :param prev_belief:     float
    :param mean:                float
    :param std_dev:             float
    :return: update_strength:    float
    """

    dividend = math.exp((((prev_belief - mean) / std_dev) ** 2 * (-0.5)))
    divisor = math.sqrt(2 * math.pi) * std_dev

    update_strength = dividend / divisor

    return update_strength


# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
#   From/for experiments.py. Not used anymore.
# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

def calculate_avg_belief(misinfo_model):
    """
    Calculates the average belief over all agents.
    :param misinfo_model: MisinfoPy
    :return: avg_belief: float
    """
    topic = Topic.VAX
    beliefs = []
    for agent in misinfo_model.schedule.agents:
        agent_belief_on_topic = agent.beliefs[topic]
        beliefs.append(agent_belief_on_topic)

    avg_belief = sum(beliefs) / len(beliefs)

    return avg_belief


def calculate_percentage_agents_above_threshold(misinfo_model, threshold):
    """
    Calculates the percentage of agents that is above the specified threshold.
    :param misinfo_model: MisinfoPy
    :param threshold: float
    :return: float
    """
    agent_beliefs = [a.beliefs[Topic.VAX] for a in misinfo_model.schedule.agents]
    n_above: int = sum([1 for a_belief in agent_beliefs if a_belief >= threshold])
    percentage_above = n_above / len(misinfo_model.schedule.agents)
    return percentage_above

