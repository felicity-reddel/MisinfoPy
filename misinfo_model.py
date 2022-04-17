import time
import statistics
import math
from scipy.special import rel_entr
from mesa import Model
from mesa.datacollection import DataCollector
from mesa.time import StagedActivation
from mesa.space import NetworkGrid
import networkx as nx

from agents import *
from enums import *

import numpy as np
import math
from matplotlib import pyplot as plt


class MisinfoPy(Model):
    """Simple model with n agents."""

    def __init__(self,
                 n_agents=1000,
                 n_edges=2,
                 agent_ratio=None,

                 media_literacy_intervention=(0.0, SelectAgentsBy.RANDOM),
                 media_literacy_intervention_durations=None,
                 ranking_visibility_adjustment=-0.0,  # by default no ranking adjustment
                 p_true_threshold_deleting=-0.1,  # by default no deleting
                 p_true_threshold_ranking=-0.1,  # by default no ranking
                 p_true_threshold_strikes=-0.1,  # by default no strike system

                 belief_update_fn=BeliefUpdate.M3,
                 show_n_seen_posts=False,
                 show_n_connections=False):
        """
        Initializes the MisinfoPy
        :param agent_ratio: dictionary {String: float},
                                     String is agent type, float is agent_ratio in range [0.0,1.0]
        :param ranking_visibility_adjustment: float, range [-0.0, -1.0],
                                     the relative visibility change for posts with GroundTruth.FALSE
        :param p_true_threshold_deleting: float, range [0.0, 1.0],
                                     if below threshold-probability that post is true, post will be deleted.
                                     Thus, if threshold is negative, no posts will be deleted.
        :param p_true_threshold_ranking:  float, range [0.0, 1.0],
                                     if below threshold-probability that post is true, post will be down-ranked.
                                     Thus, if threshold is negative, no posts will be down-ranked.
        :param p_true_threshold_strikes:  float, range [0.0, 1.0],
                                     if below threshold-probability that post is true, post will cause strikes.
                                     Thus, if threshold is negative, no posts will cause strikes.
        :param n_agents: int, how many agents the model should have
        :param n_edges: int, with how many edges gets attached to the already built network
        :param media_literacy_intervention: tuple(float, SelectAgentsBy)
                float:
                    - domain [0,1)
                    - meaning: Percentage of agents empowered by media literacy intervention.
                                If 0.0: nobody is empowered by it, i.e., no media literacy intervention.
                                If 1.0: everybody is empowered by it.
        """
        super().__init__()

        if agent_ratio is None:
            agent_ratio = {NormalUser.__name__: 0.9, Disinformer.__name__: 0.1}
        # Making sure that the agent ratios add up to 1.0
        if sum(agent_ratio.values()) != 1.0:
            raise ValueError(f"The agent ratios add up to {sum(agent_ratio.values())}, "
                             f"while they should add up to 1.0.")

        if media_literacy_intervention_durations is None:
            media_literacy_intervention_durations = {"game duration": 300,
                                                     "MediaLiteracy.LOW": 3,
                                                     "MediaLiteracy.HIGH": 30}

        self.n_agents = n_agents
        self.schedule = StagedActivation(self, stage_list=["share_post_stage", "update_beliefs_stage"])
        self.G = random_graph(n_nodes=n_agents, m=n_edges)  # n_nodes = n_agents, exactly 1 agent per node
        self.grid = NetworkGrid(self.G)
        self.post_id_counter = 0
        self.agents_data = {'n_followers_range': (0, 0),
                            'n_following_range': (0, 0)}
        self.init_agents(agent_ratio)
        self.init_followers_and_following()
        self.belief_update_fn = belief_update_fn

        self.apply_media_literacy_intervention(media_literacy_intervention)
        if not (-1.0 <= ranking_visibility_adjustment <= -0.0):
            raise ValueError(f"Visibility adjustment for ranking was {ranking_visibility_adjustment}, "
                             f"while it should be in range [-0.0, -1.0]")
        self.media_literacy_intervention_durations = media_literacy_intervention_durations
        self.p_true_threshold_deleting = p_true_threshold_deleting
        self.n_posts_deleted = 0
        self.p_true_threshold_ranking = p_true_threshold_ranking
        self.ranking_visibility_adjustment = ranking_visibility_adjustment
        self.p_true_threshold_strikes = p_true_threshold_strikes
        self.show_n_seen_posts = show_n_seen_posts

        self.data_collector = DataCollector(model_reporters={
            "Avg Vax-Belief": self.get_avg_belief,
            "Avg Vax-Belief above threshold": self.get_avg_belief_above_threshold,
            "Avg Vax-Belief below threshold": self.get_avg_belief_below_threshold,
            "Total seen posts": self.get_total_seen_posts,
            "Free speech constraint": self.get_free_speech_constraint,
            "User effort": self.get_avg_user_effort})

        # DataCollector2: follow individual agents
        self.data_collector2 = DataCollector(model_reporters={
            f"Agent 0": self.get_belief_a0,
            f"Agent 1": self.get_belief_a10,
            f"Agent 2": self.get_belief_a20,
            f"Agent 3": self.get_belief_a30,
            f"Agent 4": self.get_belief_a40,
            f"Agent 5": self.get_belief_a50,
            f"Agent 6": self.get_belief_a60,
            f"Agent 7": self.get_belief_a70,
            f"Agent 8": self.get_belief_a80,
            f"Agent 9": self.get_belief_a90,
            f"Agent 10": self.get_belief_a100,
        })

        if show_n_connections:
            # Overview of how many agents have how many connections
            data = [len(agent.followers) for agent in self.schedule.agents]

            bins = np.linspace(math.ceil(min(data)),
                               math.floor(max(data)),
                               40)  # a fixed number of bins

            plt.xlim([min(data) - 5, max(data) + 5])

            plt.hist(data, bins=bins, alpha=0.5)
            plt.xlabel(f'Number of followers (highest: {max(data)})')
            plt.ylabel('Agent count')
            plt.show()

    def step(self):
        """Advance the model by one step."""
        self.schedule.step()
        self.data_collector.collect(self)
        self.data_collector2.collect(self)

        if self.show_n_seen_posts:
            data = [x.n_seen_posts for x in self.schedule.agents]

            bins = np.linspace(math.ceil(min(data)),
                               math.floor(max(data)),
                               40)  # a fixed number of bins

            plt.xlim([min(data) - 5, max(data) + 5])

            plt.hist(data, bins=bins, alpha=0.5)
            plt.xlabel(f'Number of seen posts (highest: {max(data)})')
            plt.ylabel('Agent count')
            plt.show()

    def run(self, steps=60, time_tracking=False, debug=False):
        """
        Runs the model for the specified number of steps.
        :param steps: int: number of model-steps the model should take
        :param time_tracking: Boolean, whether to print timing information
        :param debug: Boolean, whether to print details
        :return: tuple: metrics values for this run (n_above_belief_threshold, variance, kl_divergence, ...)
        """

        start_time = time.time()

        for i in range(steps):
            if debug:
                print(f'––––––––––––––––––––––––––––––––––––––– Step: {i} –––––––––––––––––––––––––––––––––––––––')
            self.step()

        if time_tracking:
            run_time = round(time.time() - start_time, 2)
            print(f'Run time: {run_time} seconds')

        if debug:
            n_inf_blocked = sum([1 for x in self.schedule.agents if x.blocked_until == math.inf])
            print()
            print(f"{n_inf_blocked}/{self.n_agents} blocked permanently")
            print(f"n_disinformers: {sum([1 for x in self.schedule.agents if isinstance(x, Disinformer)])}")

        # Calculate metrics for this run
        n_agents_above_belief_threshold = self.get_n_above_belief_threshold()
        polarization_variance = variance(model=self)
        polarization_kl_divergence_from_polarized = kl_divergence(model=self)
        engagement = self.get_total_seen_posts()
        free_speech_constraint = self.get_free_speech_constraint()
        avg_user_effort = self.get_avg_user_effort()

        return n_agents_above_belief_threshold, \
               polarization_variance, \
               polarization_kl_divergence_from_polarized, \
               engagement, \
               free_speech_constraint, \
               avg_user_effort

    # –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    # Init functions
    # –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    def init_agents(self, agent_ratio):
        """Initializes the agents.
        :param agent_ratio: dictionary, {String: float}
        """

        # Saving scenario
        types = []
        percentages = []
        for agent_type, percentage in agent_ratio.items():
            types.append(agent_type)
            percentages.append(percentage)

        # Create agents & add them to the scheduler
        for i in range(self.n_agents):

            # Pick which type should be added
            agent_type = random.choices(population=types, weights=percentages, k=1)[0]

            # Add agent of that type
            if agent_type is NormalUser.__name__:
                a = NormalUser(i, self)
                self.schedule.add(a)
            elif agent_type is Disinformer.__name__:
                a = Disinformer(i, self)
                self.schedule.add(a)

        # Place each agent in its node. (& save node_position into agent)
        for node in self.G.nodes:  # each node is just an integer (i.e., a node_id)
            agent = self.schedule.agents[node]

            # save node_position into agent
            self.grid.place_agent(agent, node)

            # add agent to node
            self.G.nodes[node]['agent'] = agent

    def init_followers_and_following(self):
        """Initializes the followers and following of each agent."""
        n_followers_list = []
        n_following_list = []

        # Init followers & following (after all agents have been set up)
        for agent in self.schedule.agents:
            # Gather connected agents
            predecessors = [self.schedule.agents[a] for a in self.G.predecessors(agent.unique_id)]
            successors = [self.schedule.agents[a] for a in self.G.successors(agent.unique_id)]

            # Assign to this agent
            agent.following = predecessors
            agent.followers = successors

            # Gather number of followers/following to this agent
            n_following_list.append(len(agent.following))
            n_followers_list.append(len(agent.followers))

        # Gather boundaries of ranges (n_followers & n_following)
        min_n_following = min(n_following_list)
        max_n_following = max(n_following_list)
        min_n_followers = min(n_followers_list)
        max_n_followers = max(n_followers_list)

        # Save ranges into agents_data
        self.agents_data["n_following_range"] = (min_n_following, max_n_following)
        self.agents_data["n_followers_range"] = (min_n_followers, max_n_followers)

    def apply_media_literacy_intervention(self, media_literacy_intervention=(0.0, SelectAgentsBy.RANDOM)):
        """
        Applies the media literacy intervention (if needed).
        :param media_literacy_intervention: float, [0,1),
                    Percentage of agents empowered by media literacy intervention.
                    If 0.0: nobody is empowered by it, i.e., no media literacy intervention.
                    If 1.0: everybody is empowered by it.
        """
        percentage, select_by = media_literacy_intervention

        # If media literacy intervention is used: select agents for intervention, adjust their media literacy.
        # (i.e., if some percentage of agents is targeted with it)
        if percentage > 0.0:
            n_select = int(len(self.schedule.agents) * percentage)
            selected_agents = self.select_agents_for_media_literacy_intervention(n_select, select_by)

            for agent in selected_agents:
                agent.media_literacy = MediaLiteracy.HIGH
                agent.received_media_literacy_intervention = True

    def select_agents_for_media_literacy_intervention(self, n_select=0, select_by=SelectAgentsBy.RANDOM):
        """
        Select agents for the intervention.
        :param n_select:    int, how many agents should be selected for the intervention
        :param select_by:   SelectBy(Enum), selection method, e.g. SelectBy.RANDOM
        :return:            list of agents, [(Base)Agent, (Base)Agent, ...]
        """
        selected_agents = []
        if select_by.__eq__(SelectAgentsBy.RANDOM):
            selected_agents = random.choices(self.schedule.agents, k=n_select)
        else:
            raise ValueError(f'Selection style {select_by} has not yet been implemented. '
                             f'To sample which agents will be empowered by the media literacy intervention,'
                             f'please use an agent selection style that has already been implemented. '
                             f'(e.g. SelectAgentsBy.RANDOM).')

        return selected_agents

    def get_beliefs(self):
        """
        Returns a list of beliefs that includes all agents' belief on Topic.VAX.
        :return: list
        """
        return [agent.beliefs[Topic.VAX] for agent in self.schedule.agents]

    # –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    # DataCollector functions
    # –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    def get_free_speech_constraint(self):
        """
        Calculates an estimate of the restriction to freedom of speech. The float represents the number of
        "deleted" posts, compared to the number of posts that the agents wanted to post.
        It is aggregated over the whold population, i.e., total "deleted" posts divided by the
        total number of how many posts the agents wanted to post.
        "deleted" includes downranked posts – If a post was downranked by 20%, it counts as 0.2 posts deleted.
        :return: float, range [0.0, 1.0]
        """

        total_posts = sum([a.preferred_n_posts for a in self.schedule.agents])
        n_posted = sum([len(a.last_posts) for a in self.schedule.agents])
        n_prevented = total_posts - n_posted
        n_downranked = sum([a.n_downranked for a in self.schedule.agents])

        constraint = n_prevented + (n_downranked * abs(self.ranking_visibility_adjustment))
        rel_constraint = constraint / total_posts

        return rel_constraint

    def get_avg_user_effort(self):
        """
        Calculates an estimate of the effort a user has for
            - judging the posts' truthfulness (summed over all posts)
            - partaking in the media literacy interventions (once)
        The effort is measured in minutes. The average effort per user is returned.

        :return: float, cannot be lower than zero
        """
        effort_per_agent = []
        for agent in self.schedule.agents:
            game_duration = int(agent.received_media_literacy_intervention) * \
                            self.media_literacy_intervention_durations["game duration"]
            judging_all_posts = agent.n_total_seen_posts * \
                                self.media_literacy_intervention_durations[str(agent.media_literacy)]
            effort = game_duration + judging_all_posts
            effort_per_agent.append(effort)

        avg_effort_seconds = sum(effort_per_agent) / len(self.schedule.agents)
        avg_effort_minutes = avg_effort_seconds / 60.0

        return avg_effort_minutes

    def get_avg_belief(self, topic=Topic.VAX, dummy=None) -> float:
        """
        Return average belief of all agents on a given topic. For the DataCollector.
        :param topic:   Topic
        :param dummy:   None: just to avoid error (.ipynb files)
        :return:        float
        """

        agent_beliefs = [a.beliefs[topic] for a in self.schedule.agents]
        avg_belief = sum(agent_beliefs) / len(agent_beliefs)

        return avg_belief

    def get_total_seen_posts(self):
        """
        Returns the total number of seen posts, summed over all agents.
        :return: total: int
        """

        per_agent = [sum(x.n_seen_posts) for x in self.schedule.agents]
        total = sum(per_agent)

        return total

    def get_topic_above_below_sizes(self, topic=Topic.VAX, threshold=50.0, dummy=None) -> tuple:
        """
        Return tuple of how many agents' belief on a given topic is above and below the provided threshold.
         For the DataCollector.
        :param topic:       Topic
        :param threshold:   float
        :param dummy:       None: just to avoid error  (.ipynb files)
        :return:            tuple
        """

        agent_beliefs = [a.beliefs[topic] for a in self.schedule.agents]
        n_above = sum([1 for a_belief in agent_beliefs if a_belief >= threshold])
        n_below = len(agent_beliefs) - n_above

        return n_above, n_below

    def get_n_above_belief_threshold(self, topic=Topic.VAX, threshold=50, dummy=None) -> int:
        """
        Returns how many agents' belief on a given topic is above and below the provided threshold.
         For the DataCollector.
        :param topic:       Topic
        :param threshold:   float
        :param dummy:       None: just to avoid error  (.ipynb files)
        :return: n_above:   int
        """

        agent_beliefs = [a.beliefs[topic] for a in self.schedule.agents]
        n_above = sum([1 for a_belief in agent_beliefs if a_belief >= threshold])

        return n_above

    def get_n_below_belief_threshold(self, topic=Topic.VAX, threshold=50, dummy=None) -> int:
        """
        Returns how many agents' belief on a given topic are below the provided threshold.
         For the DataCollector.
        :param topic:       Topic
        :param threshold:   float
        :param dummy:       None: just to avoid error  (.ipynb files)
        :return: n_below    int
        """

        agent_beliefs = [a.beliefs[topic] for a in self.schedule.agents]
        n_below = sum([1 for a_belief in agent_beliefs if a_belief < threshold])

        return n_below

    def get_avg_belief_above_threshold(self, topic=Topic.VAX, threshold=50, dummy=None) -> float:
        """
        Returns the average belief of agents that are above the provided threshold.
         For the DataCollector.
        :param topic:       Topic
        :param threshold:   float
        :param dummy:       None: just to avoid error  (.ipynb files)
        :return: avg:       float
        """

        beliefs_above_threshold = [a.beliefs[topic] for a in self.schedule.agents if a.beliefs[topic] >= threshold]
        if len(beliefs_above_threshold) == 0:
            avg = self.get_avg_belief_below_threshold()  # If nobody above threshold, take avg of below threshold.
        else:
            avg = sum(beliefs_above_threshold) / len(beliefs_above_threshold)
        return avg

    def get_avg_belief_below_threshold(self, topic=Topic.VAX, threshold=50, dummy=None) -> float:
        """
        Returns the average belief of agents that are below the provided threshold.
         For the DataCollector.
        :param topic:       Topic
        :param threshold:   float
        :param dummy:       None: just to avoid error  (.ipynb files)
        :return: avg:       float
        """

        beliefs_below_threshold = [a.beliefs[topic] for a in self.schedule.agents if a.beliefs[topic] < threshold]

        if len(beliefs_below_threshold) == 0:
            avg = self.get_avg_belief_above_threshold()  # If nobody below threshold, take avg of above threshold.
        else:
            avg = sum(beliefs_below_threshold) / len(beliefs_below_threshold)

        return avg

    def get_indiv_beliefs(self, topic=Topic.VAX, agent_ids_list=None) -> dict:
        """
        Returns a dictionary of the current get_beliefs of the agents with the unique_ids listed in agent_ids_list.
        :param topic:       Topic
        :param agent_ids_list: list of agent.unique_id's
        :return: dict, {unique_id: vax_belief}
        """
        if agent_ids_list is None:
            agent_ids_list = [a.unique_id for a in self.schedule.agents]
        vax_beliefs: dict[str, float] = {}
        agents = [a for a in self.schedule.agents if a.unique_id in agent_ids_list]
        for agent in agents:
            belief = agent.beliefs[topic]
            vax_beliefs[f'belief of agent {id}'] = belief

        return vax_beliefs

    def get_belief_a0(self, topic=Topic.VAX, dummy=None) -> float:
        """
        Returns the belief of agent 0 at current tick.
        For data_collector2.
        :param topic:       Topic
        :param dummy:       None: just to avoid error  (.ipynb files)
        :return: belief:    float
        """
        agent_i = [a for a in self.schedule.agents if a.unique_id == self.n_agents * 0][0]
        belief = agent_i.beliefs[topic]
        return belief

    def get_belief_a10(self, topic=Topic.VAX, dummy=None) -> float:
        """
        Returns the belief a specific agent at current tick. (The agent at 10% of unique_ids.)
        For data_collector2.
        :param topic:       Topic
        :param dummy:       None: just to avoid error  (.ipynb files)
        :return: belief:    float
        """
        agent_i = [a for a in self.schedule.agents if a.unique_id == self.n_agents * 0.1][0]
        belief = agent_i.beliefs[topic]
        return belief

    def get_belief_a20(self, topic=Topic.VAX, dummy=None) -> float:
        """
        Returns the belief a specific agent at current tick. (The agent at 20% of unique_ids.)
        For data_collector2.
        :param topic:       Topic
        :param dummy:       None: just to avoid error  (.ipynb files)
        :return: belief:    float
        """
        agent_i = [a for a in self.schedule.agents if a.unique_id == self.n_agents * 0.2][0]
        belief = agent_i.beliefs[topic]
        return belief

    def get_belief_a30(self, topic=Topic.VAX, dummy=None) -> float:
        """
        Returns the belief a specific agent at current tick. (The agent at 30% of unique_ids.)
        For data_collector2.
        :param topic:       Topic
        :param dummy:       None: just to avoid error  (.ipynb files)
        :return: belief:    float
        """
        agent_i = [a for a in self.schedule.agents if a.unique_id == self.n_agents * 0.3][0]
        belief = agent_i.beliefs[topic]
        return belief

    def get_belief_a40(self, topic=Topic.VAX, dummy=None) -> float:
        """
        Returns the belief a specific agent at current tick. (The agent at 40% of unique_ids.)
        For data_collector2.
        :param topic:       Topic
        :param dummy:       None: just to avoid error  (.ipynb files)
        :return: belief:    float
        """
        agent_i = [a for a in self.schedule.agents if a.unique_id == self.n_agents * 0.4][0]
        belief = agent_i.beliefs[topic]
        return belief

    def get_belief_a50(self, topic=Topic.VAX, dummy=None) -> float:
        """
        Returns the belief a specific agent at current tick. (The agent at 50% of unique_ids.)
        For data_collector2.
        :param topic:       Topic
        :param dummy:       None: just to avoid error  (.ipynb files)
        :return: belief:    float
        """
        agent_i = [a for a in self.schedule.agents if a.unique_id == self.n_agents * 0.5][0]
        belief = agent_i.beliefs[topic]
        return belief

    def get_belief_a60(self, topic=Topic.VAX, dummy=None) -> float:
        """
        Returns the belief a specific agent at current tick. (The agent at 60% of unique_ids.)
        :param topic:       Topic
        :param dummy:       None: just to avoid error  (.ipynb files)
        :return: belief:    float
        """
        agent_i = [a for a in self.schedule.agents if a.unique_id == self.n_agents * 0.6][0]
        belief = agent_i.beliefs[topic]
        return belief

    def get_belief_a70(self, topic=Topic.VAX, dummy=None) -> float:
        """
        Returns the belief a specific agent at current tick. (The agent at 70% of unique_ids.)
        For data_collector2.
        :param topic:       Topic
        :param dummy:       None: just to avoid error  (.ipynb files)
        :return: belief:    float
        """
        agent_i = [a for a in self.schedule.agents if a.unique_id == self.n_agents * 0.7][0]
        belief = agent_i.beliefs[topic]
        return belief

    def get_belief_a80(self, topic=Topic.VAX, dummy=None) -> float:
        """
        Returns the belief a specific agent at current tick. (The agent at 80% of unique_ids.)
        For data_collector2.
        :param topic:       Topic
        :param dummy:       None: just to avoid error  (.ipynb files)
        :return: belief:    float
        """
        agent_i = [a for a in self.schedule.agents if a.unique_id == self.n_agents * 0.8][0]
        belief = agent_i.beliefs[topic]
        return belief

    def get_belief_a90(self, topic=Topic.VAX, dummy=None) -> float:
        """
        Returns the belief a specific agent at current tick. (The agent at 90% of unique_ids.)
        For data_collector2.
        :param topic:       Topic
        :param dummy:       None: just to avoid error  (.ipynb files)
        :return: belief:    float
        """
        agent_i = [a for a in self.schedule.agents if a.unique_id == self.n_agents * 0.9][0]
        belief = agent_i.beliefs[topic]
        return belief

    def get_belief_a100(self, topic=Topic.VAX, dummy=None) -> float:
        """
        Returns the belief a specific agent at current tick. (The agent at 100% of unique_ids (i.e., last agent))
        For data_collector2.
        :param topic:       Topic
        :param dummy:       None: just to avoid error  (.ipynb files)
        :return: belief:    float
        """
        agent_i = [a for a in self.schedule.agents if a.unique_id == self.n_agents - 1][0]
        belief = agent_i.beliefs[topic]
        return belief


# –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
def discretize(belief_list, n_bins=25):
    """
    Transform a list of (belief values) into a discretized distribution
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
    :param epsilon: float: smallest value, needed because values should not be 0 because of ln(0)=inf in KL-div
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
    the current belief belief_list of the agents (or a the provided belief_list).
    If both, belief_list and model, are not specified, no KL divergence can be calculated and the function returns: None
    :param belief_list: list: listing the belief value of each agent
    :param model: MisinfoPy model
    :param template_pdf: list: discrete probability density function as a list of length n_bins
    :param n_bins: int: number of bins for discretization of belief_list
    :param n_digits: int: to how many digits the value should be rounded, for non-rounded use high number (e.g., 20)
    :return: float: symmetric kl-divergence
    """

    belief_list = init_belief_list(belief_list, model)

    # Discretize belief_list
    discrete_distribution = discretize(belief_list, n_bins)

    # Transform to probability belief_list (i.e., transform belief_list into a relative belief_list)
    relative_distribution = [x / sum(discrete_distribution) for x in discrete_distribution]

    # Define the template of a polarized pdf. (Values should not be 0 because of ln(0)=inf in KL-divergence.)
    if template_pdf is None:
        template_pdf = create_polarized_pdf(epsilon=0.001, n_bins=n_bins)

    # Actual KL-divergence calculation (which is the same as the sum over the relative entropy)
    direction1 = sum(rel_entr(discrete_distribution, template_pdf))
    direction2 = sum(rel_entr(template_pdf, discrete_distribution))
    symmetric_kl_div = (direction1 + direction2) / 2

    rounded_kl_div = round(symmetric_kl_div, n_digits)

    return rounded_kl_div


def variance(belief_list=None, model=None, n_digits=2):
    """
    Calculates the variance between the template of a polarized belief_list and
    the current belief belief_list of the agents (or a the provided belief_list).
    :param belief_list: list: listing the belief value of each agent
    :param model: MisinfoPy model
    :param n_digits: int: to how many digits the value should be rounded, for non-rounded use high number (e.g., 20)
    :return: float: variance
    """
    belief_list = init_belief_list(belief_list, model)

    var = round(statistics.variance(belief_list), n_digits)

    return var


def init_belief_list(belief_list=None, model=None):
    """
    Makes sure there is a belief_list (either provided, or belief_list from the model's schedule),
    e.g., before calculating polarization metrics KL-divergence & variance, or belief metric
    :param belief_list: list: listing the belief value of each agent
    :param model: MisinfoPy model
    :return: list
    """

    if belief_list is None:
        if model is None:
            raise ValueError("The model is not yet initialized and no belief-list has been provided.")
        else:
            belief_list = model.get_beliefs()

    return belief_list


#   Graph Functions
# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

def random_graph(n_nodes, m, seed=None, directed=True) -> nx.Graph:
    """
    Generates a random graph a la Barabasi Albert.
    :param n_nodes:     int, number of nodes
    :param m:           int, avg number of edges added per node
    :param seed:        int, random seed
    :param directed:    bool, undirected or directed graph

    :return:            nx.Graph, the resulting stochastic graph (barabasi albert G)

    # Note:     Using Barabasi Albert graphs, because they are fitting for social networks.
    #           ( https://en.wikipedia.org/wiki/Barab%C3%A1si%E2%80%93Albert_model )
    # Later:    Potential extension: parameter for skew of node degree.
    # FYI:      n=10, m=3, doesn't create 30 edges, but only e.g., 21. Not each node has 3 edges.
    """
    graph = nx.barabasi_albert_graph(n_nodes, m, seed)

    if directed:  # --> has key
        # Make graph directed (i.e., asymmetric edges possible = multiple directed edges)
        graph = nx.MultiDiGraph(graph)  # undirected --> "=bidirectional"

        # Add edge weights
        for edge in graph.edges:
            from_e = edge[0]
            to_e = edge[1]
            key = edge[2]

            # Sample weights & save them
            weight = random.randint(0, 100)
            graph.edges[from_e, to_e, key]['weight'] = weight

    else:  # not directed --> no key
        # Add edge weights
        for edge in graph.edges:
            from_e = edge[0]
            to_e = edge[1]

            # Sample weights & save them
            weight = 1 + random.random() * random.choice([-1, 1])  # weights in range [0,2]: no visible change
            graph.edges[from_e, to_e]['weight'] = weight

    return graph
