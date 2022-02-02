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
                 ranking_intervention=False):
        """
        Initializes the MisinfoPy
        :param agent_ratio: dictionary {String: float}
        :param ranking_intervention: boolean
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

        self.n_agents = n_agents
        self.schedule = StagedActivation(self, stage_list=["share_post_stage", "update_beliefs_stage"])
        self.G = random_graph(n_nodes=n_agents, m=n_edges)  # n_nodes = n_agents, exactly 1 agent per node
        self.grid = NetworkGrid(self.G)
        self.post_id_counter = 0
        self.agents_data = {'n_followers_range': (0, 0),
                            'n_following_range': (0, 0)}
        self.init_agents(agent_ratio)
        self.init_followers_and_following()

        self.apply_media_literacy_intervention(media_literacy_intervention)
        self.ranking_intervention = ranking_intervention

        self.data_collector = DataCollector(model_reporters={
            "Avg Vax-Belief": self.get_avg_vax_belief,
            "Avg Vax-Belief above threshold": self.get_avg_above_vax_threshold,
            "Avg Vax-Belief below threshold": self.get_avg_below_vax_threshold})

        # DataCollector2: follow individual agents
        self.data_collector2 = DataCollector(model_reporters={
            f"Agent 0": self.get_vax_belief_0,
            f"Agent 1": self.get_vax_belief_10,
            f"Agent 2": self.get_vax_belief_20,
            f"Agent 3": self.get_vax_belief_30,
            f"Agent 4": self.get_vax_belief_40,
            f"Agent 5": self.get_vax_belief_50,
            f"Agent 6": self.get_vax_belief_60,
            f"Agent 7": self.get_vax_belief_70,
            f"Agent 8": self.get_vax_belief_80,
            f"Agent 9": self.get_vax_belief_90,
            f"Agent 10": self.get_vax_belief_100,
        })

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

            # Benefiting agents (only agents with low media literacy can benefit from the intervention)
            benefiting_agents = [agent for agent in selected_agents if agent.media_literacy.__eq__(MediaLiteracy.LOW)]

            for agent in benefiting_agents:
                agent.media_literacy = MediaLiteracy.HIGH

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
            print(f'ERROR: Selection style not yet implemented. '
                  f'To sample which agents will be empowered by the media literacy intervention,'
                  f'Please use an agent selection style that has already been implemented. (e.g. random)')

        return selected_agents

    # –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    # DataCollector functions
    # –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    def get_avg_vax_belief(self, dummy) -> float:  # dummy parameter: to avoid error
        """
        Return average belief of all agents on a given topic. For the DataCollector.
        :return:        float
        """
        topic = str(Topic.VAX)

        agent_beliefs = [a.beliefs[topic] for a in self.schedule.agents]
        avg_belief = sum(agent_beliefs) / len(agent_beliefs)

        return avg_belief

    def get_vax_category_sizes(self, dummy) -> tuple:  # dummy parameter: to avoid error
        """
        Return tuple of how many agents' belief on a given topic is above and below the provided threshold.
         For the DataCollector.
        # :param threshold:   float  # to make it more programmatic later
        # :param topic:       Topic  # to make it more programmatic later
        :return:            tuple
        """
        topic = Topic.VAX
        threshold = 50.0

        agent_beliefs = [a.beliefs[topic] for a in self.schedule.agents]
        n_above = sum([1 for a_belief in agent_beliefs if a_belief >= threshold])
        n_below = len(agent_beliefs) - n_above

        return n_above, n_below

    def get_above_vax_threshold(self, dummy) -> int:  # adjust code later: threshold_dict={Topic.VAX: 50.0}?
        """
        Returns how many agents' belief on a given topic is above and below the provided threshold.
         For the DataCollector.
        # :param threshold_dict:   dict {Topic: float}  # to make it more programmatic later. Not sure whether possible.
        :return: int
        """
        topic = Topic.VAX
        threshold = 50.0

        agent_beliefs = [a.beliefs[topic] for a in self.schedule.agents]
        n_above = sum([1 for a_belief in agent_beliefs if a_belief >= threshold])

        return n_above

    def get_below_vax_threshold(self, dummy) -> int:  # dummy parameter: to avoid error
        """
        Returns how many agents' belief on a given topic is above and below the provided threshold.
         For the DataCollector.
        # :param threshold:   float  # to make it more programmatic later
        # :param topic:       Topic  # to make it more programmatic later
        :return:            tuple
        """
        topic = Topic.VAX
        threshold = 50.0

        agent_beliefs = [a.beliefs[topic] for a in self.schedule.agents]
        n_below = sum([1 for a_belief in agent_beliefs if a_belief < threshold])

        return n_below

    def get_avg_above_vax_threshold(self, dummy) -> float:
        """
        Returns the average belief of agents that are above the provided threshold.
         For the DataCollector.
        :return: float
        """
        topic = str(Topic.VAX)
        threshold = 50.0

        beliefs_above_threshold = [a.beliefs[topic] for a in self.schedule.agents if a.beliefs[topic] >= threshold]
        if len(beliefs_above_threshold) == 0:
            avg = self.get_avg_below_vax_threshold(self)  # If nobody above threshold, take avg of below threshold.
        else:
            avg = sum(beliefs_above_threshold) / len(beliefs_above_threshold)
        return avg

    def get_avg_below_vax_threshold(self, dummy) -> float:
        """
        Returns the average belief of agents that are below the provided threshold.
         For the DataCollector.
        :return: float
        """
        topic = str(Topic.VAX)
        threshold = 50.0

        beliefs_below_threshold = [a.beliefs[topic] for a in self.schedule.agents if a.beliefs[topic] < threshold]

        if len(beliefs_below_threshold) == 0:
            avg = self.get_avg_above_vax_threshold(self)  # If nobody below threshold, take avg of above threshold.
        else:
            avg = sum(beliefs_below_threshold) / len(beliefs_below_threshold)

        return avg

    def get_vax_beliefs(self) -> list:
        """
        Returns list of vax-belief of all agents.
        For the DataCollector.
        :return: list (of floats)
        """
        topic = str(Topic.VAX)
        vax_beliefs = [agent.beliefs[topic] for agent in self.schedule.agents]

        return vax_beliefs

    def get_indiv_vax_beliefs(self, agent_ids_list) -> dict:
        """
        Returns a dictionary of the current get_vax_beliefs of the agents with the unique_ids listed in agent_ids_list.
        :param agent_ids_list: list of agent.unique_id's
        :return: dict, {unique_id: vax_belief}
        """
        topic = str(Topic.VAX)
        vax_beliefs: dict[str, float] = {}
        agents = [a for a in self.schedule.agents if a.unique_id in agent_ids_list]
        for agent in agents:
            belief = agent.beliefs[topic]
            vax_beliefs[f'belief of agent {id}'] = belief

        return vax_beliefs

    def get_vax_belief_0(self, dummy) -> float:
        """
        Returns the belief of agent 0 at current tick.
        For data_collector2.
        :param dummy:   to avoid error
        :return:        float
        """
        topic = str(Topic.VAX)
        agent_i = [a for a in self.schedule.agents if a.unique_id == self.n_agents * 0][0]
        belief = agent_i.beliefs[topic]
        return belief

    def get_vax_belief_10(self, dummy) -> float:
        """
        Returns the belief a specific agent at current tick. (The agent at 10% of unique_ids.)
        For data_collector2.
        :param dummy:   to avoid error
        :return:        float
        """
        topic = str(Topic.VAX)
        agent_i = [a for a in self.schedule.agents if a.unique_id == self.n_agents * 0.1][0]
        belief = agent_i.beliefs[topic]
        return belief

    def get_vax_belief_20(self, dummy) -> float:
        """
        Returns the belief a specific agent at current tick. (The agent at 20% of unique_ids.)
        For data_collector2.
        :param dummy:   to avoid error
        :return:        float
        """
        topic = str(Topic.VAX)
        agent_i = [a for a in self.schedule.agents if a.unique_id == self.n_agents * 0.2][0]
        belief = agent_i.beliefs[topic]
        return belief

    def get_vax_belief_30(self, dummy) -> float:
        """
        Returns the belief a specific agent at current tick. (The agent at 30% of unique_ids.)
        For data_collector2.
        :param dummy:   to avoid error
        :return:        float
        """
        topic = str(Topic.VAX)
        agent_i = [a for a in self.schedule.agents if a.unique_id == self.n_agents * 0.3][0]
        belief = agent_i.beliefs[topic]
        return belief

    def get_vax_belief_40(self, dummy) -> float:
        """
        Returns the belief a specific agent at current tick. (The agent at 40% of unique_ids.)
        For data_collector2.
        :param dummy:   to avoid error
        :return:        float
        """
        topic = str(Topic.VAX)
        agent_i = [a for a in self.schedule.agents if a.unique_id == self.n_agents * 0.4][0]
        belief = agent_i.beliefs[topic]
        return belief

    def get_vax_belief_50(self, dummy) -> float:
        """
        Returns the belief a specific agent at current tick. (The agent at 50% of unique_ids.)
        For data_collector2.
        :param dummy:   to avoid error
        :return:        float
        """
        topic = str(Topic.VAX)
        agent_i = [a for a in self.schedule.agents if a.unique_id == self.n_agents * 0.5][0]
        belief = agent_i.beliefs[topic]
        return belief

    def get_vax_belief_60(self, dummy) -> float:
        """
        Returns the belief a specific agent at current tick. (The agent at 60% of unique_ids.)
        For data_collector2.
        :param dummy:   to avoid error
        :return:        float
        """
        topic = str(Topic.VAX)
        agent_i = [a for a in self.schedule.agents if a.unique_id == self.n_agents * 0.6][0]
        belief = agent_i.beliefs[topic]
        return belief

    def get_vax_belief_70(self, dummy) -> float:
        """
        Returns the belief a specific agent at current tick. (The agent at 70% of unique_ids.)
        For data_collector2.
        :param dummy:   to avoid error
        :return:        float
        """
        topic = str(Topic.VAX)
        agent_i = [a for a in self.schedule.agents if a.unique_id == self.n_agents * 0.7][0]
        belief = agent_i.beliefs[topic]
        return belief

    def get_vax_belief_80(self, dummy) -> float:
        """
        Returns the belief a specific agent at current tick. (The agent at 80% of unique_ids.)
        For data_collector2.
        :param dummy:   to avoid error
        :return:        float
        """
        topic = str(Topic.VAX)
        agent_i = [a for a in self.schedule.agents if a.unique_id == self.n_agents * 0.8][0]
        belief = agent_i.beliefs[topic]
        return belief

    def get_vax_belief_90(self, dummy) -> float:
        """
        Returns the belief a specific agent at current tick. (The agent at 90% of unique_ids.)
        For data_collector2.
        :param dummy:   to avoid error
        :return:        float
        """
        topic = str(Topic.VAX)
        agent_i = [a for a in self.schedule.agents if a.unique_id == self.n_agents * 0.9][0]
        belief = agent_i.beliefs[topic]
        return belief

    def get_vax_belief_100(self, dummy) -> float:
        """
        Returns the belief a specific agent at current tick. (The agent at 100% of unique_ids (i.e., last agent))
        For data_collector2.
        :param dummy:   to avoid error
        :return:        float
        """
        topic = str(Topic.VAX)
        agent_i = [a for a in self.schedule.agents if a.unique_id == self.n_agents - 1][0]
        belief = agent_i.beliefs[topic]
        return belief


# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
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
