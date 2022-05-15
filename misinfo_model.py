import pandas as pd
from agents import *  # incl.: utils import
import time
import statistics
from mesa import Model
from mesa.datacollection import DataCollector
from mesa.time import StagedActivation
from mesa.space import NetworkGrid
import numpy as np
import math
from matplotlib import pyplot as plt
# # To profile the code:
# import cProfile, pstats, io
#
#
# def profile(fnc):
#     """Decorator using cProfile to profile a provided function"""
#
#     def inner(*args, **kwargs):
#
#         pr = cProfile.Profile()
#         pr.enable()
#         retval = fnc(*args, **kwargs)
#         pr.disable()
#         s = io.StringIO()
#         sortby = 'cumulative'
#         ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
#         ps.print_stats()
#         print(s.getvalue())
#         return retval
#
#     return inner


class MisinfoPy(Model):
    """Simple model with n agents."""

    def __init__(  # New, Minimal init
            self,
            # ––– Network –––
            n_agents=1000,
            n_edges=2,
    ):
        """
        Initializes the MisinfoPy.

        ––– Network –––
        @param n_agents:                    int, how many agents the model should have
        @param n_edges:                     int, with how many edges gets attached to the already built network
        """

        super().__init__()

        # ––– Network & Setup –––
        self.n_agents = n_agents
        self.n_edges = n_edges
        self.schedule = None  # StagedActivation(self, stage_list=["share_post_stage", "update_beliefs_stage"])
        self.G = None
        self.grid = None
        self.post_id_counter = 0
        self.agents_data = {'n_followers_range': (0, 0),
                            'n_following_range': (0, 0)}

        # ––– Posting behavior –––
        self.sigma = None
        self.mean_normal_user = None
        self.mean_disinformer = None
        self.adjustment_based_on_belief = None

        # ––– Levers –––
        self.mlit_dur_init = None
        self.mlit_dur_low = None
        self.mlit_dur_high = None

        self.del_t = None
        self.rank_t = None
        self.rank_punish = None
        self.strikes_t = None

        # ––– Belief updating behavior –––
        self.belief_update_fn = None
        self.sampling_p_update = None
        self.deffuant_mu = None

        # ––– Plots –––
        self.belief_metric_threshold = None

        # ––– Data Collectors –––
        self.n_posts_deleted = 0
        self.data_collector = None

    def set_up(
            self,
            # ––– Network –––
            ratio_normal_user=0.99,

            # ––– Posting behavior –––
            sigma=0.7,
            mean_normal_user=1,
            mean_disinformer=10,
            adjustment_based_on_belief=2,

            # ––– Levers –––
            mlit_select=0.0,
            mlit_dur_init=3600,
            mlit_dur_low=3,
            mlit_dur_high=30,
            rank_punish=-0.0,
            del_t=0.0,
            rank_t=0.0,
            strikes_t=0.0,

            # ––– Belief updating behavior –––
            belief_update_fn=BeliefUpdate.SIT,
            sampling_p_update=0.02,
            deffuant_mu=0.02,
            belief_metric_threshold=50.0,

            # ––– Plots –––
            show_n_connections=False
    ):
        """
        Sets up the initial, barebone MisinfoPy model.

        ––– Network –––
        @param ratio_normal_user:           float, in range [0.0, 1.0]

        ––– Posting behavior –––
        @param sigma:                       float, std_dev to sample from a normal distribution how many posts an agent
                                            want to post in a tick ( n_posts )
        @param mean_normal_user:            float, mean of normal user for sampling n_posts
        @param mean_disinformer:            float, mean of normal user for sampling n_posts
        @param adjustment_based_on_belief:  float, the extremeness of an agent adjusts the mean for sampling n_posts

        ––– Levers –––
        @param mlit_select: float, in domain [0,1], Percentage of agents empowered by media literacy intervention.
        @param mlit_dur_init: int, how long the initial media literacy intervention takes for a user
        @param mlit_dur_low: int, how long a person with LOW media literacy takes to judge the truthfulness of a post
        @param mlit_dur_high: int, how long a person with HIGH media literacy takes to judge the truthfulness of a post
        @param rank_punish: float, range [-0.0, -1.0], relative visibility change for posts with GroundTruth.FALSE
        @param del_t: float, range [0.0, 1.0], p_true_threshold for deleting (below -> may be deleted)
        @param rank_t: float, range [0.0, 1.0], p_true_threshold for down-ranking (below -> may be down-ranked)
        @param strikes_t: float, range [0.0, 1.0], p_true_threshold for strike system (below -> may result in strike)

        ––– Belief updating behavior –––
        @param belief_update_fn:            BeliefUpdate (enum)
        @param sampling_p_update:           float, probability that the agent will update
        @param deffuant_mu:                 float, updating parameter, indicates how strongly the belief is updated
                                            towards the post's belief. If mu=0.1, the update is 10% towards the
                                            post's belief.
        @param belief_metric_threshold:     float, threshold for the belief metric (agents above belief threshold)

        ––– Plots –––
        @param show_n_connections:          boolean
        """
        # ––– Network & Setup –––
        self.G = random_graph(n_nodes=self.n_agents, m=self.n_edges)  # n_nodes = n_agents, exactly 1 agent per node
        self.grid = NetworkGrid(self.G)
        self.schedule = StagedActivation(self, stage_list=["share_post_stage", "update_beliefs_stage"])

        # ––– Posting behavior –––
        self.sigma = sigma
        self.mean_normal_user = mean_normal_user
        self.mean_disinformer = mean_disinformer
        self.adjustment_based_on_belief = adjustment_based_on_belief

        # ––– Levers –––
        self.mlit_dur_init = mlit_dur_init
        self.mlit_dur_low = mlit_dur_low
        self.mlit_dur_high = mlit_dur_high
        self.apply_media_literacy_intervention(mlit_select)
        if not (-1.0 <= rank_punish <= -0.0):
            raise ValueError(f"Visibility adjustment for ranking was {rank_punish}, "
                             f"while it should be in range [-0.0, -1.0]")
        self.del_t = del_t
        self.rank_t = rank_t
        self.rank_punish = rank_punish
        self.strikes_t = strikes_t

        # ––– init agents –––
        self.init_agents(ratio_normal_user)
        self.init_followers_and_following()

        # ––– Belief updating behavior –––
        self.belief_update_fn = belief_update_fn
        self.sampling_p_update = sampling_p_update
        self.deffuant_mu = deffuant_mu

        # ––– Plots –––
        self.belief_metric_threshold = belief_metric_threshold

        # self.data_collector = DataCollector(model_reporters={
        #     "n_agents_above_belief_threshold": self.get_n_above_belief_threshold,
        #     "engagement": self.get_total_seen_posts,
        #     "free_speech_constraint": self.get_free_speech_constraint,
        #     "avg_user_effort": self.get_avg_user_effort
        # })

        # # For additional plots in the browser
        # "Avg Vax-Belief": self.get_avg_belief,
        # "Avg Vax-Belief above threshold": lambda m: m.get_avg_belief_above_threshold(
        #     threshold=belief_metric_threshold),
        # "Avg Vax-Belief below threshold": lambda m: m.get_avg_belief_below_threshold(
        #     threshold=belief_metric_threshold)})

        # n_depicted = 10
        # ids = [*range(0, n_agents, math.floor(n_agents / n_depicted))]
        # DataCollector2: follow individual agents
        # self.data_collector2 = DataCollector(model_reporters={  # As many as n_depicted
        #     f"Agent 0": lambda m: m.get_belief(agent_id=ids[0]),
        #     f"Agent 1": lambda m: m.get_belief(agent_id=ids[1]),
        #     f"Agent 2": lambda m: m.get_belief(agent_id=ids[2]),
        #     f"Agent 3": lambda m: m.get_belief(agent_id=ids[3]),
        #     f"Agent 4": lambda m: m.get_belief(agent_id=ids[4]),
        #     f"Agent 5": lambda m: m.get_belief(agent_id=ids[5]),
        #     f"Agent 6": lambda m: m.get_belief(agent_id=ids[6]),
        #     f"Agent 7": lambda m: m.get_belief(agent_id=ids[7]),
        #     f"Agent 8": lambda m: m.get_belief(agent_id=ids[8]),
        #     f"Agent 9": lambda m: m.get_belief(agent_id=ids[9]),
        # })

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
        # self.data_collector.collect(self)
        # self.data_collector2.collect(self)

    # @profile
    def __call__(
        self,
        # ––– Network –––
        ratio_normal_user=0.99,

        # ––– Posting behavior –––
        sigma=0.7,
        mean_normal_user=1,
        mean_disinformer=10,
        adjustment_based_on_belief=2,

        # ––– Levers –––
        mlit_select=0.0,
        mlit_dur_init=3600,
        mlit_dur_low=3,
        mlit_dur_high=30,
        rank_punish=-0.0,
        del_t=0.0,
        rank_t=0.0,
        strikes_t=0.0,

        # ––– Belief updating behavior –––
        belief_update_fn=BeliefUpdate.SIT,
        sampling_p_update=0.02,
        deffuant_mu=0.02,
        belief_metric_threshold=50.0,

        # ––– Call parameters –––
        steps=60,
        time_tracking=False,
        debug=False,
    ):
        """
        Runs the model for the specified number of steps.

        @param steps:           int, number of model-steps the model should take
        @param time_tracking:   Boolean, whether to print timing information
        @param debug:           Boolean, whether to print details
        @return: tuple: metrics values for this run (n_above_belief_threshold, variance, ...)
        """

        self.set_up(
            # ––– Network –––
            ratio_normal_user=ratio_normal_user,

            # ––– Posting behavior –––
            sigma=sigma,
            mean_normal_user=mean_normal_user,
            mean_disinformer=mean_disinformer,
            adjustment_based_on_belief=adjustment_based_on_belief,

            # ––– Levers –––
            mlit_select=mlit_select,
            mlit_dur_init=mlit_dur_init,
            mlit_dur_low=mlit_dur_low,
            mlit_dur_high=mlit_dur_high,
            rank_punish=rank_punish,
            del_t=del_t,
            rank_t=rank_t,
            strikes_t=strikes_t,

            # ––– Belief updating behavior –––
            belief_update_fn=belief_update_fn,
            sampling_p_update=sampling_p_update,
            deffuant_mu=deffuant_mu,
            belief_metric_threshold=belief_metric_threshold,
        )

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

        results_dict = {
            'n_agents_above_belief_threshold': n_agents_above_belief_threshold,
            'polarization_variance': polarization_variance,
            'polarization_kl_divergence_from_polarized': polarization_kl_divergence_from_polarized,
            'engagement': engagement,
            'free_speech_constraint': free_speech_constraint,
            'avg_user_effort': avg_user_effort,
        }

        # print()
        # pretty = pd.Series(results_dict)
        # print(pretty)

        return results_dict

    # –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    # Init functions
    # –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    def init_agents(self, ratio_normal_user):
        """Initializes the agents.
        @param ratio_normal_user: float, in range [0.0, 1.0]
        """
        types = [NormalUser, Disinformer]
        percentages = [ratio_normal_user, 1 - ratio_normal_user]

        # Create agents & add them to the scheduler
        for i in range(self.n_agents):
            # Pick which type should be added
            agent_type = random.choices(population=types, weights=percentages, k=1)[0]

            # Add agent of that type
            if agent_type is NormalUser:
                a = NormalUser(i, self)
                self.schedule.add(a)
            elif agent_type is Disinformer:
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

    def apply_media_literacy_intervention(self, percentage_selected=0.0, select_by=SelectAgentsBy.RANDOM):
        """
        Applies the media literacy intervention (if needed).
        @param percentage_selected: float, [0,1], Percentage of agents empowered by media literacy intervention.
        @param select_by: SelectAgentsBy, enum how to select agents
        """
        # If media literacy intervention is used: select agents for intervention, adjust their media literacy.
        # (i.e., if some percentage of agents is targeted with it)
        if percentage_selected > 0.0:
            n_select = int(len(self.schedule.agents) * percentage_selected)
            selected_agents = self.select_agents_for_media_literacy_intervention(n_select, select_by)

            for agent in selected_agents:
                agent.media_literacy = MediaLiteracy.HIGH
                agent.received_media_literacy_intervention = True

    def select_agents_for_media_literacy_intervention(self, n_select=0, select_by=SelectAgentsBy.RANDOM):
        """
        Select agents for the intervention.
        @param n_select:    int, how many agents should be selected for the intervention
        @param select_by:   SelectBy(Enum), selection method, e.g. SelectBy.RANDOM
        @return:            list of agents, [(Base)Agent, (Base)Agent, ...]
        """
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
        Returns a list of tweet_beliefs that includes all agents' belief on Topic.VAX.
        @return: list
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
        @return: float, range [0.0, 1.0]
        """

        total_posts = sum([a.preferred_n_posts for a in self.schedule.agents])
        n_posted = sum([len(a.visible_posts) for a in self.schedule.agents])
        n_prevented = total_posts - n_posted
        n_downranked = sum([a.n_downranked for a in self.schedule.agents])

        constraint = n_prevented + (n_downranked * abs(self.rank_punish))
        rel_constraint = constraint / total_posts

        return rel_constraint

    def get_avg_user_effort(self):
        """
        Calculates an estimate of the effort a user has for
            - judging the posts' truthfulness (summed over all posts)
            - partaking in the media literacy intervention, i.e., the initial investment (once)
        The effort is measured in minutes. The average effort per user is returned.

        @return: float, cannot be lower than zero
        """
        effort_per_agent = []
        for agent in self.schedule.agents:
            initial_investment = int(agent.received_media_literacy_intervention) * self.mlit_dur_init
            if agent.media_literacy == MediaLiteracy.HIGH:
                judging_all_posts = agent.n_total_seen_posts * self.mlit_dur_high
            elif agent.media_literacy == MediaLiteracy.LOW:
                judging_all_posts = agent.n_total_seen_posts * self.mlit_dur_low
            else:
                raise ValueError("Only MediaLiteracy.HIGH and MediaLiteracy.LOW are currently implemented.")
            effort = initial_investment + judging_all_posts
            effort_per_agent.append(effort)

        avg_effort_seconds = sum(effort_per_agent) / len(self.schedule.agents)
        avg_effort_minutes = round(avg_effort_seconds / 60.0, 2)

        return avg_effort_minutes

    def get_posts_per_month(self):
        """
        For additonal calibration of agent vocality.
        Returns a list of how many posts each agent has been posting over the whole run.
        @return: list of floats
        """
        post_totals = [len(a.visible_posts) for a in self.schedule.agents]
        n_days = self.schedule.time
        n_months = n_days / 30

        posts_per_month = [total / n_months for total in post_totals]

        return posts_per_month

    def get_median_errors(self):
        """
        For additonal calibration of agent vocality.
        Calculates the error with respect to the
        'median number of posts within the agents that posted the most posts (top-10%)' and
        'median number of posts within the agents that did not post the most posts (bottom-90%)'.

        @return: 2-tuple of floats, both in domain [0.0, math.inf)
        """
        # Gather data & make sure the list is in ascending order
        posts_per_month = self.get_posts_per_month()
        posts_per_month.sort()  # by default in ascending order

        # Find the cutoff index & split the list into two
        cutoff = round(len(posts_per_month) * 0.9)
        top_10 = posts_per_month[cutoff:]
        bottom_90 = posts_per_month[:cutoff]

        # Get the median within both lists
        top_10_median = np.median(top_10)
        bottom_90_median = np.median(bottom_90)

        # Calculate the two errors
        top_10_error = abs(self.top_10_target - top_10_median)
        bottom_90_error = abs(self.bottom_90_target - bottom_90_median)

        return top_10_error, bottom_90_error

    def get_avg_belief(self, topic=Topic.VAX) -> float:
        """
        Return average belief of all agents on a given topic. For the DataCollector.
        @param topic:   Topic
        @return:        float
        """

        agent_beliefs = [a.beliefs[topic] for a in self.schedule.agents]
        avg_belief = sum(agent_beliefs) / len(agent_beliefs)

        return avg_belief

    def get_total_seen_posts(self):
        """
        Returns the total number of seen posts, summed over all agents.
        @return: total: int
        """

        per_agent = [sum(x.n_seen_posts) for x in self.schedule.agents]
        total = sum(per_agent)

        return total

    def get_topic_above_below_sizes(self, topic=Topic.VAX, threshold=50.0) -> tuple:
        """
        Return tuple of how many agents' belief on a given topic is above and below the provided threshold.
         For the DataCollector.
        @param topic:       Topic
        @param threshold:   float
        @return:            tuple
        """

        agent_beliefs = [a.beliefs[topic] for a in self.schedule.agents]
        n_above = sum([1 for a_belief in agent_beliefs if a_belief >= threshold])
        n_below = len(agent_beliefs) - n_above

        return n_above, n_below

    def get_n_above_belief_threshold(self, topic=Topic.VAX, threshold=50) -> int:
        """
        Returns how many agents' belief on a given topic is above and below the provided threshold.
         For the DataCollector.
        @param topic:       Topic
        @param threshold:   float
        @return: n_above:   int
        """

        agent_beliefs = [a.beliefs[topic] for a in self.schedule.agents]
        n_above = sum([1 for a_belief in agent_beliefs if a_belief >= threshold])

        return n_above

    def get_n_below_belief_threshold(self, topic=Topic.VAX, threshold=50) -> int:
        """
        Returns how many agents' belief on a given topic are below the provided threshold.
         For the DataCollector.
        @param topic:       Topic
        @param threshold:   float
        @return: n_below    int
        """

        agent_beliefs = [a.beliefs[topic] for a in self.schedule.agents]
        n_below = sum([1 for a_belief in agent_beliefs if a_belief < threshold])

        return n_below

    def get_avg_belief_above_threshold(self, topic=Topic.VAX, threshold=50) -> float:
        """
        Returns the average belief of agents that are above the provided threshold.
         For the DataCollector.
        @param topic:       Topic
        @param threshold:   float
        @return: avg:       float
        """

        beliefs_above_threshold = [a.beliefs[topic] for a in self.schedule.agents if a.beliefs[topic] >= threshold]
        if len(beliefs_above_threshold) == 0:
            avg = self.get_avg_belief_below_threshold()  # If nobody above threshold, take avg of below threshold.
        else:
            avg = sum(beliefs_above_threshold) / len(beliefs_above_threshold)
        return avg

    def get_avg_belief_below_threshold(self, topic=Topic.VAX, threshold=50) -> float:
        """
        Returns the average belief of agents that are below the provided threshold.
         For the DataCollector.
        @param topic:       Topic
        @param threshold:   float
        @return: avg:       float
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
        @param topic:       Topic
        @param agent_ids_list: list of agent.unique_id's
        @return: dict, {unique_id: vax_belief}
        """
        if agent_ids_list is None:
            agent_ids_list = [a.unique_id for a in self.schedule.agents]
        vax_beliefs: dict[str, float] = {}
        agents = [a for a in self.schedule.agents if a.unique_id in agent_ids_list]
        for agent in agents:
            belief = agent.beliefs[topic]
            vax_beliefs[f'belief of agent {id}'] = belief

        return vax_beliefs

    def get_belief(self, agent_id=None, topic=Topic.VAX) -> float:
        """
        Returns the belief of agent 0 at current tick.
        For data_collector2.
        @param agent_id:    int
        @param topic:       Topic
        @return: belief:    float
        """
        try:
            agent_i = [a for a in self.schedule.agents if a.unique_id == agent_id][0]
        except ValueError:
            raise ValueError(f"There is no agent with unique_id {agent_id}.")
        belief = agent_i.beliefs[topic]
        return belief


def random_graph(n_nodes, m, seed=None, directed=True) -> nx.Graph:
    """
    Generates a random graph à la Barabasi Albert.
    @param n_nodes:     int, number of nodes
    @param m:           int, avg number of edges added per node
    @param seed:        int, random seed
    @param directed:    bool, undirected or directed graph

    @return:            nx.Graph, the resulting stochastic graph (barabasi albert G)

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


if __name__ == '__main__':
    model = MisinfoPy(n_agents=100)
    results = model(steps=15)

    [print(r) for r in results]
