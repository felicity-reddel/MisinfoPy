import numpy as np
from enums import *


class Post:

    def __init__(self, unique_id, source, stances=None, p_true_threshold_ranking=0.1):
        self.unique_id = unique_id
        self.source = source
        if stances is None:
            self.stances = {}
        else:
            # stances represented in the post. self.stances is {Topic: int_belief}
            self.stances = self.sample_stances(based_on_agent=self.source)
        self.visibility = self.estimate_visibility()
        self.ground_truth = GroundTruth.get_groundtruth(post_belief=self.stances[Topic.VAX])
        self.p_true = self.factcheck_algorithm()
        self.visibility_ranking_intervention = self.get_adjusted_visibility(p_true_threshold_ranking)

    @staticmethod
    def sample_stances(max_n_topics=1, based_on_agent=None) -> dict:
        """
        Generates and returns dict of stances for one post (i.e., topic & value):  {Topic.TOPIC1: int}
        :param max_n_topics:    int,    maximal number of topics in one post
        :param based_on_agent:  Agent,  if None: generate random belief,
                                        if agent: generate post-stances based that agent's beliefs
        :return: dict of stances (i.e., topics with value)
        """
        # Sample how many topics should be included in post.
        n_topics = random.randint(1, max_n_topics)  # min. 1 topic per post

        # Sample stances (stance = topic with value)
        stances = {}

        for _ in range(n_topics):

            # Pick topic
            topic = Topic.get_random()  # Ext: could adjust weights for diff. topics

            # Sample value on topic
            if based_on_agent is None:
                value = random.randint(0, 100)
            else:
                current_belief = based_on_agent.beliefs[topic]
                value = np.random.normal(loc=current_belief, scale=5, size=1)[0]
                value = max(min(value, 100), 0)

            stances[topic] = value

        return stances

    def factcheck_algorithm(self, topic=Topic.VAX):
        """
        Simulates the factcheck algorithm.
        The factcheck algorithm assigns the post a probability of having GroundTruth.TRUE by using the post's belief.
        This probability is returned.
        :param topic: Topic
        :return: float, in range [0,1]
        """
        p_true = self.stances[topic]/100
        return p_true

    def estimate_visibility(self):
        """
        Estimates the visibility of the post.
        Here: just the extremeness of its stances
        :return:    float
        """

        extremeness = self.calculate_extremeness()
        engagement = extremeness

        return engagement

    def calculate_extremeness(self):
        """
        Calculates how extreme the post is.
        Here: the posts extremeness = the average extremeness of all its stances.
        :return: float  [0,1)
        """
        stances = self.stances.values()
        extremeness_values = []
        for stance in stances:
            extremeness = abs(50 - stance)
            extremeness_values.append(extremeness)

        avg_extremeness = sum(extremeness_values) / len(extremeness_values)

        # Scale to domain [0,1)
        avg_extremeness /= 50

        return avg_extremeness

    def get_adjusted_visibility(self, p_true_threshold_ranking=0.1):
        """
        The ranking intervention is applied. This method adjusts the visibility of the posts if the factcheck result
        is of sufficient certainty that the post is false (i.e., post has a sufficiently low p_true).
        This adjustment is dependent on the Post's GroundTruth:
            if TRUE     -> same visibility
            if FALSE    -> visibility reduced by 50%
        :return:  float, [0,1)
        """
        adjusted_visibility = self.visibility
        if self.stances[Topic.VAX] <= p_true_threshold_ranking:
            adjusted_visibility = self.visibility * self.ground_truth.value

        return adjusted_visibility
