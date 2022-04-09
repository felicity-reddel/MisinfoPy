import numpy as np
from enums import *


class Post:

    def __init__(self, unique_id, source, stances=None):
        self.unique_id = unique_id
        self.source = source
        if stances is None:
            self.stances = {}
        else:
            # stances represented in the post. self.stances is {Topic: int_belief}
            self.stances = self.sample_stances(based_on_agent=self.source)
        self.visibility = self.estimate_visibility()
        self.ground_truth = GroundTruth.get_groundtruth(stances=self.stances)
        self.factcheck_result = self.factcheck_algorithm()
        self.visibility_ranking_intervention = self.get_adjusted_visibility()

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

    def factcheck_algorithm(self, accuracy=0.8):
        """
        Simulates the factcheck algorithm.
        Accuracy is the probability that the post's ground_truth and the overall leaning of the factcheck_result agree.
        The overall leaning of the factcheck_result:
            is TRUE if: p_false < 0.5
            is FALSE if: p_false >= 0.5

        Consequently,
        If GroundTruth.TRUE: return with 'accuracy'-probability: p_false in range [0.0, 0.5).
                                    and with 1-'accuracy': p_false in range [0.5, 1.0).
        If GroundTruth.FALSE: return with 'accuracy'-probability: p_false in range [0.5, 1.0).
                                    and with 1-'accuracy': p_false in range [0.0, 0.5).

        :return: float, in range [0,1]
        """
        correct_category = True if random.uniform(0.0, 1.0) < accuracy else False

        # Categorized as misinfo -> p_false: [0.5, 1.0)
        if ((self.ground_truth == GroundTruth.FALSE and correct_category) or
                (self.ground_truth == GroundTruth.TRUE and not correct_category)):
            lower, higher = 0.5, 1.0

        # Categorized as true -> p_false: [0.0, 0.5)
        elif ((self.ground_truth == GroundTruth.TRUE and correct_category) or
              (self.ground_truth == GroundTruth.FALSE and not correct_category)):
            lower, higher = 0.0, 0.5

        # Catch-all
        else:
            print("The factcheck_algorithm can currently on handle GroundTruth.TRUE and GroundTruth.FALSE.")

        p_false = random.uniform(lower, higher)

        return p_false

        # TODO: continue implementation

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

    def get_adjusted_visibility(self):
        """
        If the ranking intervention is applied, this method adjusts the visibility of the posts.
        This adjustment is dependent on the Post's GroundTruth:
            if TRUE     -> same visibility
            if FALSE    -> visibility reduced by 50%
        :return:  float, [0,1)
        """
        adjusted_visibility = self.visibility * self.ground_truth.value

        return adjusted_visibility
