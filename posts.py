import numpy as np
from enums import GroundTruth, MediaLiteracy, SelectAgentsBy, BeliefUpdate
from utils import *


class Post:

    def __init__(self, unique_id, source, tweet_beliefs=None, rank_t=0.0):
        """
        :param unique_id: int
        :param source: Agent (NormalUser or Disinformer)
        :param tweet_beliefs: dict, {Topic: int_belief}, tweet_beliefs to be represented in the post.
        :param rank_t: float, below which the ranking intervention decreases the post's visibility
        """
        self.unique_id = unique_id
        self.source = source

        if tweet_beliefs is None:
            self.tweet_beliefs = self.sample_beliefs(agent=self.source)
        else:
            self.tweet_beliefs = tweet_beliefs

        self.ground_truth = GroundTruth.get_groundtruth(tweet_belief=self.tweet_beliefs[Topic.VAX])
        self.p_true = self.factcheck_algorithm()
        self.detected_as_misinfo = self.detected_as_misinfo()
        self.visibility = self.get_visibility(rank_t)

    # @staticmethod

    def factcheck_algorithm(self, topic=Topic.VAX):
        """
        Simulates the factcheck algorithm.
        The factcheck algorithm assigns the post a probability of having GroundTruth.TRUE by using the post's belief.
        This probability is returned.
        :param topic: Topic
        :return: float, in range [0,1]
        """
        p_true = self.tweet_beliefs[topic] / 100
        return p_true

    def estimate_visibility(self):
        """
        Estimates the visibility of the post.
        Here: just the extremeness of its tweet_beliefs
        :return:    float
        """
        extremeness = calculate_extremeness(self.tweet_beliefs)
        engagement = extremeness

        return engagement

    def get_visibility(self, rank_t=0.1):
        """
        The ranking intervention is applied. This method adjusts the visibility of the posts if the factcheck result
        is of sufficient certainty that the post is false (i.e., post has a sufficiently low p_true).
        This adjustment is dependent on the Post's GroundTruth:
            if TRUE     -> same visibility
            if FALSE    -> visibility reduced by 50%
        :param rank_t: float, range [0.0, 1.0]
        :return:  float, [0,1)
        """
        visibility = self.estimate_visibility()
        # Visibility adjustment for (~41% of) posts that are factchecked as false (with high certainty).
        # source: brennen_2020 (see below)
        if (self.tweet_beliefs[Topic.VAX] <= rank_t) and self.detected_as_misinfo:
            visibility *= (1 + self.source.model.rank_punish)
            self.source.n_downranked += 1

        return visibility

    def detected_as_misinfo(self, p_detected=0.41):
        """
        Determines whether a misinfo-post is detected as such.

        Default p_detected based on Brennen et al., 2020. (59% stays up)
        @misc{brennen_2020,
            title={Types, sources, and claims of covid-19 misinformation},
            url={https://reutersinstitute.politics.ox.ac.uk/types-sources-and-claims-covid-19-misinformation},
            journal={Reuters Institute for the Study of Journalism},
            author={Brennen, Scott and Simon, Felix M and Howard, Philip N and Kleis Nielsen, Rasmus},
            year={2020},
            month={Apr}
        }

        :param p_detected: float, in range [0.0, 1.0]
        :return: Boolean
        """
        rd = self.source.model.random
        detected = rd.choices(population=[True, False], weights=[p_detected, 1-p_detected])[0]

        return detected
