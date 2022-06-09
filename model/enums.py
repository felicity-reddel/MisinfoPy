from enum import Enum

# import random


class Topic(Enum):
    """
    Implemented Topics (for tweet_beliefs of posts & tweet_beliefs of agents).
    Easily extendable to include more topics (e.g., MASKS, EVOLUTION, etc.)
    """

    VAX = 0

    # MASKS = 1
    # EVOLUTION = 2

    @staticmethod
    def get_random(rng):
        """
        Samples Topic completely independent of the tweet_beliefs represented in the post.
        @param rng: Random number generator from mesa or normal Python random
        :return: result: Topic
        """
        result = rng.choice(list(Topic))
        return result


class GroundTruth(Enum):
    """
    Enumeration representing the groundtruth n_seen_posts_repl of a post.
    """

    FALSE = 0
    TRUE = 1

    @staticmethod
    def get_random(rng):
        """
        Samples GroundTruth completely independent of the tweet_beliefs represented in the post.
        @param rng: Random number generator from mesa or normal Python random
        :return: result: GroundTruth
        """
        result = rng.choice(list(GroundTruth))
        return result

    @staticmethod
    def get_groundtruth(rng, tweet_belief=50.0):
        """
        Simple implementation to sample the groundtruth of a post by using the post's belief as the probability that
        GroundTruth.TRUE.

        Assumes:
        – GroundTruth has only two possible values (TRUE, FALSE)
        – higher post_beliefs are more likely to be true

        @param tweet_belief: float, range [0.0, 100.0]
        @param rng: Random number generator from mesa or normal Python random
        :return: GroundTruth
        """
        # Transform belief into probability
        p_true = tweet_belief / 100

        # Weighted sampling to set groundtruth of post
        groundtruth = rng.choices(
            population=[GroundTruth.TRUE, GroundTruth.FALSE],
            weights=[p_true, 1 - p_true],
        )[0]

        return groundtruth


class MediaLiteracy(Enum):
    """
    Media Literacy Levels.
    The n_seen_posts_repl represents the duration of a user for judging the truthfulness of a post
    (on average, time in seconds).
    """

    LOW = 3
    HIGH = 30

    @staticmethod
    def get_random(rng, mlit_weights=None):
        """
        Samples MediaLiteracy completely independent of the tweet_beliefs represented in the post.
        @param rng: Random number generator from mesa or normal Python random
        @param mlit_weights: list of two floats, for MediaLiteracy.LOW and MediaLiteracy.HIGH, respectively
        :return: result: MediaLiteracy
        """
        if mlit_weights is None:
            mlit_weights = [0.7, 0.3]
        result = rng.choices(population=list(MediaLiteracy), weights=mlit_weights, k=1)[
            0
        ]
        return result


class SelectAgentsBy(Enum):
    """
    Possibilities to select agents. E.g., for who will be empowered by the Media Literacy Intervention.
    Easily extendable to e.g., pick agents based on an agent-characteristic (e.g., age, if age is an agent attribute).
    """

    RANDOM = 0
    # HIGH_AGE = 1
    # LOW_AGE = 2


class BeliefUpdate(Enum):
    """
    Different options for updating tweet_beliefs.
    In order to explore structural uncertainty of the belief update function.
    """

    SAMPLE = 0  # Sampling, used by e.g., Sari et al. (2021), Wang et al. (2019), and Gausen et al. (2021)
    DEFFUANT = 1  # Deffuant, used by e.g., Du et al. (2021), Rajabi et al. (2020), and Mason et al. (2020)
    SIT = 2  # Social Impact Theory, adjusted to the social media environment by Reddel (2021)


class Metrics(Enum):
    """
    The considered objectives.
    """

    N_AGENTS_ABOVE_BELIEF_THRESHOLD = 0
    POLARIZATION_VARIANCE = 1
    ENGAGEMENT = 2
    FREE_SPEECH_CONSTRAINT = 3
    AVG_USER_EFFORT = 4
