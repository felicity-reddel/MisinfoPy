from enum import Enum
import random


class Topic(Enum):
    """
    Implemented Topics (for stances of posts & beliefs of agents).
    Easily extendable to include more topics (e.g., MASKS, EVOLUTION, etc.)
    """
    VAX = 0

    # MASKS = 1
    # EVOLUTION = 2

    @staticmethod
    def get_random():
        """
        Samples Topic completely independent of the post's stance.
        :return: result: Topic
        """
        result = random.choice(list(Topic))
        return result


class GroundTruth(Enum):
    """
    Enumeration representing the ground truth value of a post.

    Each value represents the adjustment in visibility.  # TODO: Adjust the comment (wrt the below TODO)
    i.e., if ranking_invervention is on AND GroundTruth.FALSE --> post has only 50% of its previous visibility.
    """

    FALSE = 0.5  # TODO: Change to 0 (after ranking intervention has been adjusted)
    TRUE = 1

    @staticmethod
    def get_random():
        """
        Samples GroundTruth completely independent of the post's stance.
        :return: result: GroundTruth
        """
        result = random.choice(list(GroundTruth))
        return result

    @staticmethod
    def get_groundtruth(stances, based_on_topic=Topic.VAX):
        """
        Returns GroundTruth dependent on the post's stance.
        if post's stance is between
                            - 0 and 20:     100% that FALSE, 0% that TRUE
                            - 20 and 80:    50% that FALSE, 50% that TRUE
                            - 80 and 100:   20% that FALSE, 80% that TRUE
        :param based_on_topic:
        :param stances: dict, {Topic: value}
        :return: GroundTruth
        """
        result = GroundTruth.TRUE
        p_false = GroundTruth.get_probability_that_false(stances, based_on_topic)

        # "Coin toss"
        random_nr = random.random()
        if random_nr < p_false:
            result = GroundTruth.FALSE

        return result

    @staticmethod
    def get_probability_that_false(stances, based_on_topic=Topic.VAX):
        """
        Returns the probability-value used for initializing a Post's GroundTruth.
        The probability value represents the probability that the post is FALSE.
        TODO: Grounding the probabilities.
         Currently: if post's stance is between
                            - 0 and 20:     100% that FALSE, 0% that TRUE
                            - 20 and 80:    50% that FALSE, 50% that TRUE
                            - 80 and 100:   20% that FALSE, 80% that TRUE

        :param stances:         dict,  {Topic: value}
        :param based_on_topic:  Topic
        :return:                float, [0,1)
        """
        topic = based_on_topic
        value = stances[topic]

        if value <= 20:
            probability = 0.0
        elif value <= 80:
            probability = 0.5
        else:
            probability = 0.8

        return probability


class MediaLiteracy(Enum):
    """
    Media Literacy Levels
    """

    LOW = 0
    HIGH = 1

    @staticmethod
    def get_random():
        """
        Samples MediaLiteracy completely independent of the post's stance.
        :return: result: MediaLiteracy
        """
        result = random.choice(list(MediaLiteracy))
        return result


class SelectAgentsBy(Enum):
    """
    Possibilities to select agents. E.g., for who will by empowered by the Media Literacy Intervention.
    Easily extendable to e.g., pick agents based on an agent-characteristic (e.g., age, if age is an agent attribute).
    """
    RANDOM = 0
    # HIGH_AGE = 1
    # LOW_AGE = 2


class BeliefUpdate(Enum):
    """
    Different options for updating beliefs. In order to explore structural uncertainty of the belief update function.

    Characteristics of the models' belief update functions:
    - M0:       Bounded Confidence      +       homogeneous μ (update parameter)
    - M1:       No Bounded Confidence   +       homogeneous μ (update parameter)
    - M2:       Bounded Confidence      +       heterogeneous μ (update parameter)
    - M3:       No Bounded Confidence   +       heterogeneous μ (update parameter)
    """

    M0 = 0
    M1 = 1
    M2 = 2
    M3 = 3
