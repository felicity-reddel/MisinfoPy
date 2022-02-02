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

    def __eq__(self, o: object) -> bool:
        if self.value is o.value:
            return True
        else:
            return False

    @staticmethod
    def get_random():
        """
        Samples Topic completely independent of the post's stance.
        :return: result: Topic
        """
        result = random.choice(list(Topic))
        return result


class FactCheckResult(Enum):
    """
    Enumeration representing the a factcheck would have (i.e., the ground truth).
    Easily extendable to include more options (e.g., MISLEADING)

    Each value represents the adjustment in visibility.
    i.e., if ranking_invervention is on AND FactCheckResult.FALSE --> post has only 50% of its previous visibility.
    """

    FALSE = 0.5
    TRUE = 1
    # MISLEADING = 0.75

    def __eq__(self, o: object) -> bool:
        if self.value is o.value:
            return True
        else:
            return False

    @staticmethod
    def get_random():
        """
        Samples FactCheckResult completely independent of the post's stance.
        :return: result: FactCheckResult
        """
        result = random.choice(list(FactCheckResult))
        return result

    @staticmethod
    def sample(stances, based_on_topic=Topic.VAX):
        """
        Samples FactCheckResult completely dependent on the post's stance.
        if post's stance is between
                            - 0 and 20:     100% that FALSE, 0% that TRUE
                            - 20 and 80:    50% that FALSE, 50% that TRUE
                            - 80 and 100:   20% that FALSE, 80% that TRUE
        :param based_on_topic:
        :param stances: dict, {Topic: value}
        :return: FactCheckResult
        """
        result = FactCheckResult.FALSE
        probability = FactCheckResult.get_ground_truth_probability(stances, based_on_topic)

        # "Coin toss"
        random_nr = random.random()
        if random_nr < probability:
            result = FactCheckResult.TRUE

        return result

    @staticmethod
    def get_ground_truth_probability(stances, based_on_topic=Topic.VAX):
        """
        Returns the probability used for initializing a Post's FactCheckResult.
        :param stances:         dict,  {Topic: value}
        :param based_on_topic:  Topic
        :return:                float, [0,1)
        """
        topic = str(based_on_topic)
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

    def __eq__(self, o: object) -> bool:
        if self.value is o.value:
            return True
        else:
            return False

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

    def __eq__(self, o: object) -> bool:
        if self.value is o.value:
            return True
        else:
            return False
