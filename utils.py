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
