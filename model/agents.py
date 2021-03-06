from model.posts import *

# import numpy as np

import math

# import random

from mesa import Agent


class BaseAgent(Agent):
    """Most simple agent to start with."""

    def __init__(self, unique_id, model):
        """
        :param unique_id: int
        :param model: MisinfoPy
        """
        super().__init__(unique_id, model)

        self.beliefs = {}
        self.init_beliefs()
        self.media_literacy = MediaLiteracy.LOW
        self.received_media_literacy_intervention = False

        self.vocality = {"mean": 0, "std_dev": self.model.sigma}
        self.followers = []
        self.following = []
        self.received_posts = []
        self.n_seen_posts = []
        self.n_total_seen_posts = 0
        self.visible_posts = []  # currently: all posts
        self.n_strikes = 0
        self.blocked_until = (
            0  # Block excluding this number -> Can post on this tick again.
        )
        self.preferred_n_posts = 0
        self.n_downranked = 0

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    #   Step function: in two Stages.
    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    def sample_number_of_posts(self):
        """
        Sample number of posts that an agent should share at one instant. It samples with a normal distribution based
        on this agent's vocality parameters (mean and std_dev).
        :return:
            nr_of_posts: int
        """

        mean = self.vocality["mean"]
        max_mean_increase = self.model.adjustment_based_on_belief
        extremeness = calculate_extremeness(self.beliefs)

        mean += extremeness * max_mean_increase
        std_dev = self.vocality["std_dev"]

        nr_of_posts = max(0, self.model.random.normalvariate(mu=mean, sigma=std_dev))

        # rounding and converting to int
        nr_of_posts = round(nr_of_posts)

        return nr_of_posts

    def share_post_stage(self):
        """
        First part of the agent's step function. The first stage what all agents do in a time tick.
        """
        nr_of_posts = self.sample_number_of_posts()
        self.preferred_n_posts += nr_of_posts

        # Normal posting stage only if the agent is currently not blocked
        if self.blocked_until <= self.model.schedule.time:

            posts = []

            # Create posts
            for i in range(nr_of_posts):
                post = self.create_post(rank_t=self.model.rank_t)

                # Deleting: Posts that have a very low probability of being true might be deleted
                if (post.p_true <= self.model.del_t / 100) and post.detected_as_misinfo:
                    # Delete post by not appending it and advancing the delete-counter
                    self.model.n_posts_deleted += 1
                else:
                    posts.append(post)

                # Strike system: Posts that have a very low probability of being true might yield a strike
                if (post.p_true <= self.model.strikes_t) and post.detected_as_misinfo:
                    # Strike
                    self.n_strikes += 1

                    # Apply strike consequences
                    if self.n_strikes == 1:
                        pass
                    elif self.n_strikes == 2 or self.n_strikes == 3:
                        break
                    elif self.n_strikes == 4:
                        self.blocked_until = self.model.schedule.time + 7
                        break
                    else:
                        self.blocked_until = math.inf
                        break

            # Share successful posts to followers
            for follower in self.followers:
                follower.received_posts += posts

            # Save own posts
            self.visible_posts += posts

    def update_beliefs_stage(self):
        """
        Second part of the agent's step function. The second stage what all agents do in an instant.
        """
        pass

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    #  SIT Belief-update
    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    def calculate_belief_update(self, post) -> dict:
        """
        Calculates the agent's updates on the post.
        :param post:    Post
        :return:        dict, {topic: update}
        """

        # Prepare updates dict (to update after each seen post)
        updates = {}
        for topic in Topic:
            updates[topic] = 0

        # Calculate updates
        for topic, post_value in post.tweet_beliefs.items():
            # Save previous tweet_beliefs
            prev_belief = self.beliefs[topic]

            # Calculate SIT components
            strength = self.calculate_strength(
                post
            )  # avg(relative_n_followers, belief_similarity)
            # belief_similarity: between own_beliefs and source's_beliefs
            immediacy = self.calculate_immediacy(post)  # tie_strength
            n_sources = self.calculate_n_sources()  # (1 / n_following) * 100, [0,100]

            # Combine components
            social_impact = (
                strength * immediacy * n_sources
            )  # [0,100] * [0,100] * [0,100] --> [0,100^3]

            # Rescale
            # downwards belief update
            if post_value < prev_belief:
                max_decrease = post_value - prev_belief
                rescaled_social_impact = social_impact / 1e6 * max_decrease
            # upwards belief update
            elif post_value > prev_belief:
                max_increase = post_value - prev_belief
                rescaled_social_impact = social_impact / 1e6 * max_increase
            else:
                rescaled_social_impact = 0

            # Calculate update elasticity
            update_elasticity = self.calculate_update_elasticity(prev_belief)

            # Calculate final update for belief on topic
            update = rescaled_social_impact * update_elasticity
            updates[topic] += update

        return updates

    def calculate_strength(self, post):
        """
        Calculates the strength component for the SIT belief update. In this case a combination of
        the relative number of followers and the belief_similarity between own belief & estimated belief of source.
        The other person's tweet_beliefs are estimated by looking at the tweet_beliefs of their last posts.
        :param post:        current post by other person (i.e., source)
        :return:            strength    float
        """
        rel_n_followers = self.get_relative_n_followers(post.source)
        belief_similarity = self.estimate_belief_similarity(post)
        strength = (rel_n_followers + belief_similarity) / 2

        return strength

    def calculate_immediacy(self, post):
        """
        Calculates immediacy component for the SIT belief update as  tie strength (i.e., edge weight).
        :return:            immediacy n_seen_posts_repl
        """

        tie_strength = self.model.G.edges[self.unique_id, post.source.unique_id, 0][
            "weight"
        ]  # Always key=0 because
        # maximally one connection in this direction possible.
        immediacy = tie_strength

        return immediacy

    def calculate_n_sources(self):
        """
        For the immediacy component of the SIT belief update, calculates the factor n_sources. The more accounts a user
        is following, the less they will update their tweet_beliefs based on each single one of them.
        :return:    float
        """
        n_following = len(self.following)
        n_sources = (1.0 / n_following) * 100

        return n_sources

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    #  Averaging Belief-update  (Toy, for comparison)
    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    def update_beliefs_avg(self, post):
        """
        Simplest update_beliefs function.
        New belief is average between own previous belief and the tweet_belief on the topic.
        """
        # Update towards post's tweet_beliefs
        for topic, value in post.tweet_beliefs.items():
            prev_belief = self.beliefs[topic]
            self.beliefs[topic] = (prev_belief + value) / 2

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    #   General Helper-Functions
    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    def init_beliefs(self):
        """
        Initialize for each topic a belief.
        """
        pass

    def create_post(self, based_on_beliefs=True, rank_t=10):
        """
        Creates a new post. Either random or based on own tweet_beliefs.
        :return: Post
        """
        # Get post_id & post's tweet_beliefs
        post_id = self.model.post_id_counter
        # Increase post_id_counter
        self.model.post_id_counter += 1

        if based_on_beliefs:
            tweet_beliefs = sample_beliefs(agent=self)
        else:
            tweet_beliefs = sample_beliefs()

        # Create post
        post = Post(post_id, source=self, tweet_beliefs=tweet_beliefs, rank_t=rank_t)

        return post

    def get_relative_n_followers(self, source):
        """
        Normalizes n_followers of agent.
        If 0.0: least n_followers in network.
        If 100.0: most n_followers in network.
        :return:    relative_n_followers    float   percentile
        """
        n_followers = len(list(self.model.G.successors(source.unique_id)))
        min_followers, max_followers = self.model.agents_data["n_followers_range"]

        relative_n_followers = (n_followers - min_followers) / (
            max_followers - min_followers
        )
        relative_n_followers = relative_n_followers * 100

        return relative_n_followers

    def estimate_belief_similarity(self, post):
        """
        For the immediacy component of the SIT belief update, estimate the belief similarity of self to other agent
        (only considering topics in this post).
        # EXTENSION: could also consider all topics mentioned in their last posts
        :param post:    Post
        :return:        float, similarity estimate
        """
        # Estimate other person's tweet_beliefs (on topics in current post)
        estimated_beliefs = {}
        n_posts = self.model.n_posts_estimate_similarity

        for topic, value in post.tweet_beliefs.items():
            # Estimate their belief on 'topic' by looking at their last n_posts
            values = []
            for p in post.source.visible_posts[-n_posts:]:
                if topic in p.tweet_beliefs:
                    value = p.tweet_beliefs[topic]
                    values.append(value)
            estimated_beliefs[topic] = sum(values) / len(values)

        # Calculate belief similarity (on tweet_beliefs in current post)
        similarities = []
        for topic, _ in post.tweet_beliefs.items():
            similarity = 100 - abs(self.beliefs[topic] - estimated_beliefs[topic])
            similarities.append(similarity)
        belief_similarity = sum(similarities) / len(similarities)

        return belief_similarity

    @staticmethod
    def calculate_update_elasticity(prev_belief, std_dev=15.0, neutral_belief=50):
        """
        Calculates the update elasticity of an agent given its previous belief.
        Needed because it takes more until someone updates away from a belief in which they have high confidence
        (i.e., belief close extremes of the belief_domain). Beliefs don't update from e.g., 99 --> 20 due to one post.
        Analogously, when someone has low confidence in their belief (i.e., close to middle of belief_domain),
        it makes sense that they update more per seen post.

        :param prev_belief:             float, previous belief of an agent  (domain: belief_domain)
        :param std_dev:                 float or int                        (domain: belief_domain)
        :return: update_elasticity:     float                               (domain: [0,1])
        :param neutral_belief:          float or int                        (domain: belief_domain)
        """
        update_strength = get_update_strength(
            prev_belief=prev_belief, mean=neutral_belief, std_dev=std_dev
        )

        # Rescale update_strength, such that at middle (e.g., 50), the update elasticity is 1:
        max_elasticity = get_update_strength(
            prev_belief=neutral_belief, mean=neutral_belief, std_dev=std_dev
        )
        update_elasticity = update_strength / max_elasticity

        return update_elasticity


class NormalUser(BaseAgent):
    """ NormalUser Agent """

    def __init__(self, unique_id, model, high_media_lit):
        """
        :param unique_id: int
        :param model: MisinfoPy
        """
        super().__init__(unique_id, model)

        self.vocality = {
            "mean": self.model.mean_normal_user,
            "std_dev": self.model.sigma,
        }
        self.media_literacy = MediaLiteracy.get_random(
            mlit_weights=[1 - high_media_lit, high_media_lit], rng=self.model.random
        )  # {LOW, HIGH}

    def init_beliefs(self):
        """
        Initialize for each topic a random belief.
        """
        for topic in Topic:
            self.beliefs[topic] = self.model.random.randint(0, 100)

    def update_beliefs_stage(self):
        """
        Second part of the agent's step function. The second stage what all agents do in an instant.
        """
        # Agent can only update tweet_beliefs if it received posts in the first stage of the time tick
        if len(self.received_posts) > 0:
            # Sample which of the received posts are actually seen (depends on ranking).
            rng = self.model.random
            seen_posts = [
                post for post in self.received_posts if (rng.random() < post.visibility)
            ]
            self.n_seen_posts.append(len(seen_posts))
            self.n_total_seen_posts += len(seen_posts)

            # For each seen post: judge truthfulness, then update tweet_beliefs (if post is judged as truthful).
            for post in seen_posts:

                # For each seen post: judge whether it is truthful.
                post_judged_as_truthful = self.judge_truthfulness_realistic(post)

                # For each seen post, which is judged as truthful: update tweet_beliefs.
                if post_judged_as_truthful:
                    # Update tweet_beliefs with required belief update function
                    self.update_beliefs(post)
        else:
            self.n_seen_posts.append(0)

        # empty received_posts again
        self.received_posts = []

    def update_beliefs(self, post):
        """
        Updates the tweet_beliefs with the required belief update function.
        :param post: Post
        """
        if self.model.belief_update_fn == BeliefUpdate.SAMPLE:
            self.update_beliefs_sample(post)
        elif self.model.belief_update_fn == BeliefUpdate.DEFFUANT:
            self.update_beliefs_deffuant(post)
        elif self.model.belief_update_fn == BeliefUpdate.SIT:
            self.update_beliefs_sit(post)
        else:
            raise ValueError("Not a defined belief update function.")

    def update_beliefs_sample(self, post):
        """
        Updates the agent's belief with the SAMPLE belief update function.
        :param post:        Post
        """
        old = self.beliefs[Topic.VAX]
        tweet_belief = post.tweet_beliefs[Topic.VAX]
        p_update = self.model.sampling_p_update
        if self.model.random.random() <= p_update:
            new = (old + tweet_belief) / 2
            self.beliefs[Topic.VAX] = new

    def update_beliefs_deffuant(self, post):
        """
        Updates the agent's belief with the DEFFUANT belief update function. (Currently only wrt Topic.VAX)
        Examples: Du et al. (2021), Rajabi et al. (2020), Mason et al. (2020)
        :param post:    Post
        """
        old = self.beliefs[Topic.VAX]
        tweet_belief = post.tweet_beliefs[Topic.VAX]
        mu = self.model.deffuant_mu

        new = old + mu * (tweet_belief - old)
        self.beliefs[Topic.VAX] = new

    def update_beliefs_sit(self, post):
        """
        Updates the tweet_beliefs of the agent based on a post, using the SIT-based belief update function.
        (i.e., Social Impact Theory based, adjusted to the social media environment by Reddel (2021))

        The post which is passed is assumed to be seen by the agent. It is also assumed that the agent actually
        updates tweet_beliefs based on the post. I.e., in the current implementation, the function is to be used after
        the agent has decided to judge the post as truthful.

        :param post:    Post, a seen post
        """

        # Calculate how the agent will update its tweet_beliefs
        updates = self.calculate_belief_update(post)

        # Update own tweet_beliefs  (after each seen post)
        for topic, update in updates.items():
            self.beliefs[topic] += update

    def judge_truthfulness_simple(self, post):
        """
        Simple version of judging the truthfulness of a post.
        Agents with high media literacy judge true posts as true, and false posts as false.
        Agents with low media literacy judge all posts as true.
        :param post: Post
        :return: boolean, whether the post is judged as true or false
        """
        judged_truthfulness = True
        if (
            self.media_literacy == MediaLiteracy.HIGH
            and post.ground_truth == GroundTruth.FALSE
        ):
            judged_truthfulness = False

        return judged_truthfulness

    def judge_truthfulness_realistic(self, post):
        """
        More realistic version of judging the truthfulness of a post.
        Uses a probability for agents with MediaLiteracy.HIGH to judge a post as truthful.
        :param post: Post
        :return: boolean, whether the post is judged as true or false
        """

        # get probability of updating to the post, dependent on media literacy
        if self.media_literacy == MediaLiteracy.HIGH:
            if post.ground_truth == GroundTruth.TRUE:
                p_judged_as_truthful = 0.8
            else:
                p_judged_as_truthful = 0.2
        else:
            p_judged_as_truthful = (
                1.0  # People with Medialiteracy.LOW will always update
            )

        # Sample whether post is judged as truthful
        if self.model.random.random() < p_judged_as_truthful:
            judged_truthfulness = True
        else:
            judged_truthfulness = False

        return judged_truthfulness


class Disinformer(BaseAgent):
    """ Disinformer Agent"""

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

        self.vocality = {
            "mean": self.model.mean_disinformer,
            "std_dev": self.model.sigma,
        }

    def init_beliefs(self):
        """
        Initialize for each topic a random extreme belief. Currently, always at the lower end of [0,100].
        """
        for topic in Topic:
            self.beliefs[topic] = self.model.random.randint(0, 10)

    def update_beliefs_stage(self):
        """
        Second part of the Disinformer agent's step function. Disinformers don't update their tweet_beliefs
        """
        # To include disinformers into the profit metric of n_seen_posts:
        seen_posts = [
            post
            for post in self.received_posts
            if (self.model.random.random() < post.visibility)
        ]
        self.n_seen_posts.append(len(seen_posts))
