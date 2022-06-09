# Project
from model.enums import BeliefUpdate
from dmdu.utils_dmdu import make_sure_path_exists

# General
import os
import pandas as pd
from dmdu.utils_dmdu import get_100_seeds
import random


def load_outcomes(belief_update_fn):
    """
    Loads outcome data from a specified belief update function.
    @param belief_update_fn: BeliefUpdate
    @return: pd.DataFrame
    """
    path = os.path.join(os.getcwd(), 'data', 'seedanalysis', belief_update_fn.name, 'outcomes.csv')
    outcomes = pd.read_csv(path)
    outcomes = outcomes.rename(columns={'Unnamed: 0': 'uncert_sample'})

    return outcomes


def get_quantile_data(data, n_seeds_list, n_samplings, quantile, seed=None):
    """  # TODO: UPDATE
    Creates a DataFrame of the means of each of the bootstrap-samples (from the provided outcomes DataFrame).
    @param data: pd.DataFrame
    @param n_seeds_list: list of integers, n_seeds included in a seed sample
    @param n_samplings: int, how many times each number of seeds should be sampled for the bootstrapping
    @param quantile: float, in range [0.0, 1.0], if 0.5 -> calculate median
    @param seed: int
    @return: dict, key = outcome names, value = quantile value over the provided DataFrame
    """
    if seed:
        random.seed(seed)
    seeds_100 = get_100_seeds()
    data_list = []

    for n_seeds in n_seeds_list:
        n_seeds_data = []

        for _ in range(n_samplings):
            # Sample the required amount of seeds, calculate the mean over them
            seed_sample = random.sample(population=seeds_100, k=n_seeds)

            # Only need rows of those seeds
            subset = data[data['seed'].isin(seed_sample)]
            # Only need to calculate the quantile data over the columns of objectives
            subset = subset.drop(columns=['uncert_sample', 'seed'])

            # Calculate the quantile data and save it
            data_1_sample = subset.quantile(axis=0, q=quantile)
            n_seeds_data.append(data_1_sample)

        df = pd.DataFrame(n_seeds_data)
        # add column to keep the information of over how many seeds the metrics were averaged
        df['n_seeds'] = n_seeds

        data_list.append(df)

    quantile_data = pd.concat(data_list)

    return quantile_data


def save_to_csv(data, belief_update_fn):
    """
    Saves the data to a csv file.

    @param data: pd.DataFrame
    @param belief_update_fn:
    @return:
    """

    dir_path = os.path.join(os.getcwd(), 'data', 'seedanalysis', belief_update_fn.name)
    make_sure_path_exists(dir_path)

    path = os.path.join(dir_path, f"quantile.csv")
    data.to_csv(path)
