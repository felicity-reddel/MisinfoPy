# Project
from model.misinfo_model import MisinfoPy
from model.enums import BeliefUpdate
from dmdu.utils_dmdu import (
    get_uncertainties,
    get_levers,
    get_outcomes,
    get_constants,
    get_epsilons,
    get_reference_scenario,
    make_sure_path_exists,
)

# General
import pandas as pd

# ema_workbench
from ema_workbench import (
    Model,
    RealParameter,
    ScalarOutcome,
    Constant,
    ema_logging,
    MultiprocessingEvaluator,
    SequentialEvaluator,
    CategoricalParameter,
    Scenario,
    ema_logging
)

_logger = ema_logging.get_module_logger(__name__)


def misinfopy(
    # uncertainties (general, below model specific)
    belief_metric_threshold,
    n_edges,
    ratio_normal_user,
    mean_normal_user,
    mean_disinformer,
    high_media_lit,
    deffuant_mu,
    sampling_p_update,
    n_posts_estimate_similarity,
    # levers
    mlit_select,
    del_t,
    rank_punish,
    rank_t,
    strikes_t,
    # constants
    steps,
    belief_update_fn,
    # optimization param (dependent on the model's sensitivity to stochastics)
    n_replications=1,
):
    """
    Function for the optimization.
    The experiment specified by the parameters is run n_replications times. The resulting data is summarized & returned.

    Explanation why needed:
    - The model is sensitive to stochastics. Consequently, a specific experiment can lead to different outcomes.
        (1 experiment = combination of a scenario and policy,
        1 scenario = combination of uncertainty-values,
        1 policy = combination of lever-values)
    - To get representative information of the policy's performance, the model is run multiple times and summarized.

    ––– Uncertainties –––
    @param belief_metric_threshold:     float, threshold for the belief metric (agents above belief threshold)
    @param n_edges:                     int, with how many edges gets attached to the already built network
    @param ratio_normal_user:           float, in range [0.0, 1.0]
    @param mean_normal_user:            float, mean of normal user for sampling n_posts
    @param mean_disinformer:            float, mean of normal user for sampling n_posts
    @param high_media_lit:              float, in range [0.0, 1.0], init ratio of agents with MediaLiteracy.HIGH
    @param deffuant_mu:                 float, updating parameter, indicates how strongly the belief is updated
                                        towards the post's belief. If mu=0.1, the update is 10% towards the
                                        post's belief.
    @param sampling_p_update:           float, probability that the agent will update
    @param n_posts_estimate_similarity: int, nr of last posts used by an agent to estimate the belief of another agent

    ––– Levers –––
    @param mlit_select: int, in domain [0,100], Percentage of agents empowered by media literacy intervention.
    @param rank_punish: int, in domain [-100,0], relative visibility change for posts with GroundTruth.FALSE
    @param del_t: int, in domain [0,100], p_true_threshold for deleting (below -> may be deleted)
    @param rank_t: int, in domain [0,100], p_true_threshold for down-ranking (below -> may be down-ranked)
    @param strikes_t: int, in domain [0,100], p_true_threshold for strike system (below -> may result in strike)

    ––– Constants –––
    @param steps:                       int
    @param belief_update_fn:            BeliefUpdate (enum)
    @param n_replications:              int, number of replications summarized over (for each optimization step)

    @return: tuple (1 value for each metric)
    """
    _logger.debug("start")
    # Initialize the model
    model = MisinfoPy()

    # Run the model for n_replications (each time with the same specified scenario & policy)
    metrics_n_replic = []
    for _ in range(n_replications):
        # run model (-> get metrics data from 1 replication)
        metrics_1_replic = model(
            # uncertainties
            belief_metric_threshold=belief_metric_threshold,
            n_edges=n_edges,
            ratio_normal_user=ratio_normal_user,
            mean_normal_user=mean_normal_user,
            mean_disinformer=mean_disinformer,
            high_media_lit=high_media_lit,
            deffuant_mu=deffuant_mu,
            sampling_p_update=sampling_p_update,
            n_posts_estimate_similarity=n_posts_estimate_similarity,
            # levers
            mlit_select=mlit_select,
            del_t=del_t,
            rank_punish=rank_punish,
            rank_t=rank_t,
            strikes_t=strikes_t,
            # constants
            steps=steps,
            belief_update_fn=belief_update_fn,
        )
        # save metrics data (dict)
        metrics_n_replic.append(metrics_1_replic)

    # Combine data into DataFrame
    data = pd.DataFrame(metrics_n_replic)

    # Summarize metrics (to get 1 value for each metric, for optimization process)
    metrics = data.mean(axis=0)  # mean for each column -> Series
    metrics = tuple(metrics)

    return metrics


if __name__ == "__main__":
    ema_logging.log_to_stderr(ema_logging.INFO)

    # Params
    steps = 3
    only_one_model = True
    n_replications = 2

    models = [BeliefUpdate.DEFFUANT] if only_one_model else list(BeliefUpdate)
    for belief_update_fn in models:
        _logger.info(f"Starting with Model {belief_update_fn.name}")

        # Set up the model
        model = Model(name=f"MisinfoPy{belief_update_fn.name}", function=misinfopy)
        _logger.debug(f"model initialized")
        model.uncertainties = get_uncertainties()
        model.levers = get_levers()
        model.constants = get_constants(
            steps=steps,
            belief_update_fn=belief_update_fn,
            n_replications=n_replications,
        )
        model.outcomes = get_outcomes()
        _logger.debug(f"model completely set up")

        params = {}
        for k in model.uncertainties:
            params[k.name] = k.lower_bound

        scenario = Scenario("test", **params)

        # Set up reference scenario
        test_params = {'belief_metric_threshold': 77.5,
                       'n_edges': 2,
                       'ratio_normal_user': 0.9875,
                       'mean_normal_user': 1,
                       'mean_disinformer': 10,
                       'high_media_lit': 0.3,
                       'deffuant_mu': 0.02,
                       'sampling_p_update': 0.02,
                       'n_posts_estimate_similarity': 10}
        test_scenario = Scenario('test', **test_params)

        # Optimization
        with MultiprocessingEvaluator(
            model
        ) as evaluator:  # TODO: PickleError here  – if MultiprocessingEvaluator
            # print(f"starting optimization")
            results = evaluator.optimize(  # TODO: TypeError: misinfopy() missing 9 arguments  – if SequentialEvaluator
                searchover="levers",
                nfe=3,  # 100000,
                epsilons=get_epsilons(),
                reference=test_scenario
            )
            # print(f"results are in")

        if saving:
            dir_path = os.path.join(
                os.getcwd(), "dmdu", "optimization", "data", f"{str(nfe)}nfe"
            )
            make_sure_path_exists(dir_path)

            file_name = f"results_{belief_update_fn.name}"
            path = os.path.join(dir_path, file_name)
            save_results(results, path)