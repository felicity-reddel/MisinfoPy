from model.misinfo_model import *
from visualization import *
import time


if __name__ == '__main__':

    # Parameters
    visualize = False
    n_agents = 100  # 1000
    ratio = 0.99
    n_edges = 2  # 3
    mlit_select = 0.0
    rank_punish = -0.0
    del_t = 0.1
    rank_t = -0.1
    strikes_t = -0.1
    belief_update_fn = BeliefUpdate.DEFFUANT
    sampling_p_update = 0.02
    deffuant_mu = 0.02

    if visualize:

        # Only needs the line below. Runs model in the browser.
        show_visualization(model=MisinfoPy,
                           n_agents=n_agents,
                           n_edges=n_edges,
                           ratio_normal_user=ratio,
                           mlit_select=mlit_select,
                           rank_punish=rank_punish,
                           del_t=del_t,
                           rank_t=rank_t,
                           strikes_t=strikes_t,
                           belief_update_fn=belief_update_fn,
                           sampling_p_update=sampling_p_update,
                           deffuant_mu=deffuant_mu
                           )

    else:

        max_run_length = 10

        model = MisinfoPy(n_agents=n_agents,
                          n_edges=n_edges,
                          ratio_normal_user=ratio,
                          mlit_select=mlit_select,
                          rank_punish=rank_punish,
                          del_t=del_t,
                          rank_t=rank_t,
                          strikes_t=strikes_t,
                          belief_update_fn=belief_update_fn,
                          sampling_p_update=sampling_p_update,
                          deffuant_mu=deffuant_mu
                          )

        print(f"Starting")
        start_time = time.time()
        for i in range(max_run_length):
            model.step()
            if i % 10 == 0:
                print(f"step {i} done")

        run_time = round(time.time() - start_time, 2)
        print(f"With {max_run_length} steps, runtime is {run_time}")
