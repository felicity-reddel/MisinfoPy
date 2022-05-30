from model.misinfo_model import *


if __name__ == "__main__":
    for i in range(10):
        model = MisinfoPy(n_agents=1000)
        model(time_tracking=True, belief_update_fn=BeliefUpdate.SIT)
#              # all high:
#              # mlit_select=1.0, del_t=0.5, rank_t=0.5, rank_punish=-1.0, strikes_t=0.5)

#              # all medium:
#              # mlit_select=0.2, del_t=0.1, rank_t=0.2, rank_punish=-0.2, strikes_t=0.1)
