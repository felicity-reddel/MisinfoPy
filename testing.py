from misinfo_model import *


if __name__ == "__main__":
    max_run_length = 5

    model = MisinfoPy(n_agents=100,
                      n_edges=3,
                      ratio_normal_user=0.99,
                      mlit_select=0.0,
                      media_literacy_intervention_durations={"initial investment": 3600,
                                                             "MediaLiteracy.LOW": 3,
                                                             "MediaLiteracy.HIGH": 30},
                      rank_punish=-0.0,  # negative n_seen_posts_repl: no ranking adjustment
                      del_t=-0.1,  # negative n_seen_posts_repl: no deleting
                      rank_t=-0.2,  # negative n_seen_posts_repl: no ranking
                      strikes_t=-0.01,  # negative n_seen_posts_repl: no strike system
                      belief_update_fn=BeliefUpdate.DEFFUANT,
                      show_n_seen_posts=False)

    # Run model and save KPIs
    [n_above_threshold,
     variance,
     kl_divergence,
     engagement,
     free_speech_constraint,
     avg_user_effort] = model(steps=max_run_length, debug=True)

    print(f"KPIs: \n"
          # f"- n_above_threshold: {n_above_threshold} \n"
          f"- variance: {variance} \n"
          # f"- kl_divergence: {kl_divergence} \n"
          # f"- engagement: {engagement} \n"
          f"- free speech constraint: {free_speech_constraint} \n"
          f"- avg user effort: {avg_user_effort}")

    # tweet_beliefs = model.get_beliefs()
    # total_seen_posts = model.get_total_seen_posts()
    # seen_posts_per_agent = [sum(agent.n_seen_posts) for agent in model.schedule.agents]

    # print(total_seen_posts)
    # print(len(seen_posts_per_agent))

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    # dummy error from above --> for now just continue testing with belief_distributions_before_after.csv
    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    # # Get a simple belief list (from belief_distributions_before_after.csv)
    # tweet_beliefs = [79.72770665896219, 82.69618715268301, 85.24117953778078, 97.0714936146096, 83.33612257312177,
    #            84.64837682680201, 45.0531456484707, 85.76322615127073, 83.69789871981297, 79.44403788580144,
    #            4.402316588208926, 4.126986794549279, 24.29952694332167, 64.85775653398102, 1.2037789279697713,
    #            91.40432858917828, 65.8810691812474, 86.40571316122025, 81.35184159382301, 84.61168720518631,
    #            97.75019082747808, 75.25532803259503, 79.44948286235342, 84.74089164632166, 84.88648152189586,
    #            34.98547641350575, 85.67354199833909, 74.91327689930019, 69.04355273662476, 89.38941039502349,
    #            91.07614815333098, 72.8793491319055, 76.02701320645733, 86.37737104026742, 90.59528384086498,
    #            55.90381331127618, 25.7911258374345, 79.6706539088877, 89.08168193602292, 18.516165903647963,
    #            84.76899752206131, 18.938664397446068, 75.29394286589485, 0, 52.354591977202325, 16.163469210065383,
    #            76.91829645493313, 93.16370801626898, 88.38199044301179, 90.29719971052653, 77.50912225743076,
    #            89.66438369016109, 49.130609751263194, 76.80247426924448, 78.37415415171084, 3.414971597208815,
    #            82.39138766498469, 86.5895922004333, 85.05304336474059, 78.50568942905659, 83.68767632610877,
    #            78.57506506705913, 88.92049157057428, 79.30964982279752, 84.73087532396423, 78.92926906579626,
    #            87.2205983824625, 86.21596932713766, 78.69123642863529, 79.18076273566771, 79.11549028003235,
    #            77.3564991682535, 13.078460323930727, 82.60487821254652, 80.67763669145917, 81.55396136549575,
    #            74.2445745428569, 31.815846664134018, 84.00362500491761, 80.60178167775133, 89.09585215062955,
    #            82.75885902917796, 10.119657450352031, 95.6600856193124, 89.07639395474429, 85.19501169459352,
    #            17.40846975600161, 88.79327710290885, 84.39695704564875, 60.09846920794175, 78.88450982812758,
    #            76.60626297773308, 86.87018097849364, 2, 79.7862688320401, 70.3192482329987, 92.77222698992175,
    #            13.140601766793882, 82.60630257023193, 77.41254851241526]
    # # --> 100 agents' tweet_beliefs

    # # Test discretize(): Transform a list of (belief values) into a discretized distribution
    # discretized_distribution = discretize(tweet_beliefs)
    # print(f'discretized_distribution: \n{discretized_distribution}')
    # print(f'sum(): {sum(discretized_distribution)}\n')
    #
    #
    # # Test create_polarized_pdf()
    # polarized_pdf = create_polarized_pdf()
    # print(f'create_polarized_pdf: \n{polarized_pdf}')
    # print(f'len(): {len(polarized_pdf)}')
    # print(f'sum(): {sum(polarized_pdf)}\n')
    #
    #
    # # Test variance
    # # variance = MisinfoPy.variance(distribution=tweet_beliefs)
    # Not worth it if self gives an error. Only maybe for rounding.
    # variance = round(statistics.variance(tweet_beliefs))
    # print(f'variance: {variance}')
    #
    #
    # # Test kl_divergence
    # kl_div = kl_divergence(belief_list=tweet_beliefs)
    # print(kl_div)
