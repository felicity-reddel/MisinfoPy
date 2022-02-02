import matplotlib.colors as colors
import matplotlib.cm as cmx
from matplotlib import pyplot as plt
from numpy import interp
from mesa.visualization.modules import ChartModule, NetworkModule
from mesa.visualization.ModularVisualization import ModularServer

from agents import *


def get_node_color(agent):
    """
    Returns the color value of an agent. This varies based on the agent's belief on Topic.VAX.
    :param agent:   Agent
    :return: c_val, tuple: (r,g,b)  all three are floats
    """
    belief = agent.beliefs[str(Topic.VAX)]
    # Map belief value to color value
    # with PiYG, a diverging colormap:
    #       100 --> green
    #       50 --> white
    #       0 --> red
    c_norm = colors.Normalize(vmin=0, vmax=100)  # because belief can be any value in [0,100]
    scalar_map = cmx.ScalarMappable(norm=c_norm, cmap=plt.get_cmap('PiYG'))

    c_val = scalar_map.to_rgba(belief)
    new_c_val = []
    for idx, val in enumerate(c_val):
        if idx != 3:  # only adjust RGB, not transparency
            new_c_val.append(val * 256)
    c_val = tuple(new_c_val)
    return c_val


def get_edge_width(weight=1, weight_borders=(0, 100)):
    """
    Returns how wide the edge should be displayed.
    :param weight:          float, edge weight
    :param weight_borders:  tuple, (min, max)
    :return:                float, width/thickness of the depicted edge
    """
    min_weight, max_weight = weight_borders
    weight_range = [min_weight, max_weight]
    width_range = [0.1, 2]

    width = interp(weight, weight_range, width_range)
    print(f"width: {width}")  # works

    return width


def show_visualization(model,
                       n_agents=100,
                       n_edges=3,
                       agent_ratio=None,
                       media_literacy_intervention=(0.0, SelectAgentsBy.RANDOM),
                       ranking_intervention=False):
    """
    Internal function to show the visualization.
    :param model:       MisinfoPy
    :param n_agents:    int
    :param n_edges:     int
    :param agent_ratio: dict {user_type: percentage}
    :param media_literacy_intervention:  tuple: (percentage_reached, how_to_select_agents)  (float, Enum)
    :param ranking_intervention:         boolean
    """

    if agent_ratio is None:
        agent_ratio = {NormalUser.__name__: 0.9, Disinformer.__name__: 0.1}

    def network_portrayal(G):
        """
        Internal function to portray a network.
        :param G:   Graph, MultiDiGraph
        :return:    dict,   {'nodes': [portrayal_details],
                             'edges': [portrayal_details]}
        """
        # The model ensures there is always 1 agent per node
        portrayal = dict()
        portrayal['nodes'] = [{"shape": "circle",
                               "color": f'rgb{get_node_color(agent)}',
                               "size": 5,
                               "tooltip": f"{round(agent.unique_id)}"
                               }
                              for (id, agent) in G.nodes.data("agent")]

        portrayal['edges'] = [{'source': source,
                               'target': target,
                               'color': 'black',
                               'width': 1,
                               # to adjust line-width based on edge-weight, use instead:
                               # 'width': get_edge_width(G.edges[source, target, key]['weight']),
                               'directed': True
                               }
                              for (source, target, key) in G.edges]

        return portrayal

    network = NetworkModule(network_portrayal, 500, 500, library='d3')
    chart_avg_belief = ChartModule([{"Label": "Avg Vax-Belief", "Color": "blue"},
                                    {"Label": "Avg Vax-Belief above threshold", "Color": "green"},
                                    {"Label": "Avg Vax-Belief below threshold", "Color": "red"}],
                                   data_collector_name="data_collector")

    chart_indiv_belief = ChartModule([{"Label": "Agent 0", "Color": "#FFCA03"},    # yellow
                                      {"Label": "Agent 1", "Color": "#FF9300"},    # orange
                                      {"Label": "Agent 2", "Color": "#F90716"},    # red
                                      {"Label": "Agent 3", "Color": "#FF00E4"},    # pink
                                      {"Label": "Agent 4", "Color": "#9C19E0"},    # purple
                                      {"Label": "Agent 5", "Color": "#3E00FF"},    # blue
                                      {"Label": "Agent 6", "Color": "#3EDBF0"},    # light blue
                                      {"Label": "Agent 7", "Color": "#54E346"},    # light green
                                      {"Label": "Agent 8", "Color": "#27AA80"},    # green
                                      {"Label": "Agent 9", "Color": "#D06224"},    # brown
                                      {"Label": "Agent 10", "Color": "#000000"}],  # black
                                     data_collector_name="data_collector2")

    server = ModularServer(model,  # class name
                           [network, chart_avg_belief, chart_indiv_belief],
                           'Misinfo Model',  # title
                           {'n_agents': n_agents,
                            'n_edges': n_edges,
                            'agent_ratio': agent_ratio,
                            'media_literacy_intervention': media_literacy_intervention,
                            'ranking_intervention': ranking_intervention})  # model parameters

    server.port = 8521  # The default
    server.launch()
