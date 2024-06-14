import mesa as ms
from agents import Nomad, Spice, Water
from model import DuneModel
from mesa.visualization.ModularVisualization import ModularServer



EXPERIMENT_NAME = "Balint_trial2"

model_params = {
    "experiment_name": EXPERIMENT_NAME,
    "width": 100,
    "height": 100,
    "n_tribes": 3,
    "n_agents": 100,
    "n_heaps": 8,
    "vision_radius": 5,
    "step_count": 100,
    "alpha": ms.visualization.Slider("Fighting cost", 0.5, 0.0, 1.0, 0.1),
    "trade_percentage": ms.visualization.Slider("Trade Percentage", 0.5, 0.0, 1.0, 0.1),
}


color_dic = {
    2: "#ff0C00",
    1: "#00AA00",
    0: "#00AAff"
}

spice_color = {
    0: "#FFFFE0",
    1: "#FFF8C1",
    2: "#FFF1A3",
    3: "#FFEB84",
    4: "#FFE465",
    5: "#FFDE47",
    6: "#FFD728",
    7: "#FFD109",
    8: "#FFCA00",
    9: "#FFB800",
    10: "#FFA500",
    11: "#FF9500",
    12: "#FF8500",
    13: "#FF7500",
    14: "#FF6500",
    15: "#FF5400",
    16: "#FF4500",
    17: "#FF3700",
    18: "#FF2900",
    19: "#FF1B00",
    20: "#FF0D00"
}

def Nomad_portrayal(agent):
    if agent is None:
        return

    if isinstance(agent, Nomad):
        color = color_dic[agent.tribe.id]
        return {
            "Color": color,
            "Shape": "circle",
            "Filled": "false",
            "r": 0.7,
            "Nomad": f"Tribe {agent.tribe.id}",
            "Layer": 0,
        }

    elif isinstance(agent, Spice):
        color = spice_color[agent.spice]
        return {
            "Color": color,
            "Shape": "rect",
            "Filled": "true",
            "w": 1,
            "h": 1,
            "Spice level": f"{agent.spice}",
            "Layer": 0,
        }
    elif isinstance(agent, Water):
        return {
            "Color": "#0000Af",
            "Shape": "rect",
            "Filled": "true",
            "w": 1,
            "h": 1,
            "Water":"",
            "Layer": 0,
        }

    return {}

canvas_element = ms.visualization.CanvasGrid(Nomad_portrayal, 100, 100, 1000, 1000)

chart_element = ms.visualization.ChartModule(
    [{"Label": "Nomad", "Color": "#AA0000"}]
)

fight_chart = ms.visualization.ChartModule(
    [{"Label": "Fights_per_step", "Color": "#00FF00"},
     {"Label": "Cooperation_per_step", "Color": "#FF0000"}],
    data_collector_name='datacollector'
)

trade_colors = ["#FFA500", "#FF4500", "#FF0000", "#00FF00", "#00FFFF", "#FF00FF", "#FFFF00", "#000000", "#FFFFFF"]
trade_labels = [f"Tribe_{i}_Trades" for i in range(model_params["n_tribes"])]
trade_chart = ms.visualization.ChartModule(
    [{"Label": label, "Color": color} for label, color in zip(trade_labels, trade_colors)],
    data_collector_name='datacollector'
)


tribe_colors = ["#0000FF", "#00FF00", "#FFA500", "#FF4500", "#FF0000", "#00FFFF", "#FF00FF", "#FFFF00", "#000000", "#FFFFFF"]
tribe_nomad_labels = [f"Tribe_{i}_Nomads" for i in range(model_params["n_tribes"])]
tribe_spice_labels = [f"Tribe_{i}_Spice" for i in range(model_params["n_tribes"])]

tribe_nomads_chart = ms.visualization.ChartModule(
    [{"Label": label, "Color": color} for label, color in zip(tribe_nomad_labels, tribe_colors)],
    data_collector_name='datacollector'
)

tribe_spice_chart = ms.visualization.ChartModule(
    [{"Label": label, "Color": color} for label, color in zip(tribe_spice_labels, tribe_colors)],
    data_collector_name='datacollector'
)

tribe_clustering_chart = ms.visualization.ChartModule(
    [{"Label": "Tribe_0_Clustering", "Color": "#FFA500"},
     {"Label": "Tribe_1_Clustering", "Color": "#FF4500"}],
    data_collector_name='datacollector'
)

server = ms.visualization.ModularServer(
    DuneModel,
    [canvas_element, fight_chart, trade_chart, tribe_nomads_chart, tribe_spice_chart, tribe_clustering_chart],
    "Dune Model",
    model_params
)

server.launch()