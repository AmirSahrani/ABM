import mesa

from agents import Nomad, Spice
from model import DuneModel

color_dic = {2: "#ff0C00",  1: "#00AA00", 0: "#00AAff"}
spice_color = {
    0: "#FFFFE0", # light yellow
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
    20: "#FF0D00", # red
}



def Nomad_portrayal(agent):
    if agent is None:
        return

    if type(agent) is Nomad:
        color = color_dic[agent.tribe.id]
        return {
            "Color": color,
            "Shape": "rect",
            "Filled": "true",
            "Layer": 0,
            "w": 1,
            "h": 1,
        }

    elif type(agent) is Spice:
        color = spice_color[agent.spice]
        return {
            "Color": color,
            "Shape": "rect",
            "Filled": "true",
            "Layer": 0,
            "w": 1,
            "h": 1,
        }

    return {}


canvas_element = mesa.visualization.CanvasGrid(Nomad_portrayal, 100, 100, 500, 500)
chart_element = mesa.visualization.ChartModule(
    [{"Label": "Nomad", "Color": "#AA0000"}]
)

model_params = {
    "width": 100,
    "height": 100,
    "n_tribes": 3,
    "n_agents": 1,
    "n_heaps": 8,
    "vision_radius": 5,
}

server = mesa.visualization.ModularServer(
    model_cls=DuneModel,
    model_params=model_params,
    visualization_elements=[canvas_element, chart_element],
    name="Dune"
)
server.launch()
