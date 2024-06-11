import mesa

from agents import Nomad, Spice
from model import DuneModel

color_dic = {2: "#ff0C00",  1: "#00AA00", 0: "#00F8ff"}
spice_color = {
    0: "#8B4513",  # SaddleBrown
    1: "#7E3E12",
    2: "#723711",
    3: "#653010",
    4: "#59290E",
    5: "#4D220C",
    6: "#401B0A",
    7: "#341409",
    8: "#280D07",
    9: "#1C0605",
    10: "#100003",
    11: "#0C0003",
    12: "#080002",
    13: "#040001",
    14: "#020001",
    15: "#010000",
    16: "#010000",
    17: "#010000",
    18: "#010000",
    19: "#010000",
    20: "#000000"   # Black
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

server = mesa.visualization.ModularServer(
    model_cls=DuneModel,
    model_params={
        "width": 100,
        "height": 100,
        "n_tribes": 3,
        "n_agents": 50
    },
    visualization_elements=[canvas_element, chart_element],
    name="Dune"
)
server.launch()
