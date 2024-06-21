import mesa as ms
from agents import Nomad, Spice, Water
from experiment_utils import *
from model import DuneModel
from mesa.visualization.ModularVisualization import ModularServer, PageHandler
import os
import tornado.web



EXPERIMENT_NAME = "Joana_trial_1"

model_params = {
    "experiment_name": EXPERIMENT_NAME,
    "width": 100,
    "height": 100,
    "n_tribes": 4,
    "n_agents": 500,
    "n_heaps": 8,
    "vision_radius": ms.visualization.Slider("Vision radius", 10, 1, 40, 1, description="How far can they see"),
    "step_count": 100,
    "alpha": ms.visualization.Slider("Fighting cost", 0.5, 0.0, 1.0, 0.1, description="How much do they lose when fighting"),
    "trade_percentage": ms.visualization.Slider("Trade Percentage", 0.5, 0.0, 1.0, 0.1, description="How much do they trade with each other"),
    "spice_movement_bias": ms.visualization.Slider("Spice movement bias", 1.0, 0.0, 1.0, 0.1, description="How much do they value moving towards spice"),
    "tribe_movement_bias": ms.visualization.Slider("Tribe movement bias", 0.0, 0.0, 1.0, 0.1, description="How much do they value moving towards their tribe"),
    "spice_generator": gen_spice_map,
    "river_generator": no_river,
    "location_generator": split_tribes_locations,
    "spice_kwargs": {
        "total_spice": 8000,
        "cov_range": (8, 20)
    },
    "spice_threshold": 7000
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
        color = tribe_colors[agent.tribe.id]
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
tribe_clustering_labels = [f"Tribe_{i}_Clustering" for i in range(model_params["n_tribes"])]

tribe_nomads_chart = ms.visualization.ChartModule(
    [{"Label": label, "Color": color} for label, color in zip(tribe_nomad_labels, tribe_colors)],
    data_collector_name='datacollector'
)

tribe_spice_chart = ms.visualization.ChartModule(
    [{"Label": label, "Color": color} for label, color in zip(tribe_spice_labels, tribe_colors)],
    data_collector_name='datacollector'
)

tribe_clustering_chart = ms.visualization.ChartModule(
    [{"Label": label, "Color": color} for label, color in zip(tribe_clustering_labels, tribe_colors)],
    data_collector_name='datacollector'
)

total_clustering_chart = ms.visualization.ChartModule(
    [{"Label": "total_Clustering", "Color": "#00FF00"}],
    data_collector_name='datacollector'
)

description = """
By Sophie, Joana, Amir, Bálint, and Sándor
"""

# package_css_includes = []
# local_css_includes = ["custom.css"]

class CustomPageHandler(PageHandler):
    def get(self):
        elements = self.application.visualization_elements
        for i, element in enumerate(elements):
            element.index = i
        self.render(
            "modular_template.html",
            port=self.application.port,
            model_name=self.application.model_name,
            description=self.application.description,
            package_js_includes=self.application.package_js_includes,
            package_css_includes=self.package_css_includes,
            local_js_includes=self.application.local_js_includes,
            local_css_includes=self.application.local_css_includes,
            scripts=self.application.js_code,
        )

class CustomModularServer(ModularServer):
    def __init__(self, model_cls, visualization_elements, name="Mesa Model", model_params=None, port=None, description="No description available"):
        super().__init__(model_cls, visualization_elements, name, model_params, port)
        self.description = description
        self.handlers[0] = (r"/", CustomPageHandler)
        self.settings["template_path"] = os.path.join(os.path.dirname(__file__), "templates")
        self.settings["static_path"] = os.path.join(os.path.dirname(__file__), "static")
        self.handlers.append((r"/static/(.*)", tornado.web.StaticFileHandler, {"path": self.settings["static_path"]}))

server = CustomModularServer(
    DuneModel,
    [canvas_element, fight_chart, trade_chart, tribe_nomads_chart, tribe_spice_chart,tribe_clustering_chart, total_clustering_chart],
    "Dune Model",
    model_params,
    description=description
)

server.port = 8521
server.launch()
