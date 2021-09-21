from avalanche.logging import StrategyLogger
from avalanche.evaluation.metric_results import MetricValue
from neptune.new.types import File
import neptune.new as neptune
from PIL.Image import Image
from matplotlib import figure
from plotly import graph_objs

class NeptuneLogger(StrategyLogger):
    def __init__(self, project_name:str, api_token:str, run_name:str = None, description:str = None):
        super().__init__()
        self.run = neptune.init(project=project_name, api_token=api_token, name=run_name, description=description)

    def log_metric(self, metric_value: MetricValue, callback: str):
        super().log_metric(metric_value, callback)
        name = metric_value.name
        value = metric_value.value

        if isinstance(value,(float, int)):
            self.run[name].log(value)
        
        elif isinstance(value, Image):
            self.run[name].log(value)

        elif isinstance(value, figure.Figure):
            self.run[name].upload(neptune.types.File.as_html(value))

        elif isinstance(value, graph_objs._figure.Figure):
            self.run[name].upload(neptune.types.File.as_html(value))
        