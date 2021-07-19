from avalanche.logging import StrategyLogger
from avalanche.evaluation.metric_results import MetricValue
import neptune.new as neptune

class NeptuneLogger(StrategyLogger):
    def __init__(self, project_name:str, api_token:str, run_id:str = None, run_name:str = None, description:str = None):
        super().__init__()
        self.run = neptune.init(project=project_name, api_token=api_token, run=run_id, name=run_name, description=description)

    def log_metric(self, metric_value: MetricValue, callback: str):
        super().log_metric(metric_value, callback)
        name = metric_value.name
        value = metric_value.value

        if isinstance(value,(float, int)):
            self.run[name].log(value)