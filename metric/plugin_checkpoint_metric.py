
from avalanche.evaluation.metrics import Accuracy
from avalanche.evaluation import GenericPluginMetric
from avalanche.evaluation.metric_results import MetricValue
from avalanche.evaluation.metric_utils import phase_and_task
import torch



class CheckpointPluginMetric(GenericPluginMetric[float]):
    """
    Plugin metric to save best and backup checkpoint
    """

    def __init__(self, bestModelPath, reset_at="stream", emit_at="stream", mode="eval" ):
        self._metric = Accuracy()
        self.bestMetricValue = 0.0
        self.bestModelPath = bestModelPath
        self.mode = mode
        self.finalEval = False

        super(CheckpointPluginMetric, self).__init__(
            self._metric, reset_at=reset_at, emit_at=emit_at,
            mode=mode)
        

    def reset(self, strategy=None) -> None:
        self._metric.reset()

    def result(self, strategy=None) -> float:
        return self._metric.result()

    def update(self, strategy):
        task_labels = strategy.experience.task_labels[0]
        self._metric.update(strategy.mb_output, strategy.mb_y, task_labels)

    def before_training_exp(self, strategy: 'PluggableStrategy'):
        print("Azzero best metric value per salvare modello migliore durante training")
        self.bestMetricValue = 0.0

    def before_eval(self, strategy: 'PluggableStrategy') -> None:
        """
        Reset the accuracy before the evaluation phase
        """
        self.reset()
        if(strategy.epoch == strategy.train_epochs):
            self.finalEval=True
            print("Evaluation finale")
        else:
            print("Evaluation durante training")
            self.finalEval=False
    
    def after_eval_iteration(self, strategy: 'PluggableStrategy') -> None:
        """
        Update the accuracy after eval iteration
        """
        self.update(strategy)
    
    def after_eval(self, strategy: 'PluggableStrategy') -> None:
        """
        Check if is best model and save it
        """
        if(self.finalEval==False): #Controllo che non sia l'evaluation FINALE (a fine training)
            value = self.result()[0]
            print("Accuracy: ", value)
            print("Best Accuracy: ", self.bestMetricValue)
            if value > self.bestMetricValue:
                print("Salvo nuovo miglior modello")
                best_dict= {
                    'training_exp':strategy.experience.current_experience, 
                    'training_epoch': strategy.epoch,
                    'model_state_dict': strategy.model.state_dict(),
                    'optimizer_state_dict': strategy.optimizer.state_dict()
                }
                self.bestMetricValue = value
                torch.save(best_dict, self.bestModelPath)



        