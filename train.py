from torch.optim import SGD, Adam
from torch.nn import CrossEntropyLoss
from avalanche.benchmarks.classic import SplitFMNIST
from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics, loss_metrics, timing_metrics, cpu_usage_metrics, StreamConfusionMatrix, disk_usage_metrics
from avalanche.models import SimpleMLP
from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.strategies import Naive, BaseStrategy
from avalanche.benchmarks.scenarios import NCScenario
from avalanche.benchmarks import nc_benchmark
from avalanche.benchmarks.utils.avalanche_dataset import AvalancheDataset

from models.ResNet import ResNet50
from trainer.strategies import ILFGIR_strategy
from data.datasets.triplet_fashion_mnist import TripletFashionMnist

if __name__ == "__main__":

    fmnist_train =  AvalancheDataset(TripletFashionMnist( train=True ))
    fmnist_test = AvalancheDataset(TripletFashionMnist( train=False ))

    #scenario = NCScenario(
    scenario = nc_benchmark(
        train_dataset=fmnist_train,
        test_dataset=fmnist_test,
        n_experiences=5,
        task_labels=False,
        shuffle=False,
        seed=0)
    #scenario = SplitFMNIST(n_experiences=5)

    # MODEL CREATION

    print("Scenario n class", scenario.n_classes)
    print("Scenario n class per exp:", scenario.n_classes_per_exp[0])

    model = ResNet50(initial_num_classes=0, num_classes=scenario.n_classes)



    # DEFINE THE EVALUATION PLUGIN and LOGGERS
    # The evaluation plugin manages the metrics computation.
    # It takes as argument a list of metrics, collectes their results and returns
    # them to the strategy it is attached to.

    # log to Tensorboard
    tb_logger = TensorboardLogger()

    # log to text file
    text_logger = TextLogger(open('log.txt', 'a'))

    # print to stdout
    interactive_logger = InteractiveLogger()

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        timing_metrics(epoch=True, epoch_running=True),
        forgetting_metrics(experience=True, stream=True),
        cpu_usage_metrics(experience=True),
        StreamConfusionMatrix(num_classes=scenario.n_classes, save_image=False),
        disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loggers=[interactive_logger, text_logger, tb_logger]
    )



    # CREATE THE STRATEGY INSTANCE (NAIVE)
    cl_strategy = ILFGIR_strategy( model, 
                                Adam(model.parameters(), lr=0.000001),
                                CrossEntropyLoss(), 
                                train_mb_size=250, 
                                train_epochs=10, 
                                eval_mb_size=50, 
                                evaluator=eval_plugin)

    # TRAINING LOOP
    print('Starting experiment...')
    results = []
    for experience in scenario.train_stream:
        print("Start of experience: ", experience.current_experience)
        print("Current Classes: ", experience.classes_in_this_experience)

        # train returns a dictionary which contains all the metric values
        res = cl_strategy.train(experience)
        print('Training completed')

        #print('Computing accuracy on the whole test set')
        # test also returns a dictionary which contains all the metric values
        #results.append(cl_strategy.eval(scenario.test_stream))