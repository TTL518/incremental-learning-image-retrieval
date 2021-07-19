import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from avalanche.benchmarks.classic import SplitFMNIST
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, timing_metrics
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.benchmarks import nc_benchmark
from avalanche.benchmarks.utils.avalanche_dataset import AvalancheDataset

from models.ResNet import ResNet50
from models.LeNet import LeNet_PP
from trainer.strategies import ILFGIR_strategy
from data.datasets.triplet_fashion_mnist import TripletFashionMnist
from data.datasets.triplet_CUB_Birds_200 import TripletCUB200
from logger.NeptuneLogger import NeptuneLogger
from metric.plugin_recall_at_k import RecallAtKPlugin

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    print("Device: ", device)

    #Hyperparameters
    number_of_exp = 2
    task_label = False
    shuffle_classes_exp = False

    lr = 0.000001
    optim = "Adam"
    train_batch = 50
    eval_batch = 50
    train_epochs = 25


    fmnist_train =  AvalancheDataset(TripletFashionMnist( train=True ))
    fmnist_test = AvalancheDataset(TripletFashionMnist( train=False ))

    '''
    train_transform=transforms.Compose([transforms.Resize((600, 600), InterpolationMode.BILINEAR),
                                    transforms.RandomCrop((448, 448)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    test_transform=transforms.Compose([transforms.Resize((600, 600), InterpolationMode.BILINEAR),
                                    transforms.CenterCrop((448, 448)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    cub200_train = AvalancheDataset(TripletCUB200(train=True, transform=train_transform))
    cub200_test = AvalancheDataset(TripletCUB200(train=False, transform=test_transform))
    '''

    scenario = nc_benchmark(
        train_dataset=fmnist_train,
        test_dataset=fmnist_test,
        n_experiences=number_of_exp,
        task_labels=task_label,
        shuffle=shuffle_classes_exp,
        seed=0)

    print("Scenario n class", scenario.n_classes)
    print("Scenario n class per exp:", scenario.n_classes_per_exp[0])


    # MODEL CREATION

    #model = ResNet50(initial_num_classes=0, num_classes=scenario.n_classes)
    model = LeNet_PP(initial_num_classes=0)

    # DEFINE THE EVALUATION PLUGIN and LOGGERS
    # The evaluation plugin manages the metrics computation.
    # It takes as argument a list of metrics, collectes their results and returns
    # them to the strategy it is attached to.

    # print to stdout
    interactive_logger = InteractiveLogger()

    '''
    If you want to use Neptune logger uncomment this block and set the project_name and the api_token of your account, and add the logger to the evaluation plugin

    # log to neptune
    project_name = ""
    api_token = ""
    neptune_logger = NeptuneLogger(project_name, api_token)

    #Log hyperparameters
    neptune_logger.run["Parameters"] = {
        "Continual learning scenario":{
            "Number of experience/task": number_of_exp,
            "Use of task labels": task_label,
            "Shuffle class (or experience) order": shuffle_classes_exp
        },
        "Optimizer":{
            "Opt":optim,
            "Learning rate":lr
        },
        "Training batch size":train_batch,
        "Evaluation batch size":eval_batch,
        "Training epochs x experience/task":train_epochs
    }
    '''

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch=True, experience=True),
        loss_metrics(minibatch=True, epoch=True, experience=True),
        timing_metrics(epoch=True, epoch_running=True),
        RecallAtKPlugin(),
        loggers=[interactive_logger] #, neptune_logger]
    )

    # CREATE THE STRATEGY INSTANCE (NAIVE)
    cl_strategy = ILFGIR_strategy( model, 
                                Adam(model.parameters(), lr=lr),
                                CrossEntropyLoss(), 
                                train_mb_size=train_batch, 
                                train_epochs=train_epochs, 
                                eval_mb_size=eval_batch, 
                                device=device,
                                evaluator=eval_plugin)

    # TRAINING LOOP
    print('Starting experiment...')
    results = []

    #Train e evaluation incrementale 
    for experience in scenario.train_stream:
        print("Start of experience: ", experience.current_experience)
        print("Current Classes: ", experience.classes_in_this_experience)

        # train returns a dictionary which contains all the metric values
        res = cl_strategy.train(experience)
        print('Training completed')

        for i in range(experience.current_experience+1):
            print("Test experience: ",scenario.test_stream[i].current_experience)
            print("Current classes: ",scenario.test_stream[i].classes_in_this_experience)

            results.append(cl_strategy.eval(scenario.test_stream[i]))


    '''
    #Train su intero training set e evaluation su singole experience

    # train returns a dictionary which contains all the metric values
    for experience in scenario.train_stream:
        print("Start of experience: ", experience.current_experience)
        print("Current Classes: ", experience.classes_in_this_experience)

        # train returns a dictionary which contains all the metric values
        res = cl_strategy.train(experience)
        print('Training completed')
    

    scenario_test = nc_benchmark(
        train_dataset=fmnist_train,
        test_dataset=fmnist_test,
        n_experiences=5,
        task_labels=task_label,
        shuffle=shuffle_classes_exp,
        seed=0)

    for experience in scenario_test.test_stream:
        print("Start of experience: ", experience.current_experience)
        print("Current Classes: ", experience.classes_in_this_experience)
        results.append(cl_strategy.eval(experience))
    '''