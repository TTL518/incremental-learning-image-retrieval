import os

from torchvision.datasets import mnist
from torchvision.datasets.utils import download_url
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import InterpolationMode
from torchvision import transforms

from avalanche.benchmarks.classic import SplitFMNIST
from avalanche.benchmarks.datasets import FashionMNIST, CUB200, MNIST
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, timing_metrics
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.benchmarks import nc_benchmark
from avalanche.benchmarks.utils.avalanche_dataset import AvalancheDataset
from avalanche.training.strategies import LwF

from models.ResNet import ResNet50
from models.LeNet import LeNet_PP
from models.GoogleLeNet import GoogLeNet
from trainer.strategies import ILFGIR_strategy, Naive_strategy
from data.datasets.reduced_fashion_mnist import ReducedFashionMNIST
#from data.datasets.triplet_CUB_Birds_200 import TripletCUB200
from logger.NeptuneLogger import NeptuneLogger
from metric.plugin_lr_metric import LRPlugin
from metric.plugin_recall_at_k import RecallAtKPluginSingleTask, RecallAtKPluginAllTasks
from metric.plugin_plot_features import PlotFeaturesPlugin
from metric.plugin_triplet_loss_metric import MinibatchTripletLoss, EpochTripletLoss
from metric.plugin_ce_loss_metric import MinibatchCELoss, EpochCELoss
from metric.plugin_mmd_loss_metric import MinibatchMMDLoss, EpochMMDLoss
from metric.plugin_kd_loss_metric import MinibatchKDLoss, EpochKDLoss


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    print("Device: ", device)

    #Hyperparameters
    number_of_exp = 2
    task_label = False
    shuffle_classes_exp = False

    lr = 0.00001
    optim = "Adam"
    train_batch = 50
    eval_batch = 10000
    train_epochs = 1

    #fmnist_train = FashionMNIST(root="data", train=True, transform=ToTensor())
    #fmnist_test = FashionMNIST(root="data", train=False, transform=ToTensor())
    
    #fmnist_train = ReducedFashionMNIST(root="data", train=True, transform=ToTensor(), classes_to_use=[0,1])
    #fmnist_test = ReducedFashionMNIST(root="data", train=False, transform=ToTensor(), classes_to_use=[0,1])
    
    '''
    class_names = {
        0:"T-shirt",
        1:"Trousers",
        2:"Pullover",
        3:"Dress",
        4:"Coat",
        5:"Sandal",
        6:"Shirt",
        7:"Sneaker",
        8:"Bag",
        9:"Ankle boot"
    }
    '''
    '''
    train_dset = MNIST(root="data", train=True, transform=ToTensor())
    test_dset = MNIST(root="data", train=False, transform=ToTensor())

    class_names = {
        0:"0",
        1:"1",
        2:"2",
        3:"3",
        4:"4",
        5:"5",
        6:"6",
        7:"7",
        8:"8",
        9:"9"
    }
    '''

    class_names = None
    
    train_transform=transforms.Compose([transforms.Resize((600, 600), InterpolationMode.BILINEAR),
                                    transforms.RandomCrop((448, 448)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    test_transform=transforms.Compose([transforms.Resize((600, 600), InterpolationMode.BILINEAR),
                                    transforms.CenterCrop((448, 448)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    train_dset = CUB200(root="data", train=True, transform=train_transform)
    test_dset = CUB200(root="data", train=False, transform=test_transform)
    

    scenario = nc_benchmark(
        train_dataset=train_dset,
        test_dataset=test_dset,
        n_experiences=number_of_exp,
        task_labels=task_label,
        shuffle=shuffle_classes_exp,
        seed=0)

    print("Scenario n class", scenario.n_classes)
    print("Scenario n class per exp:", scenario.n_classes_per_exp[0])

    # MODEL CREATION

    #model = ResNet50(initial_num_classes=0, num_classes=scenario.n_classes)
    #model = LeNet_PP(initial_num_classes=2, bias_classifier=False, norm_classifier=True)
    model = GoogLeNet(initial_num_classes=0, aux_logits=False)

    #OPTIMIZER CREATION
    optim = Adam(model.parameters(), lr=lr)

    # TRAIN PROCEDURE SPECIFICATION
    
    start_from_second_exp = False

    #checkpoint = torch.load("saved_models/BaseModelForCL02_CE+Triplet.pt")
    #print(checkpoint['model_state_dict'])
    #model.load_state_dict(checkpoint['model_state_dict'])
    #optim.load_state_dict(checkpoint['optimizer_state_dict'])

    
    # DEFINE  LOGGERS

    interactive_logger = InteractiveLogger()

    '''
    If you want to use Neptune logger uncomment this block and set the project_name and the api_token of your account, and add the logger to the evaluation plugin
    '''
    project_name = ""
    api_token = ""
    run_id = ""
    run_name = "PROVA CUB"
    #run_name = "5TaskInc-CE+MMD"
    description = "PROVA CUB"
    #description = "Second Incremental training from second task (scenario with 5 task) using CE + Triple with hard miner + KD + MMD. Used A-Softmax"
    #description = "Incremental training starting from first task. 5 Task. Loss used CE+MMD (new mmd impl). Used A-Softmax "
    neptune_logger = NeptuneLogger(project_name=project_name, api_token=api_token, run_name=run_name, description=description)

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


    # DEFINE THE EVALUATION PLUGIN
    # The evaluation plugin manages the metrics computation.
    # It takes as argument a list of metrics, collectes their results and returns
    # them to the strategy it is attached to.

    eval_plugin = EvaluationPlugin(
        accuracy_metrics( epoch=True, experience=True),
        loss_metrics( minibatch=True, epoch=True, experience=True),
        timing_metrics(minibatch=True, epoch=True, epoch_running=True),
        RecallAtKPluginSingleTask(),
        RecallAtKPluginAllTasks(),
        MinibatchTripletLoss(),
        MinibatchCELoss(),
        MinibatchKDLoss(),
        MinibatchMMDLoss(),
        EpochTripletLoss(),
        EpochCELoss(),
        EpochKDLoss(),
        EpochMMDLoss(),
        #PlotFeaturesPlugin(class_names),
        #LRPlugin(),
        loggers=[interactive_logger, neptune_logger],
    )

    # CREATE THE STRATEGY INSTANCE (NAIVE)
    cl_strategy = ILFGIR_strategy(
                        model, 
                        optim,
                        CrossEntropyLoss(), 
                        train_mb_size=train_batch, 
                        train_epochs=train_epochs, 
                        eval_mb_size=eval_batch, 
                        device=device,
                        evaluator=eval_plugin
                        #eval_every=1
                        )

    # TRAINING LOOP
    print('Starting experiment...')
    results = []

    

    if(start_from_second_exp):
        #Train e evaluation incrementale dal secondo task
        for i in range(len(scenario.train_stream)-1):
            print("Start of experience: ", scenario.train_stream[i+1].current_experience)
            print("Current Classes: ", scenario.train_stream[i+1].classes_in_this_experience)

            # train returns a dictionary which contains all the metric values
            res = cl_strategy.train(scenario.train_stream[i+1])
            print('Training completed')

            #Eval su tutte le esperienze passate (SINGOLARMENTE)
            '''
            for i in range(experience.current_experience+1):
                print("Test experience: ",scenario.test_stream[i].current_experience)
                print("Current classes: ",scenario.test_stream[i].classes_in_this_experience)
                results.append(cl_strategy.eval(scenario.test_stream[i]))
            '''

            #Eval su tutte le experience passate (TUTTE INSIEME)
            results.append(cl_strategy.eval(scenario.test_stream[0:i+2]))
    else:
        #Train e evaluation incrementale dal primo task
        for experience in scenario.train_stream:
            print("Start of experience: ", experience.current_experience)
            print("Current Classes: ", experience.classes_in_this_experience)

            # train returns a dictionary which contains all the metric values
            res = cl_strategy.train(experience)
            print('Training completed')

            #Eval su tutte le esperienze passate (SINGOLARMENTE)
            '''
            for i in range(experience.current_experience+1):
                print("Test experience: ",scenario.test_stream[i].current_experience)
                print("Current classes: ",scenario.test_stream[i].classes_in_this_experience)
                results.append(cl_strategy.eval(scenario.test_stream[i]))
            '''

            #Eval su tutte le experience passate (TUTTE INSIEME)
            results.append(cl_strategy.eval(scenario.test_stream[0:experience.current_experience+1]))


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