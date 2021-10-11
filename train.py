import os

from torch.optim import lr_scheduler
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import torch
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn import CrossEntropyLoss

import numpy as np

from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, timing_metrics
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.benchmarks import nc_benchmark

from models.ResNetN import INCREMENTAL_ResNet
from models.LeNet import LeNet_PP, LeNet
from models.GoogleLeNet import GoogLeNet
from models.ResNet_CIFAR import resnet32
from trainer.strategies import ILFGIR_strategy, Naive_strategy
from utils import set_seed

from data.datasets.utils import get_train_test_dset

from logger.NeptuneLogger import NeptuneLogger
from metric.plugin_lr_metric import LRPlugin
from metric.plugin_recall_at_k import RecallAtKPluginSingleTask, RecallAtKPluginAllTasks
from metric.plugin_checkpoint_metric import CheckpointPluginMetric
from metric.plugin_plot_features import PlotFeaturesPlugin
from metric.plugin_triplet_loss_metric import MinibatchTripletLoss, EpochTripletLoss
from metric.plugin_ce_loss_metric import MinibatchCELoss, EpochCELoss
from metric.plugin_mmd_loss_metric import MinibatchMMDLoss, EpochMMDLoss
from metric.plugin_kd_loss_metric import MinibatchKDLoss, EpochKDLoss
from metric.plugin_global_loss_metric import MinibatchGlobalLoss, EpochGlobalLoss
from metric.plugin_cl_accuracy import MyEvalExpAccuracy


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    print("Device: ", device)

    
    seed = 0
    #seed = np.random.randint(0, 10000)
    if(seed!=-1):
        set_seed(seed)

    #Hyperparameters
    number_of_exp = 2
    #per_exp_classes = {
    #    0:100
    #}
    per_exp_classes = None
    initial_num_classes = 50
    task_label = False
    shuffle_classes_exp = False

    bias_classifier = False
    norm_weights_classsifier = True

    #bestModelPath = "saved_models/bestModel_2T_RN32_CIFAR100_CE+Triplet+MMD+KD-140epochs-WeightedLOSS.pth"
    #bestModelPath = "saved_models/bestModel_2T_RN50_CUB_CE+Triplet+MMD_512.pth"
    #bestModelPath = "saved_models/bestModel_2T_Inception_CUB_CE+Triplet+MMD.pth"
    bestModelPath = "PROVA.pth"

    optim = "SGD"
    lr = 0.1
    momentum = 0.9
    weight_decay = 2e-4

    #optim = "Adam"
    #lr = 0.00001
    
    train_batch = 32
    eval_batch = 32
    train_epochs =4
    eval_every = 2

    dataset = "CIFAR100"
    train_dset, test_dset, class_names = get_train_test_dset(dataset)

    scenario = nc_benchmark(
        train_dataset=train_dset,
        test_dataset=test_dset,
        n_experiences=number_of_exp,
        per_exp_classes = per_exp_classes,
        task_labels=task_label,
        shuffle=-shuffle_classes_exp,
        seed=seed)

    print("Scenario n class", scenario.n_classes)
    print("Scenario n class per exp:", scenario.n_classes_per_exp[0])

    # MODEL CREATION

    modelName = "CIFAR resnet32"
    #modelName = "Resnet50 pretrained on imagenet"
    #model = INCREMENTAL_ResNet(ResNet_N=50, pretrained=True, save_dir="net_checkpoints/", initial_num_classes=initial_num_classes)
    #model = LeNet_PP(initial_num_classes=2, bias_classifier=False, norm_classifier=True)
    #model = LeNet(initial_num_classes=2, bias_classifier=False, norm_classifier=True)
    model = resnet32(bias_classifier=bias_classifier, norm_weights_classifier=norm_weights_classsifier, num_classes=initial_num_classes)
    #model = GoogLeNet(initial_num_classes=0, aux_logits=False)
    
    print(model)
    print("Parametri totali= ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    #OPTIMIZER CREATION
    #optim = Adam(model.parameters(), lr=lr)
    optim = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay )
    scheduler_lr = MultiStepLR(optim, milestones=[49, 69, 89, 109], gamma=0.1)

    # optim = Adam([
    #             {'params': model.parameters() },
    #             {'params': model.fc.parameters(), 'lr': 1e-5},
    #         ], lr=1e-6)

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
    project_name = "tibi/TESIMAG"
    
    run_id = ""
    #run_name = "TaskInc-CE+Triplet+MMD-CUB-ResNet50_512"
    run_name = "2TaskInc-CE+Triplet+MMD+KD-CIFAR100-ResNet32-140epochs-WeightedLOSS"
    #run_name = "2Task_Inception_CUB_CE+Triplet+MMD"
    #description = "PROVA solo mmd"
    #description = "Second Incremental training from second task (scenario with 5 task) using CE + Triple with hard miner + KD + MMD. Used A-Softmax"
    description = "Incremental training 2 Task. CIFAR100. ResNet32. Loss used CE+Triplet+MMD+KD. With classifier weights normalization and Bias False. 140 epochs. WeightedLOSS"
    #description = "Incremental training 2 Task. CUB200. ResNet50 pretrained on imagenet. Loss used CE+Triplet+MMD. With classifier weights normalization and Bias False. 512 dim of normalized features for retrieval"
    #description = "Incremental training 2 Task. CUB200. Inception. Loss used CE+Triplet+MMD. PARAMETERS LIKE On the exploration"
    
    neptune_logger = NeptuneLogger(project_name=project_name, run_name=run_name, description=description)

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
        "Seed": seed,
        "Training batch size":train_batch,
        "Evaluation batch size":eval_batch,
        "Training epochs x experience/task":train_epochs,
        "Evaluation every": eval_every,
        "Classifier Bias": bias_classifier,
        "Normalization classifier weights": norm_weights_classsifier,
        "Best model path": bestModelPath,
        "Initial num classes": initial_num_classes,
        "Dataset": dataset,
        "Model": modelName

    }


    # DEFINE THE EVALUATION PLUGIN
    # The evaluation plugin manages the metrics computation.
    # It takes as argument a list of metrics, collectes their results and returns
    # them to the strategy it is attached to.

    eval_plugin = EvaluationPlugin(
        accuracy_metrics( minibatch=True, epoch=True, experience=True, stream=True),
        loss_metrics( minibatch=True, epoch=True, experience=True),
        timing_metrics(minibatch=True, epoch=True, epoch_running=True),
        RecallAtKPluginSingleTask(),
        RecallAtKPluginAllTasks(),
        CheckpointPluginMetric(bestModelPath),
        MinibatchTripletLoss(),
        MinibatchCELoss(),
        MinibatchKDLoss(),
        MinibatchMMDLoss(),
        MinibatchGlobalLoss(),
        EpochTripletLoss(),
        EpochCELoss(),
        EpochKDLoss(),
        EpochMMDLoss(),
        EpochGlobalLoss(),
        MyEvalExpAccuracy(),
        #PlotFeaturesPlugin(class_names),
        #LRPlugin(),
        loggers=[interactive_logger, neptune_logger],
    )

    # CREATE THE STRATEGY INSTANCE (NAIVE)
    cl_strategy = ILFGIR_strategy(
                        model, 
                        optim,
                        CrossEntropyLoss(), 
                        bestModelPath=bestModelPath,
                        #test_stream=scenario.test_stream,
                        lr_scheduler=scheduler_lr,
                        train_mb_size=train_batch, 
                        train_epochs=train_epochs, 
                        eval_mb_size=eval_batch, 
                        device=device,
                        evaluator=eval_plugin,
                        eval_every=eval_every
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
            res = cl_strategy.train(experience, [scenario.test_stream[0:experience.current_experience+1]])
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