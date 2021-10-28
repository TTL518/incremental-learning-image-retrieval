import os

from torch.optim import lr_scheduler
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import copy
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
from metric.plugin_recall_at_k import RecallAtKPluginSingleTask, RecallAtKPluginAllTasks, RecallAtKPluginAllTasksAndCheckpoint
from metric.plugin_checkpoint_metric import CheckpointPluginMetric
from metric.plugin_plot_features import PlotFeaturesPlugin
from metric.plugin_triplet_loss_metric import MinibatchTripletLoss, EpochTripletLoss
from metric.plugin_ce_loss_metric import MinibatchCELoss, EpochCELoss
from metric.plugin_mmd_loss_metric import MinibatchMMDLoss, EpochMMDLoss
from metric.plugin_kd_loss_metric import MinibatchKDLoss, EpochKDLoss
from metric.plugin_global_loss_metric import MinibatchGlobalLoss, EpochGlobalLoss
from metric.plugin_coral_loss_metric import MinibatchCoralLoss, EpochCoralLoss
from metric.plugin_featsDist_loss import MinibatchFeatsDistLoss, EpochFeatsDistLoss
from metric.plugin_cl_accuracy import MyEvalExpAccuracy
from metric.plugin_centerDist_loss_metric import MinibatchCenterDistLoss, EpochCenterDistLoss


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
    initial_num_classes = 100
    task_label = False
    shuffle_classes_exp = False

    bias_classifier = False
    norm_weights_classsifier = True

    #bestModelPath = "saved_models/bestModel_2T_RN32_CIFAR100_CE+Triplet+MMD_HPO39.pth"
    #bestModelPath = "saved_models/bestModel_2T_RN32_CIFAR100_CE+Triplet+KD_HPO2.pth"
    #bestModelPath = "saved_models/prova_ce+coral.pth"
    
    #bestModelPath = "saved_models/bestModel_Inc_2T_RN32_CIFAR100_CE+Triplet+10xMMD_BS32_1.pth"
    bestModelPath = "saved_models/bestModel_Inc_2T_RN50_CUB200_CE+Triplet.pth"
    


    #optim = "SGD"
    #lr = 0.01
    #momentum = 0.9
    #weight_decay = 2e-4

    optim = "Adam"
    lr = 0.00001
    weight_decay = 5e-4
    
    train_batch = 32
    eval_batch = 32
    train_epochs = 201
    eval_every = 2

    dataset = "CUB200"
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

    #modelName = "CIFAR resnet32"
    modelName = "Resnet50 pretrained on imagenet"
    model = INCREMENTAL_ResNet(ResNet_N=50, pretrained=True, save_dir="net_checkpoints/", initial_num_classes=initial_num_classes)
    #model = LeNet_PP(initial_num_classes=2, bias_classifier=False, norm_classifier=True)
    #model = LeNet(initial_num_classes=2, bias_classifier=False, norm_classifier=True)
    #model = resnet32(bias_classifier=bias_classifier, norm_weights_classifier=norm_weights_classsifier, num_classes=initial_num_classes)
    #model = GoogLeNet(initial_num_classes=0, aux_logits=False)
    
    print(model)
    print("Parametri totali= ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    #OPTIMIZER CREATION
    optim = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    milestone = None
    scheduler_lr = None
    
    #optim = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay )
    #milestone = [39, 79, 99, 129]
    #scheduler_lr = MultiStepLR(optim, milestones=milestone, gamma=0.1)

    ################################## DIFFERENT LR FOR CLASSIFIER AND FEAT EXTRACTOR - BEGIN 

    # my_list = ['inc_classifier.classifier.weight']
    # base_params = dict(filter(lambda kv: kv[0] not in my_list, model.named_parameters()))
    # classifier_params = dict(filter(lambda kv: kv[0] in my_list, model.named_parameters()))

    # # print(f"modello: {base_params.keys()}")
    # # print(f"classififcatore: {classifier_params.keys()}")

    # optim = Adam([
    #             {'params': base_params.values()},
    #             {'params': classifier_params.values(), 'lr': 1e-4}
    #         ], lr=1e-5, weight_decay=2e-4)
    # milestone = None
    # scheduler_lr = None

    # optim = SGD([
    #             {'params': base_params.values()},
    #             {'params': classifier_params.values(), 'lr': 0.1}
    #         ], lr=0.01, momentum=0.9, weight_decay=2e-4)
    #milestone = [49, 99, 129]
    #scheduler_lr = MultiStepLR(optim, milestones=milestone, gamma=0.1)

    ################################## DIFFERENT LR FOR CLASSIFIER AND FEAT EXTRACTOR - END


    #initial_model_path = "saved_models/initial_model_RN32_CIFAR100_50classes_New5.pth"
    initial_model_path = "saved_models/initial_model_RN50_CUB200_100classes.pth"
    checkpoint = torch.load(initial_model_path)
    #print(checkpoint['model_state_dict'])
    model.load_state_dict(checkpoint['model_state_dict'])
    #optim.load_state_dict(checkpoint['optimizer_state_dict'])

    prev_model_frozen = copy.deepcopy(model).to(device)
    prev_model_frozen.load_state_dict(checkpoint['model_state_dict'])

    
    # DEFINE  LOGGERS

    interactive_logger = InteractiveLogger()

    '''
    If you want to use Neptune logger uncomment this block and set the project_name and the api_token of your account, and add the logger to the evaluation plugin
    '''
    project_name = "tibi/TESIMAGNEW"
    
    run_id = ""
    #run_name = "TaskInc-CE+Triplet+MMD-CUB-ResNet50_512"
    #run_name = "2TaskInc-CE+Triplet+MMD-CIFAR100-ResNet32"
    #run_name = "2Task_Inception_CUB_CE+Triplet+MMD"
    #run_name = "2Task_CIFAR100_CE+Coral"
    #run_name = "2TaskInc-CE+Triplet+10xMMD-BS32-CIFAR100-ResNet32"
    run_name = "2TaskInc-CE+Triplet-CUB200-ResNet50"

    #description = "MMD. Incremental training starting from second task (scenario with 2 task) using CE + Triplet + 10xMMD. Adam 0.00001. BS32. No weight-decay"
    description = "Fine Tune. CUB200. Incremental training starting from second task (scenario with 2 task) using CE + Triplet. Adam 0.00001."
    
    #description = "Incremental training starting from second task (scenario with 2 task) using CE + Triplet + 10XMMD. Used A-Softmax. NEW mmd implementation. Adam 0.00001. BS32. RecCheckpoint. DropLastTrue "
    #description = "Incremental training starting from second task (scenario with 2 task) using CE + Triplet + KD. Used A-Softmax. Adam 0.00001. BS32. RecCheckpoint "
    #description = "Incremental training 2 Task. CIFAR100. ResNet32. Loss used CE+Triplet+MMD+KD. With classifier weights normalization and Bias False. 140 epochs.OLD MMD"
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
            "Learning rate":lr,
            "Milestone": milestone
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
        "Model": modelName,
        "Initial model path": initial_model_path

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
        #RecallAtKPluginAllTasks(),
        RecallAtKPluginAllTasksAndCheckpoint(bestModelPath),
        #CheckpointPluginMetric(bestModelPath),
        MinibatchTripletLoss(),
        MinibatchCELoss(),
        MinibatchKDLoss(),
        MinibatchMMDLoss(),
        MinibatchGlobalLoss(),
        MinibatchCoralLoss(),
        MinibatchCenterDistLoss(),
        MinibatchFeatsDistLoss(),
        EpochTripletLoss(),
        EpochCELoss(),
        EpochKDLoss(),
        EpochMMDLoss(),
        EpochGlobalLoss(),
        EpochCoralLoss(),
        EpochCenterDistLoss(),
        EpochFeatsDistLoss(),
        MyEvalExpAccuracy(),
        #PlotFeaturesPlugin(class_names),
        #LRPlugin(),
        loggers=[interactive_logger, neptune_logger],
    )

    # CREATE THE STRATEGY INSTANCE (NAIVE)
    cl_strategy = ILFGIR_strategy(
                        model, 
                        prev_model_frozen,
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

    


    for i in range(len(scenario.train_stream)-1):
        print("Start of experience: ", scenario.train_stream[i+1].current_experience)
        print("Current Classes: ", scenario.train_stream[i+1].classes_in_this_experience)

        # train returns a dictionary which contains all the metric values
        kwargs = {"drop_last": True}
        res = cl_strategy.train(scenario.train_stream[i+1], [scenario.test_stream[0:i+2]], **kwargs)
        print('Training completed')

        #Eval su tutte le experience passate
        results.append(cl_strategy.eval(scenario.test_stream[0:i+2]))
