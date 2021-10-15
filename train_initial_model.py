import os

from torch.optim import lr_scheduler
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

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
from metric.plugin_coral_loss_metric import MinibatchCoralLoss, EpochCoralLoss
from metric.plugin_cl_accuracy import MyEvalExpAccuracy


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    seed = 2
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

    bestModelPath = "saved_models/initial_model_RN32_CIFAR100_50classes.pth"

    optim = "SGD"
    lr = 0.1
    momentum = 0.9
    weight_decay = 2e-4

    #optim = "Adam"
    #lr = 0.00001
    
    train_batch = 128
    eval_batch = 128
    train_epochs = 151
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
    milestone = [49, 99, 129]
    scheduler_lr = MultiStepLR(optim, milestones=milestone, gamma=0.1)

    # optim = Adam([
    #             {'params': model.parameters() },
    #             {'params': model.fc.parameters(), 'lr': 1e-5},
    #         ], lr=1e-6)

    # DEFINE  LOGGERS

    interactive_logger = InteractiveLogger()

    project_name = "tibi/TESIMAG"
    
    run_id = ""

    run_name = "Training_Initial_Model_RN32_CIFAR100_50Classes"
    description = "Training of the initial model for incremental learning on CIFAR100 with ResNet32. 50 classes."
    
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
        MinibatchCoralLoss(),
        EpochTripletLoss(),
        EpochCELoss(),
        EpochKDLoss(),
        EpochMMDLoss(),
        EpochGlobalLoss(),
        EpochCoralLoss(),
        MyEvalExpAccuracy(),
        #PlotFeaturesPlugin(class_names),
        #LRPlugin(),
        loggers=[interactive_logger, neptune_logger],
    )

    # CREATE THE STRATEGY INSTANCE (NAIVE)
    cl_strategy = ILFGIR_strategy(
                        model,
                        None, 
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

    # train returns a dictionary which contains all the metric values
    res = cl_strategy.train(scenario.train_stream[0], [scenario.test_stream[0]])
    print('Training completed')

    #Eval su tutte le experience passate (TUTTE INSIEME)
    results.append(cl_strategy.eval(scenario.test_stream[0]))