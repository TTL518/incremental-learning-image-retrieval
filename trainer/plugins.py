import copy
from torch._C import device
from torch.cuda.memory import caching_allocator_alloc
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F

from avalanche.training.plugins import StrategyPlugin
from avalanche.training.strategies import BaseStrategy
from avalanche.models import avalanche_forward, DynamicModule, MultiTaskModule
from numpy import positive
import torch
from torch.autograd import backward
from models.loss import mmd_loss, kd_loss, MMD_loss, DistillationLoss
from torch.nn import CrossEntropyLoss

from pytorch_metric_learning import losses, miners, distances



class ILFGIR_plugin(StrategyPlugin):
    """

    """

    def __init__(self, bestModelPath, scheduler_lr):
        super().__init__()
        self.global_loss = torch.tensor(0.0)
        self.tripletLoss = torch.tensor(0.0)
        self.ceLoss = torch.tensor(0.0)
        self.kdLoss = torch.tensor(0.0)
        self.mmdLoss = torch.tensor(0.0)
        
        self.bestModelPath = bestModelPath
        self.scheduler_lr = scheduler_lr
        self.prev_model_frozen = None
        
        self.mb_out_flattened_features = None
        self.out_logits = None
        self.Triplet_Loss = losses.TripletMarginLoss(distance = distances.DotProductSimilarity(), margin=1.0)
        self.CELoss = CrossEntropyLoss()
        self.miner = miners.BatchHardMiner()
        self.MMDLoss = MMD_loss(kernel_num=10)
          
    def before_training(self, strategy: 'BaseStrategy', **kwargs):
        '''Before incremental training load the starting model'''
        #Comment line below if you start the incremental train from the first task
        #print("Effettuo copia del modello, uno rimarrà invariato (il teacher) , l'altro imparerà nuovo task cercando di non dimenticare i precedenti (teacher) ")
        #self.prev_model_frozen =  copy.deepcopy(strategy.model)
        
        '''
        #checkpoint = torch.load("saved_models/BaseModelForCL02_CE+Triplet.pt")
        #strategy.model.load_state_dict(checkpoint['model_state_dict'])
        #self.prev_model_frozen.load_state_dict(checkpoint['model_state_dict'])
        #strategy.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        #strategy.loss = checkpoint['loss']
        '''
        
    def after_training(self, strategy: 'BaseStrategy', **kwargs):
        '''After training save the model'''
        #Comment line below if you not want save the model at the end of the complete training
        #torch.save({
        #           'model_state_dict':strategy.model.state_dict(), 
        #           'optimizer_state_dict': strategy.optimizer.state_dict(),
        #           'loss': strategy.loss,
        #           }, "saved_models/BaseModelForCL02_CE+Triplet.pt")

    def after_forward(self, strategy: 'BaseStrategy', **kwargs):
        '''
        After forward since the model output two values (logits, out_flattened_features), out_flattened_feature are taken to compute MMD and Triplet,
        and strategy.mb_output is set to only logits, to compute CE with avalanche.
        '''
        self.mb_out_flattened_features = strategy.mb_output[1]
        strategy.mb_output = strategy.mb_output[0]
        # print("----------")
        # print("Strategy mb out ", strategy.mb_output.shape)
        # print("Strategy mb y ", torch.unique(strategy.mb_y))
        # print("Logits shape: ", strategy.mb_output.shape)
        # print("Flattened feature shape: ", self.mb_out_flattened_features.shape)
        # print(strategy.optimizer.param_groups[0]['lr'])
        # print("----------")

    def before_backward(self, strategy: 'BaseStrategy', **kwargs):
        """ 
        If is the first experience add Triplet Loss to Cross entropy loss.
        If isn't the First experience Compute a new loss=MMD+Triplet+Distillation+CE
        """
        
        if self.prev_model_frozen is None:
            self.ceLoss = strategy.loss
            #CALCOLO E AGGIUNGO TRIPLET LOSS ALLA CE
            hard_pairs = self.miner(self.mb_out_flattened_features, strategy.mb_y)
            self.tripletLoss =  self.Triplet_Loss(self.mb_out_flattened_features, strategy.mb_y, hard_pairs)

            strategy.loss = self.ceLoss + self.tripletLoss 
            self.global_loss = strategy.loss
        else:
            """Calcolo nuova loss perche avalanche in automatica calcola CE su tutti i logits invece va calcolata solo su i logits relativi alle nuove classi e con le vecchie calcolo KD"""
            #print("Experience successiva alla prima, prima del backward calcolo loss nuova MMD + KD + Triplet + Cross entropy")
            #print("Shape mb_x: ", strategy.mb_x.shape)
            #FORWARD SU MODELLO PRECEDENTE
            frozen_prev_model_logits, frozen_prev_model_flattened_features = self.prev_model_frozen(strategy.mb_x)
            #print("Shape frozen_model_logits: ", frozen_model_logits.shape)
            #print("Shape frozen_model_flattened_features: ", frozen_prev_model_flattened_features1.shape)
            #print("Shape flattened_features: ", self.mb_out_flattened_features1.shape)

            #CALCOLO TRIPLET LOSS
            hard_pairs = self.miner(self.mb_out_flattened_features, strategy.mb_y)
            self.tripletLoss =  0.15*(self.Triplet_Loss(self.mb_out_flattened_features, strategy.mb_y, hard_pairs))

            #CALCOLO MMD LOSS
            #self.mmdLoss = mmd_loss(self.mb_out_flattened_features, frozen_prev_model_flattened_features)
            self.mmdLoss = (self.MMDLoss(self.mb_out_flattened_features,frozen_prev_model_flattened_features))
            print("MMD", self.mmdLoss)
            #self.mmdLoss2 = self.MMDLoss(self.mb_out_flattened_features2, frozen_prev_model_flattened_features2)
            #print(self.mmdLoss)

            #Separo logits dell'esperienze vecchie dai logits delle nuove classi nella nuova esperienza
            n_old_class = frozen_prev_model_logits.shape[1]
            n_new_class = strategy.mb_output.shape[1]-n_old_class
            print(" n_old_class: ", n_old_class)
            print(" n_new_class: ", n_new_class)
            mb_old_class_logits = strategy.mb_output[:, :n_old_class]
            #print("Shape  mb_old_class_logits: ",  mb_old_class_logits.shape)
            mb_new_class_logits = strategy.mb_output[:, n_old_class:]
            #print("Shape  mb_new_class_logits: ",  mb_new_class_logits.shape)
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            #CALCOLO CE SOLO SUI LOGITS DELLE CLASSI DEL TASK ATTUALE
            '''
            #CON MAPPING DELLE LABEL
            actual_class_values = torch.unique(strategy.mb_y, sorted=True)
            #print("Actual class values: ", actual_class_values)
           
            temp_class_values = torch.arange(n_new_class).to(device)
            #print("Temporary class values: ", temp_class_values)
            temp_mb_y = strategy.mb_y.clone()
            #for i in range(n_new_class):
            #    temp_mb_y[strategy.mb_y==actual_class_values[i]]=temp_class_values[i]
            #self.ceLoss = self.CELoss(mb_new_class_logits, temp_mb_y)
            '''

            #USANDO PARAMETRO WEIGHT
            ce_weights = torch.zeros(strategy.mb_output.shape[1]).to(device)
            ce_weights[n_old_class:] = 1
            print("ce weights: ", ce_weights)
            cross_entropy_loss = CrossEntropyLoss(weight=ce_weights)
            self.ceLoss = 0.15*(cross_entropy_loss(strategy.mb_output, strategy.mb_y))

            #CALCOLO KD LOSS SOLO SUI LOGITS DELLE CLASSI DEI TASK PRECEDENTI
            n_new_images = strategy.mb_x.shape[0]
            #print("number of new images in minibatch: ", n_new_images)
            self.kdLoss = 0.3*(kd_loss(n_new_images, mb_old_class_logits, frozen_prev_model_logits))
            #self.kdLoss = DistillationLoss(mb_old_class_logits, frozen_prev_model_logits)

            #NUOVA LOSS DA OTTIMIZZARE
            strategy.loss = self.ceLoss + self.tripletLoss + self.mmdLoss + self.kdLoss # #  + self.mmdLoss2   10mmd 2kd  #aggiungo alla cross entropy calcolata solo sulle nuove classi le altre loss
            #strategy.loss =  0.2*self.ceLoss + 0.1*self.tripletLoss + 0.4*self.mmdLoss + 0.3*self.kdLoss
            self.global_loss = strategy.loss
            
    def before_training_epoch(self, strategy:'BaseStrategy', **kwargs):
        """ Before training epoch"""
        #print("STATO:", strategy.optimizer.state_dict)
    
    def after_training_epoch(self, strategy:'BaseStrategy', **kwargs):
        """ Before training epoch"""
        if(self.scheduler_lr!=None):
            self.scheduler_lr.step()

    def before_training_exp(self, strategy:'BaseStrategy', **kwargs):
        """ Before training experience re-initialize the optimizer"""
        print("Istanzio nuovo optimizer prima di ogni experience")
        #strategy.optimizer = Adam(strategy.model.parameters(), lr=0.00001)

        strategy.optimizer = SGD(strategy.model.parameters(), lr=0.1, momentum=0.9, weight_decay=2e-4 )
        self.scheduler_lr = MultiStepLR(strategy.optimizer, milestones=[49, 69, 89, 109] , gamma=0.1) # [49, 69, 89, 109]
        
        #strategy.eval(strategy.test_stream[0:strategy.experience.current_experience+1])
        #strategy.model.train()

    def after_training_exp(self, strategy:'BaseStrategy', **kwargs):
        """ After training experience copy the model"""
        print("Carico miglior modello ottenuto durante il training e effettuo copia per training exp successiva")
        checkpoint = torch.load(self.bestModelPath)
        self.prev_model_frozen = copy.deepcopy(strategy.model)
        self.prev_model_frozen.load_state_dict(checkpoint['model_state_dict'])
        strategy.model.load_state_dict(checkpoint['model_state_dict'])
        

    def after_eval_forward(self, strategy: 'BaseStrategy', **kwargs):
        '''
        After forward since the model output two values (logits, out_flattened_features), out_flattened_feature are taken to compute MMD and Triplet,
        and strategy.mb_output is set to only logits, to compute CE with avalanche.
        '''
        self.mb_out_flattened_features = strategy.mb_output[1]
        strategy.mb_output = strategy.mb_output[0]


class Naive_plugin(StrategyPlugin):
    """

    """
    def __init__(self):
        super().__init__()

        self.mb_out_flattened_features = None
        self.out_logits = None
        
    def after_forward(self, strategy: 'BaseStrategy', **kwargs):
        '''
        After forward since the model output two values (logits, out_flattened_features), out_flattened_feature are taken,
        and strategy.mb_output is set to only logits, to compute CE with avalanche.
        '''
        self.mb_out_flattened_features = strategy.mb_output[1]
        strategy.mb_output = strategy.mb_output[0]
    
    def after_eval_forward(self, strategy: 'BaseStrategy', **kwargs):
        '''
        After forward since the model output two values (logits, out_flattened_features), out_flattened_feature are taken to compute MMD and Triplet,
        and strategy.mb_output is set to only logits, to compute CE with avalanche.
        '''
        self.mb_out_flattened_features = strategy.mb_output[1]
        strategy.mb_output = strategy.mb_output[0]


class CIFLIR(StrategyPlugin):
    """

    """
    def __init__(self, best_model_path):
        super().__init__()
        self.tripletLoss = torch.tensor(0.0)
        self.ceLoss = torch.tensor(0.0)
        self.kdLoss = torch.tensor(0.0)
        self.mmdLoss = torch.tensor(0.0)

        self.prev_model_frozen = None
        self.best_model_path = best_model_path

        self.prev_features_list = []
        self.prev_y_list = []
        self.prev_features_tensor = None
        self.prev_y_tensor = None

        self.mb_out_flattened_features = None
        self.out_logits = None
        
        self.Triplet_Loss = losses.TripletMarginLoss(margin=1.0)
        self.miner = miners.BatchEasyHardMiner(pos_strategy="hard", neg_strategy="semihard")
        self.MMDLoss = MMD_loss(kernel_num=10)

    def before_training(self, strategy: 'BaseStrategy', **kwargs):
        '''Before incremental training'''

    def before_training_exp(self, strategy:'BaseStrategy', **kwargs):
        """ Before training experience re-initialize the optimizer"""
        print("Istanzio nuovo optimizer prima di ogni experience")
        strategy.optimizer = Adam(strategy.model.parameters(), lr=0.001)
        self.prev_features_list = []
        self.prev_y_list = []

    def before_training_epoch(self, strategy: 'BaseStrategy', **kwargs):
        '''Before training epoch control if is the last epoch to save the features (and its labels) extracted from the best model'''
        if(strategy.epoch == strategy.train_epochs-1):
            print("Ultima epoca mi carico il miglior modello per salvare features")
            checkpoint = torch.load(self.best_model_path)
            self.prev_model_frozen = copy.deepcopy(strategy.model)
            self.prev_model_frozen.load_state_dict(checkpoint['model_state_dict'])
    
    def before_training_iteration(self, strategy: 'BaseStrategy', **kwargs):
        '''Before training iteration save features extracted from the best model'''
        if(strategy.epoch == strategy.train_epochs-1):
            print("Ultima epoca salvo features e labels")
            _ , feats = self.prev_model_frozen(strategy.mb_x)
            self.prev_features_list.append(feats)
            self.prev_y_list.append(strategy.mb_y)

    def after_forward(self, strategy: 'BaseStrategy', **kwargs):
        '''
        After forward since the model output two values (logits, out_flattened_features), out_flattened_feature are taken to compute MMD and Triplet,
        and strategy.mb_output is set to only logits, to compute CE with avalanche.
        '''
        self.mb_out_flattened_features1 = strategy.mb_output[1]
        strategy.mb_output = strategy.mb_output[0]
        #print("----------")
        #print("Strategy mb y ", torch.unique(strategy.mb_y))
        #print("Logits shape: ", strategy.mb_output.shape)
        #print("Flattened feature shape: ", self.mb_out_flattened_features.shape)
        #print("----------")

    def before_backward(self, strategy: 'BaseStrategy', **kwargs):
        """ 
        If is the first experience add Triplet Loss to Cross entropy loss.
        If isn't the First experience Compute a new loss=MMD+Triplet+Distillation+CE
        """
        
        if self.prev_model_frozen is None:
            self.ceLoss = strategy.loss
            #CALCOLO E AGGIUNGO TRIPLET LOSS ALLA CE
            #hard_pairs = self.miner(self.mb_out_flattened_features1, strategy.mb_y)
            #self.tripletLoss =  self.Triplet_Loss(self.mb_out_flattened_features1, strategy.mb_y, hard_pairs)
            #strategy.loss += self.tripletLoss 
        else:
            """Calcolo nuova loss perche avalanche in automatica calcola CE su tutti i logits invece va calcolata solo su i logits relativi alle nuove classi e con le vecchie calcolo KD"""
            #print("Experience successiva alla prima, prima del backward calcolo loss nuova MMD + KD + Triplet + Cross entropy")
            #print("Shape mb_x: ", strategy.mb_x.shape)
            #FORWARD SU MODELLO PRECEDENTE
            frozen_prev_model_logits, frozen_prev_model_flattened_features1 = self.prev_model_frozen(strategy.mb_x)
            #print("Shape frozen_model_logits: ", frozen_model_logits.shape)
            #print("Shape frozen_model_flattened_features: ", frozen_prev_model_flattened_features1.shape)
            #print("Shape flattened_features: ", self.mb_out_flattened_features1.shape)

            #CALCOLO TRIPLET LOSS
            #hard_pairs = self.miner(self.mb_out_flattened_features1, strategy.mb_y)
            #self.tripletLoss =  self.Triplet_Loss(self.mb_out_flattened_features1, strategy.mb_y, hard_pairs)

            #CALCOLO MMD LOSS
            #self.mmdLoss = mmd_loss(self.mb_out_flattened_features, frozen_prev_model_flattened_features)
            self.mmdLoss1 = self.MMDLoss(self.mb_out_flattened_features1, frozen_prev_model_flattened_features1)
            #self.mmdLoss2 = self.MMDLoss(self.mb_out_flattened_features2, frozen_prev_model_flattened_features2)
            #print(self.mmdLoss)

            #Separo logits dell'esperienze vecchie dai logits delle nuove classi nella nuova esperienza
            n_old_class = frozen_prev_model_logits.shape[1]
            n_new_class = strategy.mb_output.shape[1]-n_old_class
            #print(" n_old_class: ", n_old_class)
            #print(" n_new_class: ", n_new_class)
            mb_old_class_logits = strategy.mb_output[:, :n_old_class]
            #print("Shape  mb_old_class_logits: ",  mb_old_class_logits.shape)
            mb_new_class_logits = strategy.mb_output[:, n_old_class:]
            #print("Shape  mb_new_class_logits: ",  mb_new_class_logits.shape)
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            #CALCOLO CE SOLO SUI LOGITS DELLE CLASSI DEL TASK ATTUALE
            '''
            #CON MAPPING DELLE LABEL
            actual_class_values = torch.unique(strategy.mb_y, sorted=True)
            #print("Actual class values: ", actual_class_values)
           
            temp_class_values = torch.arange(n_new_class).to(device)
            #print("Temporary class values: ", temp_class_values)
            temp_mb_y = strategy.mb_y.clone()
            #for i in range(n_new_class):
            #    temp_mb_y[strategy.mb_y==actual_class_values[i]]=temp_class_values[i]
            #self.ceLoss = self.CELoss(mb_new_class_logits, temp_mb_y)
            '''

            #USANDO PARAMETRO WEIGHT
            ce_weights = torch.zeros(strategy.mb_output.shape[1]).to(device)
            ce_weights[n_old_class:] = 1
            #print("ce weights: ", ce_weights)
            cross_entropy_loss = CrossEntropyLoss(weight=ce_weights)
            self.ceLoss = cross_entropy_loss(strategy.mb_output, strategy.mb_y)

            #CALCOLO KD LOSS SOLO SUI LOGITS DELLE CLASSI DEI TASK PRECEDENTI
            n_new_images = strategy.mb_x.shape[0]
            #print("number of new images in minibatch: ", n_new_images)
            self.kdLoss = kd_loss(n_new_images, mb_old_class_logits, frozen_prev_model_logits)

            #NUOVA LOSS DA OTTIMIZZARE
            strategy.loss = self.ceLoss +self.mmdLoss1 + self.kdLoss + self.tripletLoss  #+ self.mmdLoss2     #aggiungo alla cross entropy calcolata solo sulle nuove classi le altre loss

    def after_training_exp(self, strategy:'BaseStrategy', **kwargs):
        """ After training experience copy the model"""
        print("Carico miglior modello ottenuto durante il training ")
        checkpoint = torch.load(self.best_model_path)
        self.prev_model_frozen = copy.deepcopy(strategy.model)
        self.prev_model_frozen.load_state_dict(checkpoint['model_state_dict'])
        
        self.prev_features_tensor = torch.cat(self.prev_features_list)
        self.prev_y_tensor = torch.cat(self.prev_y_list)
        self.prev_features_tensor = torch.squeeze(self.prev_features_tensor)
        self.prev_y_tensor = torch.squeeze(self.prev_y_tensor)
    
    def after_training(self, strategy: 'BaseStrategy', **kwargs):
        '''After training save task features and its labels '''

    def after_eval_forward(self, strategy: 'BaseStrategy', **kwargs):
        '''
        After forward since the model output two values (logits, out_flattened_features), out_flattened_feature are taken to compute MMD and Triplet,
        and strategy.mb_output is set to only logits, to compute CE with avalanche.
        '''
        self.mb_out_flattened_features = strategy.mb_output[1]
        strategy.mb_output = strategy.mb_output[0]