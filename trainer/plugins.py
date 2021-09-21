import copy
from torch._C import device
from torch.cuda.memory import caching_allocator_alloc
from torch.optim import Adam

from avalanche.training.plugins import StrategyPlugin
from avalanche.training.strategies import BaseStrategy
from avalanche.models import avalanche_forward, DynamicModule, MultiTaskModule
from numpy import positive
import torch
from torch.autograd import backward
from models.loss import mmd_loss, kd_loss, MMD_loss
from torch.nn import CrossEntropyLoss

from pytorch_metric_learning import losses, miners



class ILFGIR_plugin(StrategyPlugin):
    """

    """

    def __init__(self):
        super().__init__()
        self.tripletLoss = torch.tensor(0.0)
        self.ceLoss = torch.tensor(0.0)
        self.kdLoss = torch.tensor(0.0)
        self.mmdLoss = torch.tensor(0.0)

        self.prev_model_frozen = None
        
        self.mb_out_flattened_features = None
        self.out_logits = None
        self.Triplet_Loss = losses.TripletMarginLoss(margin=1.0)
        self.CELoss = CrossEntropyLoss()
        self.miner = miners.BatchEasyHardMiner()
        self.MMDLoss = MMD_loss()
          
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
        #print("----------")
        #print("Strategy mb out ", strategy.mb_output.shape)
        #print("Strategy mb y ", torch.unique(strategy.mb_y))
        #print("Logits shape: ", strategy.mb_output.shape)
        #print("Flattened feature shape: ", self.mb_out_flattened_features.shape)
        #print(strategy.optimizer.param_groups[0]['lr'])
        #print("----------")

    def before_backward(self, strategy: 'BaseStrategy', **kwargs):
        """ 
        If is the first experience add Triplet Loss to Cross entropy loss.
        If isn't the First experience Compute a new loss=MMD+Triplet+Distillation+CE
        """

        if self.prev_model_frozen is None:
            self.ceLoss = strategy.loss
            #CALCOLO E AGGIUNGO TRIPLET LOSS ALLA CE
            #hard_pairs = self.miner(self.mb_out_flattened_features, strategy.mb_y)
            #self.tripletLoss =  self.Triplet_Loss(self.mb_out_flattened_features, strategy.mb_y, hard_pairs)
            #strategy.loss += self.tripletLoss 
        else:
            """Calcolo nuova loss perche avalanche in automatica calcola CE su tutti i logits invece va calcolata solo su i logits relativi alle nuove classi e con le vecchie calcolo KD"""
            #print("Experience successiva alla prima, prima del backward calcolo loss nuova MMD + KD + Triplet + Cross entropy")
            #print("Shape mb_x: ", strategy.mb_x.shape)
            #FORWARD SU MODELLO PRECEDENTE
            frozen_prev_model_logits, frozen_prev_model_flattened_features, _ , _ = self.prev_model_frozen(strategy.mb_x)
            #print("Shape frozen_model_logits: ", frozen_model_logits.shape)
            #print("Shape frozen_model_flattened_features: ", frozen_model_flattened_features.shape)
            
            #CALCOLO TRIPLET LOSS
            #hard_pairs = self.miner(self.mb_out_flattened_features, strategy.mb_y)
            #self.tripletLoss =  self.Triplet_Loss(self.mb_out_flattened_features, strategy.mb_y, hard_pairs)

            #CALCOLO MMD LOSS
            #self.mmdLoss = mmd_loss(self.mb_out_flattened_features, frozen_prev_model_flattened_features)
            self.mmdLoss = self.MMDLoss(self.mb_out_flattened_features, frozen_prev_model_flattened_features)
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
            #self.kdLoss = kd_loss(n_new_images, mb_old_class_logits, frozen_prev_model_logits)

            #NUOVA LOSS DA OTTIMIZZARE
            strategy.loss = self.ceLoss + self.mmdLoss #  + self.tripletLoss + self.kdLoss   #aggiungo alla cross entropy calcolata solo sulle nuove classi le altre loss
    
    def before_training_epoch(self, strategy:'BaseStrategy', **kwargs):
        """ Before training epoch"""
        #print("STATO:", strategy.optimizer.state_dict)

    def before_training_exp(self, strategy:'BaseStrategy', **kwargs):
        """ Before training experience re-initialize the optimizer"""
        print("Istanzio nuovo optimizer prima di ogni experience")
        strategy.optimizer = Adam(strategy.model.parameters(), lr=0.0001)

    def after_training_exp(self, strategy:'BaseStrategy', **kwargs):
        """ After training experience copy the model"""
        self.prev_model_frozen = copy.deepcopy(strategy.model)

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


class ILFGIR_plugin_new(StrategyPlugin):
    """

    """
    def __init__(self, temperature=2):
        super().__init__()
        self.tripletLoss = torch.tensor(0.0)
        self.ceLoss = torch.tensor(0.0)
        self.kdLoss = torch.tensor(0.0)
        self.mmdLoss = torch.tensor(0.0)

        self.prev_model = None
        self.temperature = temperature
        self.prev_classes = {'0': set()}

        self.mb_out_flattened_features = None
        self.out_logits = None
        
        self.Triplet_Loss = losses.TripletMarginLoss(margin=1.0)
        self.miner = miners.BatchEasyHardMiner()

        

    def _distillation_loss(self, out, prev_out, active_units):
        """
        Compute distillation loss between output of the current model and
        and output of the previous (saved) model.
        """

        log_p = torch.log_softmax(out / self.temperature, dim=1)
        q = torch.softmax(prev_out / self.temperature, dim=1)
        print("SHAPE LOG P", log_p.shape)
        print("SHAPE Q", q.shape)
        res = torch.nn.functional.kl_div(log_p, q, reduction='none')
        res = res[:, list(active_units)]  # mask unused units
        res = res.sum() / out.shape[0]  # batch-mean
        return res
    
    def penalty(self, out, x, alpha, curr_model):
        """
        Compute weighted distillation loss.
        """

        if self.prev_model is None:
            return 0
        else:
            with torch.no_grad():
                if isinstance(self.prev_model, MultiTaskModule):
                    # output from previous output heads.
                    y_prev, feats_prev = avalanche_forward(self.prev_model, x, None)
                    # in a multitask scenario we need to compute the output
                    # from all the heads, so we need to call forward again.
                    # TODO: can we avoid this?
                    y_curr, feats_curr = avalanche_forward(curr_model, x, None)
                else:  # no task labels
                    logits, feats = self.prev_model(x)
                    y_prev = {'0': logits}
                    y_curr = {'0': out}

            dist_loss = 0
            for task_id in y_prev.keys():
                # compute kd only for previous heads.
                if task_id in self.prev_classes:
                    yp = y_prev[task_id]
                    yc = y_curr[task_id]
                    au = self.prev_classes[task_id]
                    dist_loss += self._distillation_loss(yc, yp, au)
            return alpha * dist_loss

    def after_forward(self, strategy: 'BaseStrategy', **kwargs):
        '''
        After forward since the model output two values (logits, out_flattened_features), out_flattened_feature are taken to compute MMD and Triplet,
        and strategy.mb_output is set to only logits, to compute CE with avalanche.
        '''
        self.mb_out_flattened_features = strategy.mb_output[1]
        strategy.mb_output = strategy.mb_output[0]

    def before_backward(self, strategy: 'BaseStrategy', **kwargs):
        print("SHAPE mb out", strategy.mb_output.shape)
        penalty = self.penalty(strategy.mb_output, strategy.mb_x, 1,
                               strategy.model)
        strategy.loss += penalty

    def after_training_exp(self, strategy:'BaseStrategy', **kwargs):
        """ After training experience copy the model"""
        self.prev_model = copy.deepcopy(strategy.model)
        task_ids = strategy.experience.dataset.task_set
        for task_id in task_ids:
            task_data = strategy.experience.dataset.task_set[task_id]
            pc = set(task_data.targets)

            if task_id not in self.prev_classes:
                self.prev_classes[str(task_id)] = pc
            else:
                self.prev_classes[str(task_id)] = self.prev_classes[task_id]\
                    .union(pc)

    def after_eval_forward(self, strategy: 'BaseStrategy', **kwargs):
        '''
        After forward since the model output two values (logits, out_flattened_features), out_flattened_feature are taken to compute MMD and Triplet,
        and strategy.mb_output is set to only logits, to compute CE with avalanche.
        '''
        self.mb_out_flattened_features = strategy.mb_output[1]
        strategy.mb_output = strategy.mb_output[0]