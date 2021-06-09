import copy

from avalanche.training.plugins import StrategyPlugin
from avalanche.training.strategies import BaseStrategy
from numpy import positive
import torch
from models.loss import mmd_loss, kd_loss
from torch.nn import CrossEntropyLoss

from torch.nn import TripletMarginLoss

class ILFGIR_plugin(StrategyPlugin):
    """

    """

    def __init__(self):
        super().__init__()
        self.prev_model_frozen = None
        self.out_flattened_feature = None
        self.out_logits = None
        self.Triplet_Loss = TripletMarginLoss(margin=1.0, p=2)

    def before_forward(self, strategy: 'BaseStrategy', **kwargs):
        print(len(strategy.mbatch))
        

    def after_forward(self, strategy: 'BaseStrategy', **kwargs):
        '''
        Dopo il forward dato che il modello restituisce due output (logits, flattened_features), prendo le out_flattened_feature per calcolare MMD e Triplet,
        e setto strategy.mb_output ai soli logits per calcolare la CE automaticamente con avalanche.
        '''

        self.out_flattened_feature = strategy.mb_output[1]
        strategy.mb_output = strategy.mb_output[0]
        #print("----------")
        #print("Logits shape: ", strategy.mb_output.shape)
        #print("Flattened feature shape: ", self.out_flattened_feature.shape)
        #print("----------")
        

    def before_backward(self, strategy: 'BaseStrategy', **kwargs):
        """ If is the first experience add Triplet Loss to Cross entropy loss.
            If isn't the First experience Compute a new loss=MMD+Triplet+Distillation+CE"""
        
        anchor_img_flattened_feature = self.out_flattened_feature
        positive_img = strategy.mbatch[2]
        negative_img = strategy.mbatch[3]
        positive_img_flattened_feature = strategy.model(positive_img)[1] #il secondo output sono le flattened features
        negative_img_flattened_feature = strategy.model(negative_img)[1]

        #print("SHAPES")
        #print(anchor_img_flattened_feature.shape)
        #print(positive_img_flattened_feature.shape)
        #print(negative_img_flattened_feature.shape)

        if self.prev_model_frozen is None:
            print("Prima experience, prima del backward aggiungo la triplet alla cross entropy")
            triplet_loss = self.Triplet_Loss(anchor_img_flattened_feature, positive_img_flattened_feature, negative_img_flattened_feature)
            print("TRIPLET LOSS: ", triplet_loss)
            strategy.loss += triplet_loss 
            
        else:
            #Calcolo nuova loss perche avalanche in automatica calcola CE su tutti i logits invece va calcolata solo su i logits relativi alle nuove classi e con le vecchie si calcola KD
            print("Experience successiva alla prima, prima del backward calcolo loss nuova MMD + KD + Triplet + Cross entropy")
            print("Shape mb_x: ", strategy.mb_x.shape)
            frozen_model_logits, frozen_model_flattened_features = self.prev_model_frozen(strategy.mb_x)
            #print("Shape frozen_model_logits: ", frozen_model_logits.shape)
            #print("Shape frozen_model_flattened_features: ", frozen_model_flattened_features.shape)

            mmdLoss = mmd_loss(self.out_flattened_feature, frozen_model_flattened_features)
            tripletLoss = self.Triplet_Loss(anchor_img_flattened_feature, positive_img_flattened_feature, negative_img_flattened_feature)

            #Separo logits dell'esperienze vecchie dai logits delle nuove classi nella nuova esperienza
            n_old_class = frozen_model_logits.shape[1]
            #print(" n_old_class: ", n_old_class)

            mb_old_class_logits = strategy.mb_output[:, :n_old_class]
            #print("Shape  mb_old_class_logits: ",  mb_old_class_logits.shape)

            mb_new_class_logits = strategy.mb_output[:, n_old_class:]
            #print("Shape  mb_new_class_logits: ",  mb_new_class_logits.shape)

            n_new_images = strategy.mb_x.shape[0]
            #print("number of new images in minibatch: ", n_new_images)

            kdLoss = kd_loss(n_new_images, frozen_model_logits, mb_old_class_logits)

            actual_tot_n_class = strategy.mb_output.shape[1]
            #print("Actual tot n class: ", actual_tot_n_class)

            '''
            #TODO Gestire il calcolo della cross entropy solo sui logits delle nuove classi 

            #ce_weights = torch.zeros(actual_tot_n_class)
            #ce_weights[n_old_class:] = 1

            #print("ce weights: ", ce_weights)

            #cross_entropy_loss = CrossEntropyLoss()
            #ceLoss = cross_entropy_loss(mb_new_class_logits,strategy.mb_y)
            '''

            self.loss += mmdLoss + kdLoss + tripletLoss  #aggiungo alla cross entropy calcolata solo sulle nuove classi le altre loss


    def after_training_exp(self, strategy:'BaseStrategy', **kwargs):
        """ After training experience copy the model"""
        self.prev_model_frozen = copy.deepcopy(strategy.model)