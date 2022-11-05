import numpy as np
import math
import functools
import random
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.font_manager

from numpy import percentile
import pandas as pd

import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P

from .layers import ccbn, identity, SNLinear, SNEmbedding

from .utils import prepare_z_y, active_sampling_V1, sample_selector_V1
from .losses import loss_dis_fake, loss_dis_real
from .pyod_utils import AUC_and_Gmean


#network of transformation
class Transformation(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128, output_dim=128, hidden_layers=1, init='ortho'):
        super(Transformation, self).__init__()
        
        self.input_dim = input_dim    #feature dimension of input data
        self.hidden_dim = hidden_dim  #dimension of hidden layer
        self.output_dim = output_dim  #dimension of output layer, default=input_dim
        self.hidden_layers = hidden_layers   #number of hidden layers

        self.which_linear = nn.Linear

        self.input_fc = self.which_linear(self.input_dim, self.hidden_dim)

        self.output_fc = self.which_linear(self.hidden_dim, output_dim)

        self.model = nn.Sequential(self.input_fc,
                                   nn.ReLU())

        for index in range(self.hidden_layers):
            middle_fc = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.model.add_module('hidden-layers-{0}'.format(index), middle_fc)
            self.model.add_module('ReLu-{0}'.format(index), nn.ReLU())
            # self.model.add_module('ReLu-{0}'.format(index), nn.Tanh())

        self.model.add_module('output_layer', self.output_fc)

        self.init_weights()

    def init_weights(self):
        #used for init the parameters
        self.param_count = 0
        for module in self.modules():
            if (isinstance(module, nn.Linear)
                    or isinstance(module, nn.Embedding)):
                if self.init == 'ortho':
                    init.orthogonal_(module.weight)
                elif self.init == 'N02':
                    init.normal_(module.weight, 0, 0.02)
                elif self.init in ['glorot', 'xavier']:
                    init.xavier_uniform_(module.weight)
                else:
                    print('Init style not recognized...')
                self.param_count += sum([p.data.nelement() for p in module.parameters()])
        print('Param count for Transformation''s initialized parameters: %d' % self.param_count)

    def forward(self, x):
        h = self.model(x)
        return h


#network of encoder
class Encoder(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=64, output_dim=1, hidden_layers=1, init='ortho'):
        super(Encoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers

        self.init = init

        self.which_linear = nn.Linear
        
        self.input_fc = self.which_linear(input_dim, self.hidden_dim)
        self.output_fc = self.which_linear(self.hidden_dim, output_dim)
        # Embedding for projection discrimination
        self.model = nn.Sequential(self.input_fc,
                                   nn.ReLU())

        # self.blocks = []
        for index in range(self.hidden_layers):
            middle_fc = self.which_linear(self.hidden_dim, self.hidden_dim)
            self.model.add_module('hidden-layers-{0}'.format(index), middle_fc)
            self.model.add_module('ReLu-{0}'.format(index), nn.ReLU())
            # self.model.add_module('ReLu-{0}'.format(index), nn.Sigmoid())
        self.model.add_module('output_layer', self.output_fc)

        self.init_weights()

    # Initialize
    def init_weights(self):
        self.param_count = 0
        for module in self.modules():
            if (isinstance(module, nn.Linear)):
                if self.init == 'ortho':
                    init.orthogonal_(module.weight)
                elif self.init == 'N02':
                    init.normal_(module.weight, 0, 0.02)
                elif self.init in ['glorot', 'xavier']:
                    init.xavier_uniform_(module.weight)
                else:
                    print('Init style not recognized...')
                self.param_count += sum([p.data.nelement() for p in module.parameters()])
        # print('Param count for D''s initialized parameters: %d' % self.param_count)

    def forward(self, x, y=None, mode=0):
        # mode 0: train the whole discriminator network
        h = self.model(x)
        return h
        

class AnomalyDector(nn.Module):
    def __init__(self, args, data_x, data_y, test_x, test_y):
        super(AnomalyDector, self).__init__()

        lr = args.lr

        self.device = torch.device("cuda" if args.cuda else "cpu")
        self.args = args
        self.data_x = data_x
        self.data_y = data_y
        self.test_x = test_x
        self.test_y = test_y
        self.iterations = 0
        
        self.feature_size = data_x.shape[1]
        self.data_size = data_x.shape[0]
        self.batch_size = min(args.batch_size, self.data_size)
        self.hidden_dim = self.feature_size // 2   #hidden feature dimension

        manualSeed = random.randint(1, 10000)
        random.seed(manualSeed)
        torch.manual_seed(manualSeed)

        # 1: encoder
        self.encoder = Encoder(input_dim=self.feature_size, hidden_dim=self.hidden_dim, output_dim=self.hidden_dim, 
                                hidden_layers=args.hidden_layers_encoder, init=args.init)
        
        # 2: create ensemble of transformations
        self.trans_Ensemble = []
        parameters_trans = list()
        
        for index in range(args.ensemble_num):
            trans = Transformation()
            self.trans_Ensemble += [trans]
            parameters_trans += list(trans.parameters())
        
        # 3: optimizer
        self.optimizer = optim.Adam(parameters_trans+list(self.encoder.parameters()), lr=lr, betas=(0.00, 0.99))

    def fit(self):
        log_dir = os.path.join('./log/', self.args.data_name)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Start iteration
        Best_Measure_Recorded = -1
        best_auc = 0
        best_gmean = 0
        self.train_history = defaultdict(list)
        for epoch in range(self.args.max_epochs):
            train_AUC, train_Gmean, test_auc, test_gmean = self.train_one_epoch(epoch)
            if train_Gmean * train_AUC > Best_Measure_Recorded:
                Best_Measure_Recorded = train_Gmean * train_AUC
                best_auc = test_auc
                best_gmean = test_gmean
                states = {
                    'epoch': epoch,
                    'encoder': self.encoder.state_dict(),
                    'auc': train_AUC
                }
                for i in range(self.args.ensemble_num):
                    netD = self.trans_Ensemble[i]
                    states['trans_dict' + str(i)] = netD.state_dict()
                torch.save(states, os.path.join(log_dir, 'checkpoint_best.pth'))
            # print(train_AUC, test_AUC, epoch)
            if self.args.print:
                print('Training for epoch %d: Train_AUC=%.4f train_Gmean=%.4f Test_AUC=%.4f  Test_Gmean=%.4f' % (
                epoch + 1, train_AUC, train_Gmean, test_auc, test_gmean))

        # step 1: load the best models
        self.Best_Ensemble = []
        states = torch.load(os.path.join(log_dir, 'checkpoint_best.pth'))
        self.encoder.load_state_dict(states['encoder_dict'])
        for i in range(self.args.ensemble_num):
            netD = self.trans_Ensemble[i]
            netD.load_state_dict(states['trans_dict' + str(i)])
            self.Best_Ensemble += [netD]

        return best_auc, best_gmean

    def predict(self, test_x, test_y, dis_Ensemble=None):
        anomaly_scores = []
        y = []

        data_size = test_x.shape[0]
        num_batches = data_size // self.batch_size
        num_batches = num_batches + 1 if data_size % self.batch_size > 0 else num_batches

        for index in range(num_batches):
            end_pos = min(data_size, (index + 1) * self.batch_size)
            real_x = test_x[index * self.batch_size: end_pos]
            real_y = test_y[index * self.batch_size: end_pos]
            #to do: calculate the anomaly score

            
        anomaly_scores = torch.cat(anomaly_scores, 0).cpu().detach().numpy()
        y = torch.cat(y, 0).cpu().detach().numpy()
        return y, anomaly_scores

    def train_one_epoch(self, epoch=1):
        
        data_size = self.data_x.shape[0]
        # feature_size = data_x.shape[1]
        batch_size = min(self.args.batch_size, data_size)

        num_batches = data_size // batch_size
        num_batches = num_batches + 1 if data_size % batch_size > 0 else num_batches

        perm_index = torch.randperm(data_size)
        self.data_x = self.data_x[perm_index]
        self.data_y = self.data_y[perm_index]
        

        anomaly_score = []

        for index in range(num_batches):
            # step 1: train the ensemble of discriminator
            # Get training data
            self.iterations += 1

            end_pos = min(data_size, (index + 1) * batch_size)
            real_x = self.data_x[index * batch_size: end_pos]
            
            #1: transformation and save the transformed x into x_transformed
            x_transformed = []
            latent_vectors = []
            for i in range(self.args.ensemble_num):
                trans = self.trans_Ensemble[i]
                x = trans(real_x)
                x_transformed += [x]
                latent_vectors += [self.encoder(x)]
            
            #2: get the latent vector of real_x by directly feeding real_x to encoder
            x_encoded = self.encoder(real_x)

            #3: get the loss, and BP
            loss = self.get_loss(x_encoded, x_transformed)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            anomaly_score += [loss.detach().clone()]

        anomaly_score = torch.cat(anomaly_score, 0)
        y_pred = torch.zeros_like(anomaly_score)
        y_pred[anomaly_score>=self.args.threshold] = 1

        auc_train, gmean_train = AUC_and_Gmean(y_pred, self.real_y)
        auc_test, gmean_test = self.evaluate()
        #todo: calculate the empirical performance

        return auc_train, gmean_train, auc_test, gmean_test
    
    def evaluate(self):
        data_size = self.test_x.shape[0]
        # feature_size = data_x.shape[1]
        batch_size = min(self.args.batch_size, data_size)

        num_batches = data_size // batch_size
        num_batches = num_batches + 1 if data_size % batch_size > 0 else num_batches

        anomaly_score = []

        for index in range(num_batches):
            # step 1: train the ensemble of discriminator
            # Get training data
            self.iterations += 1

            end_pos = min(data_size, (index + 1) * batch_size)
            real_x = self.test_x[index * batch_size: end_pos]
            
            #1: transformation and save the transformed x into x_transformed
            x_transformed = []
            latent_vectors = []
            for i in range(self.args.ensemble_num):
                trans = self.trans_Ensemble[i]
                x = trans(real_x)
                x_transformed += [x]
                latent_vectors += [self.encoder(x)]
            
            #2: get the latent vector of real_x by directly feeding real_x to encoder
            x_encoded = self.encoder(real_x)

            #3: get the loss, and BP
            loss = self.get_loss(x_encoded, x_transformed)
            anomaly_score += [loss.detach().clone()]
        
        anomaly_score = torch.cat(anomaly_score, 0)
        y_pred = torch.zeros_like(anomaly_score)
        y_pred[anomaly_score>=self.args.threshold] = 1
        auc, gmean = AUC_and_Gmean(y_pred, self.test_y)
        return auc, gmean


    
    def get_loss(self, x, x_transformed):
        loss = 0
        for i, xi in enumerate(x_transformed):
            h_x_xi = torch.cosine_similarity(x, xi)
            h_denominator = 0
            for j, xj in enumerate(x_transformed):
                if i==j:
                    continue
                h_denominator += torch.cosine_similarity(xi, xj)
            
            loss_i = -torch.log(h_x_xi/(h_x_xi+h_denominator))
            loss += loss_i
        return loss





