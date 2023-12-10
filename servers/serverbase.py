import torch
import os
from utils.data import read_client_data
import numpy as np
import loguru as log
from loguru import logger
import random
import copy
import sys
LOG_PATH = 'log'
# 配置文件日志记录器
flogger = logger.bind(sink=os.path.join(LOG_PATH,'result.log'))
flogger.add("file.log", rotation="500 MB")

clogger = logger.bind(sink=sys.stdout)
clogger.add(sys.stdout, colorize=True)

class Server:
    def __init__(self,args,times) -> None:
        self.args = args
        self.device = args.device
        self.dataset = args.dataset
        self.num_clients = args.num_clients
        
        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_train_loss = []
    
    def set_clients(self,clientObj):
        for i in range(self.num_clients):
            train_data = read_client_data(self.dataset,i,is_train=True)
            test_data = read_client_data(self.dataset,i,is_train=False)
            client = clientObj(self.args,id = i,
                               train_samples = len(train_data),
                               test_samples = len(test_data),
                               )
            self.clients.append(client)
    
    def select_clients(self):
        if self.random_join_ratio:
            self.current_num_join_clients = np.random.choice(self.num_join_clients,self.num_clients+1)
        else:
            self.current_num_join_clients = self.num_join_clients
        selected_clients = list(np.random.choice(self.clients,self.current_num_join_clients,replace=False))
        return selected_clients

    def send_models(self):
        if len(self.clients) <= 0:
            logger.exception(f"couldn't find {self.clients} in models")
        
        active_clients = random.sample(
            self.select_clients,int((1-self.client_drop_rate))
        )
        
        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in active_clients:
            client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            if client_time_cost <= self.time_threthold:
                total_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model)
        
        for i,w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / total_samples
    
    def aggregate_parameters(self):
        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero()
        
        for w,client_model in zip(self.uploaded_weights,self.uploaded_models):
            self.add_parameters(w,client_model)
    
    def add_parameters(self,w,client_model):
        for server_param,client_param in zip(self.global_model.parameters(),client_model.parameters()):
            server_param.data += client_param() * w
    
    def save_global_model(self):
        model_path = os.path.join('models',self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path,self.algorithm + '_server' + '.pt')
        torch.save(self.global_model,model_path)
    
    def save_results(self):
        flogger.info(f'rs_test_acc:{self.rs_test_acc},
                     rs_test_auc:{self.rs_test_auc},
                     rs_train_loss:{self.rs_train_loss}')
    
    
    def test_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()

        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.clients:
            ct,ns,auc = c.test_metrics()
            tot_correct.append(ct * 1.0)
            tot_auc.append(auc * ns)
            num_samples.append(ns)
        
        ids = [c.id for c in self.clients]
        
        return ids,num_samples,tot_correct,tot_auc
    
    def train_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            return [0],[1],[0]
        
        num_samples = []
        losses = []
        for c in self.clients:
            cl,ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl * 1.0)
        
        ids = [c.id for c in self.clients]
        return ids,num_samples,losses
    
    def evaluate(self,acc=None,loss=None):
        stats = self.test_metrics()
        
    def evaluate(self,acc=None,loss=None):
        
        