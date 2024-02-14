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
    def __init__(self,args) -> None:
        self.args = args
        self.device = args['device']
        self.dataset = args['dataset']
        self.num_clients = args['num_clients']
        self.join_ratio = args['join_ratio']
        self.client_drop_rate = args['client_drop_rate']
        
        self.global_rounds = args['global_rounds']
        self.time_threthold = args['time_threthold']
        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_train_loss = []
        self.dataset_abs_dir = os.getcwd()
        self.clients = []
        self.random_clients_selected = args['random_clients_selected']
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        # false temporary
        self.eval_new_clients = False
        self.fine_tuning_epoch = 0
        self.num_new_clients = 0
        self.algorithm = args['algorithm']
        
    def set_clients(self,clientObj):
        for i in range(self.num_clients):
            train_data = read_client_data(self.dataset,i,self.dataset_abs_dir,is_train=True)
            test_data = read_client_data(self.dataset,i,self.dataset_abs_dir,is_train=False)
            client = clientObj(self.args,id = i,
                               train_samples = len(train_data),
                               test_samples = len(test_data),
                               )
            self.clients.append(client)
    
    def select_clients(self):
        logger.info('Starting select clients for server')
        if self.random_clients_selected:
            self.current_num_join_clients = np.random.choice(int(self.num_clients * self.join_ratio),self.num_clients+1)
        else:
            self.current_num_join_clients = self.num_join_clients
        selected_clients = list(np.random.choice(self.clients,self.current_num_join_clients,replace=False))
        return selected_clients

    def send_models(self):
        if len(self.clients) <= 0:
            logger.exception(f"couldn't find {self.clients} in models")
        logger.debug(type(self.select_clients))
        active_clients = random.sample(
            self.selected_clients,int((1-self.client_drop_rate))
        )
        
        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        total_samples = 0
        for client in active_clients:
            client_time_cost = client.train_time['total_cost'] / client.train_time['rounds'] + \
                        client.send_time['total_cost'] / client.send_time['rounds']
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
            param.data.zero_()
        
        for w,client_model in zip(self.uploaded_weights,self.uploaded_models):
            self.add_parameters(w,client_model)
    
    def add_parameters(self,w,client_model):
        for server_param,client_param in zip(self.global_model.parameters(),client_model.parameters()):
            server_param.data += client_param.data.clone() * w
    
    def save_global_model(self):
        model_path = os.path.join('models',self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path,self.algorithm + '_server' + '.pt')
        torch.save(self.global_model,model_path)
    
    def save_results(self):
        flogger.info(f"rs_test_acc:{self.rs_test_acc},rs_test_auc:{self.rs_test_auc},rs_train_loss:{self.rs_train_loss}")
    
    
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
        test_metrics_res = self.test_metrics()
        train_metrics_res = self.train_metrics()
        
        test_acc = sum(test_metrics_res[2]) * 1.0 / sum(test_metrics_res[1])
        test_auc = sum(test_metrics_res[3]) * 1.0 / sum(test_metrics_res[1])
        
        train_loss = sum(train_metrics_res[2]) * 1.0 / sum(train_metrics_res[1])
        accuracies = [correct / num for correct,num in zip(test_metrics_res[2],test_metrics_res[1])]
        #about auc, reference:https://zhuanlan.zhihu.com/p/569006692
        auc_collections = [acc / num for acc,num in zip(test_metrics_res[3],test_metrics_res[1])]
        
        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)
        
        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)
        flogger.info('server: avg train loss:{:.3f}'.format(train_loss))
        flogger.info('server: avg test accuracy:{:.3f}'.format(test_acc))
        flogger.info('server: avg test AUC:{:.3f}'.format(test_auc))
        
        flogger.info('server: test accuracy:{:.3f}'.format(np.std(accuracies)))
        flogger.info('server test AUC:{:.3f}'.format(np.std(auc_collections)))
    
    def receive_models(self):
        if len(self.selected_clients) <= 0:
            clogger.exception("selected clients couldn't 0")
        
        self.uploaded_cids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        total_samples = 0
        for client in self.selected_clients:
            try:
                avg_train_time_cost = client.train_time['total_cost'] / client.train_time['rounds']
                avg_send_time_cost = client.send_time['total_cost'] / client.send_time['rounds']
                client_time_cost = avg_train_time_cost + avg_send_time_cost
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost > self.time_threthold:
                continue
            total_samples += client.train_samples
            self.uploaded_cids.append(client.id)
            self.uploaded_weights.append(client.train_samples)
            self.uploaded_models.append(client.model)
        
        for i,w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / total_samples
        