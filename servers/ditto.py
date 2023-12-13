import copy
import numpy as np
import time
from clients.ditto import Ditto as ClientDitto
from serverbase import Server
from threading import Thread
from loguru import logger
import os
import torch
import torch.nn.functional as F
import math

LOG_PATH = 'log'
flogger = logger.bind(sink=os.path.join(LOG_PATH,'server_ditto.log'))

class Ditto(Server):
    def __init__(self, args, times) -> None:
        super().__init__(args, times)
        
        self.set_clients(ClientDitto)
        flogger.info(f"join ratio total clients:{self.join_ratio / self.num_clients}")
        flogger.info('Finished creating server and clients')

        self.budget = []
    
    def train(self):
        for i in range(self.global_rounds + 1):
            start_time = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()
            
            if i % self.eval_gap == 0:
                flogger.debug(f"-------------Round number: {i}-------------")
                flogger.debug("start evaluating model")
                self.evaluate()
                flogger.debug('Evaluating personalized models')
                self.evaluate_personalized()

            #select different clients each round
            for client in self.selected_clients:
                client.pretrain()
                client.train()
                
            self.receive_models()
            if self.dlg_eval and i % self.dlg_gap == 0:
                self.call_dlg(i)
            
            self.aggregate_parameters()
    
    def evaluate_personalized(self,accuracies = None,loss = None):
        
        return
    
    def train_metrics(self):
        
    
    def test_metrics_personalized(self,all_clientsaccuracies = None,):
        # return super().test_metrics()
        num_samples = []
        total_corrects = []
        total_auc = []
        ids = [0] * len(self.new_clients)
        for i,c in enumerate(self.new_clients):
            c_corrects,c_num_samples,c_auc = c.test_metrics()
            total_corrects.append(c_corrects*1.0)
            total_auc.append(c_auc * c_num_samples)
            num_samples.append(c_num_samples)
            ids[i] = c.id
        
        return ids,num_samples,total_corrects,total_auc 
    
    def call_dlg(self,id):
        cnt = 0
        psnr_val = 0
        for cid,client_model in zip(self.uploaded_ids,self.uploaded_models):
            client_model.eval()
            origin_grad = []
            for gp, pp in zip(self.global_model.parameters(),client_model.parameters()):
                origin_grad.append(gp.data - pp.data)
            
            target_inputs = []
            train_loader = self.clients[cid].load_train_data()
            with torch.no_grad():
                for i,(x,y) in enumerate(train_loader):
                    x = x.to(self.device)
                    # y = y.to(self.device)
                    output = client_model(x)
                    target_inputs.append((x,output))
            
            d = DLG(client_model,origin_grad,target_inputs)
        
        def DLG(net,origin_grad,target_inputs):
            criterion = torch.nn.MSELoss()
            cnt = 0
            psnr_val = 0
            for idx,(gt_data,gt_out) in enumerate(target_inputs):
                #generate tensor like gt_data, and open grad computing
                dummy_data = torch.randn_like(gt_data,requires_grad = True)
                dummy_out = torch.randn_like(gt_out,requires_grad=True)
                #
                optimizer = torch.optim.LBFGS([dummy_data,dummy_out])
                history = [gt_data.data.cpu().numpy(),F.sigmoid(dummy_data).data.cpu().numpy()]
                for iters in range(100):
                    def closure():
                        optimizer.zero_grad()
                        dummy_pred = net(F.sigmoid(dummy_data))
                        dummy_loss = criterion(dummy_pred,dummy_out)
                        dummy_grad = torch.autograd.grad(dummy_loss,net.parameters,create_graph=True)
                        
                        grad_diff = 0
                        for gx,gy in zip(dummy_grad,origin_grad):
                            grad_diff += ((gx-gy) ** 2).sum()
                        grad_diff.backward()
                        
                        return grad_diff
                    optimizer.step(closure)
                history.append(F.sigmoid(dummy_data).data.cpu().numpy())
                p = psnr(history[0],history[2])
                if not math.isnan(p):