import copy
import numpy as np
import time
from clients.ditto import Ditto as ClientDitto
from .serverbase import Server
from .ditto import Ditto
from threading import Thread
import torch.nn.functional as F
import utils.dlg as dlg
from algorithm.sim.lsh import SignRandomProjections
from clients.lshditto import LSHDittoClient as ClientLshash
import time
from fedlog.logbooker import slogger
from utils.data import read_client_data
from sklearn.cluster import KMeans
from cluster.clusterbase import ClusterBase
import torch

class LSHDittoServer(Server):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.fedAlgorithm = args['fedAlgorithm']['lsh']
        self.data_volume = self.fedAlgorithm['data_volume']
        self.hashF = SignRandomProjections(
            each_hash_num=self.fedAlgorithm['hash_num'],
            data_volume=self.data_volume,
            data_dimension=self.fedAlgorithm['cv_dim'],
            random_seed=args['random_seed']
        )
        self.fine_tuning_epoch = args['fine_tuning_epoch']
        self.sketches = dict()
        self.clients_ids_map = dict()
        self.set_for_lsh_clients(ClientLshash)
        self.num_clusters = args['cluster']['cluster_num']
        self.pre_cluster(self.num_clusters)
        self.set_clusters(ClusterBase,args)

    def set_clusters(self,clusterObj,args):
        self.clusters = []
        for i,cluster in enumerate(self.cluster_ids):
            self.clusters.append(clusterObj(i,cluster,self.cluster_map_clients[i],args['model']))
    
    def set_for_lsh_clients(self,clientObj):
        self.set_clients(clientObj)
        self.set_late_clients(clientObj)
        #first we need to calculate the sketch for all clients.
        start_time = time.time()
        for i,client in enumerate(self.all_clients):
            client.count_sketch(self.hashF)
            #  = self.hashF.hash(client)
            self.sketches[client.id] = client.minisketch
            self.clients_ids_map[client.id] = i
        slogger.info('total calculating time {:.3f}s'.format(time.time() - start_time))
            
    def train(self):
        slogger.debug('server','starting to train')
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i%self.eval_gap == 0:
                slogger.debug(f"-------------Round number: {i}-------------")
                slogger.debug("start evaluating model")
                self.evaluate()
                slogger.debug('Evaluating personalized models')
                self.evaluate_personalized()

            for client in self.selected_clients:
                client.train()
                client.train_personalized()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            self.cluster_receive_models()
            # if self.dlg_eval and i%self.dlg_gap == 0:
            #     self.call_dlg(i)
            self.aggregate_parameters()
            self.cluster_aggregate_parameters()
            
            self.budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.budget[-1])

            # if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
            #     break
            if self.new_clients_settings['enabled'] and self.new_clients_joining_round >= i:
                if len(self.late_clients) < self.num_join_each_round:
                    new_attend_clients = self.late_clients
                else:
                    new_attend_clients = self.late_clients[:self.num_join_each_round]
                    self.late_clients = self.late_clients[self.num_join_each_round:]
                #it need to be fine-tuned before attending
                self.fine_tuning_new_clients(new_attend_clients)
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.budget[1:])/len(self.budget[1:]))

        self.save_results()
        self.save_global_model()

        # if self.num_new_clients > 0:
        #     self.eval_new_clients = True
        #     self.set_new_clients(clientAVG)
        #     print(f"\n-------------Fine tuning round-------------")
        #     print("\nEvaluate new clients")
        #     self.evaluate()
    
    # def select_clients(self):
    #     selected_clients = []
    def cluster_aggregate_parameters(self):
        for cluster in self.clusters:
            cluster.aggregate_parameters()

    
    def cluster_receive_models(self):
        self.cluster_attend_clients = dict()
        self.cluster
        for attend_client in self.select_clients:
            cluster_id = self.clients_map_clusters[attend_client.id]
            if cluster_id not in self.cluster_attend_clients:
                self.cluster_attend_clients = []
            #attender serial id in this round
            self.cluster_attend_clients[cluster_id].append(attend_client)
        
        for cluster in self.clusters:
            if cluster.id in self.cluster_attend_clients.keys():
                cluster.receive_models(self.cluster_attend_clients[cluster.id])
    
    def pre_cluster(self,cluster_num):
        sketches1dim = []
        cluster_start_time = time.time()
        for client in self.all_clients:
            sketch = client.minisketch
            sketches1dim.append(sketch.reshape(1,-1)[0]) 
        kmeans = KMeans(n_clusters=cluster_num)
        #fit the kmeans with the one-dimensional data to cluster
        kmeans_res = kmeans.fit(sketches1dim)
        
        self.clients_map_clusters = dict()
        '''
        from the cluster form a map to the {client_id:cluster_id}
        from the cluster form a map to the {cluster id: [clients_obj]}
        '''
        self.cluster_map_clients = [[] for _ in range(self.num_clusters)]
        self.cluster_ids = [[] for _ in range(self.num_clusters)]
        for client_sid,cluster_id in enumerate(kmeans_res.labels_):
            self.clients_map_clusters[self.all_clients[client_sid].id] = cluster_id
            self.cluster_map_clients[cluster_id].append(self.all_clients[client_sid])
            self.cluster_ids[cluster_id].append(client_sid)
        slogger.info('server,cluster time:{:.3f}s'.format(time.time() - cluster_start_time))
        print('-'*25)
        for i,c in enumerate(self.cluster_ids):
            print('cluster {}:{}'.format(i,c))
            print('-'*25)
    
    def fine_tuning_new_clients(self,new_clients):
        for new_client in new_clients:
            which_cluster = self.clients_map_clusters[new_client.id]
            new_client.set_parameters(which_cluster.model)
            optimizer = torch.optim.SGD(new_client.model.parameters(),lr = self.learning_rate)
            lossFunc = torch.nn.CrossEntropyLoss()
            train_loader = new_client.load_train_data()
            new_client.model.train()
            for _ in range(self.fine_tuning_epoch):
                for i,(x,y) in enumerate(train_loader):
                    if isinstance(x,list):
                        x[0] = x[0].to(new_client.device)
                        x = x[0]
                    else:
                        x = x.to(new_client.device)
                    y = y.to(new_client.device)
                    output = new_client.model(x)
                    loss =  lossFunc(output,y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            new_client.model_person = copy.deepcopy(new_client.model)
    
    def evaluate_personalized(self,acc=None,loss=None):
        test_metrics_res = self.test_metrics_personalized()
        train_metrics_res = self.train_metrics_personalized()
        
        test_acc = sum(test_metrics_res[2]) * 1.0 / sum(test_metrics_res[1])
        test_auc = sum(test_metrics_res[3]) * 1.0 / sum(test_metrics_res[1])
        
        train_loss = sum(train_metrics_res[2]) * 1.0 / sum(train_metrics_res[1])
        accuracies = [correct / num for correct,num in zip(test_metrics_res[2],test_metrics_res[1])]
        #about auc, reference:https://zhuanlan.zhihu.com/p/569006692
        auc_collections = [acc / num for acc,num in zip(test_metrics_res[3],test_metrics_res[1])]
        
        if accuracies == None:
            self.rs_test_acc.append(test_acc)
        else:
            accuracies.append(test_acc)
        
        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)
        slogger.info('server: avg train loss:{:.3f}'.format(train_loss))
        slogger.info('server: avg test accuracy:{:.3f}'.format(test_acc))
        slogger.info('server: avg test AUC:{:.3f}'.format(test_auc))
        
        slogger.info('std: test accuracy:{:.3f}'.format(np.std(accuracies)))
        slogger.info('std test AUC:{:.3f}'.format(np.std(auc_collections)))
    
    def test_metrics_personalized(self):
        # return super().test_metrics()
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()
        num_samples = []
        total_corrects = []
        total_auc = []
        ids = [0] * len(self.clients)
        for i,c in enumerate(self.clients):
            if c.id.startswith('late'):
                self.clusters[self.clients_map_clusters[c.id]].generalized()
                continue
            c_corrects,c_num_samples,c_auc = c.test_metrics()
            total_corrects.append(c_corrects*1.0)
            total_auc.append(c_auc * c_num_samples)
            num_samples.append(c_num_samples)
            ids[i] = c.id
        
        return ids,num_samples,total_corrects,total_auc
    
    def train_metrics_personalized(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            return [0],[1],[0]
        num_samples = []
        losses = []
        for c in self.clients:
            cl,ns = c.train_personalized_with_metrics()
            num_samples.append(ns)
            losses.append(cl * 1.0)
        ids = [c.id for c in self.clients]
        
        return ids,num_samples,losses