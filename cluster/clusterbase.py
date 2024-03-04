import copy
from fedlog.logbooker import clogger

class ClusterBase():
    def __init__(self,id,cluster_ids,clients,model) -> None:
        self.id = id
        self.clients = clients
        self.cluster_ids = cluster_ids
        self.cluster_model = copy.deepcopy(model)
        self.cluster_samples = 0
        self.get_cluster_sampels()
    
    def get_cluster_sampels(self):
        for client in self.clients:
            self.cluster_samples += client.train_samples
    
    def aggregate_parameters(self):
        self.cluster_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.cluster_model.parameters():
            param.data.zero_()
        
        for w,client_model in zip(self.uploaded_weights,self.uploaded_models):
            self.add_parameters(w,client_model)
    
    def add_parameters(self,w,client_model):
        for server_param,client_param in zip(self.cluster_model.parameters(),client_model.parameters()):
            server_param.data += client_param.data.clone() * w
    
    def receive_models(self,selected_clients):
        if len(selected_clients) <= 0:
            clogger.warn("the cluster clients selected is 0")
            return
        
        self.uploaded_cids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        total_samples = 0
        for client in selected_clients:
            try:
                avg_train_time_cost = client.train_time['total_cost'] / client.train_time['rounds']
                avg_send_time_cost = client.send_time['total_cost'] / client.send_time['rounds']
                client_time_cost = avg_train_time_cost + avg_send_time_cost
            except ZeroDivisionError:
                client_time_cost = 0
            # if client_time_cost > self.time_threthold:
            #     continue
            total_samples += client.train_samples
            self.uploaded_cids.append(client.id)
            self.uploaded_weights.append(client.train_samples)
            self.uploaded_models.append(client.model)
        
        for i,w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / total_samples
    
    def test_personal_model_generalized(self,test_client):
        '''
        This function is used to test the personalized model generalized ability in cluster. 
        '''
        clogger.debug('Starting to evaluate the generalization of cluster {}'.format(self.id))
        avg_test_acc,total_test_num,avg_auc = 0,0,0
        if len(self.clients) <= 1:
            clogger.warn('test_client:{}, cluster clients less than 1,can not test the generallization'.format(test_client.id))
            return
        if test_client.train_time['rounds'] == 0:
            clogger.info('this is first time to test the generalization of {}'.format(test_client.id))
        for client in self.clients:
            if test_client.id == client.id:
                continue
            test_acc,test_num,auc = client.test_other_personalized_model(test_client.model)
            clogger.info('cluster {} new coming client {} be tested client {}'.format(self.id,test_client.id,client.id))
            clogger.info('test_accuracy:{}%% test_num:{} test_auc:{}'.format(test_acc*100.0/test_num,test_num,auc))
            total_test_num += test_num
            avg_test_acc += test_acc
            avg_auc += auc*test_num
        
        # if avg_test_acc == 0:
        avg_test_acc =  (avg_test_acc * 1.0) / total_test_num 
        avg_auc =  (avg_auc * 1.0) / total_test_num
        clogger.info('-------avg_acc:{:.3f}%% avg_auc:{:.3f}'.format(avg_test_acc*100,avg_auc))
        