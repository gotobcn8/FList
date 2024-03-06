import copy
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from loguru import logger
from sklearn import metrics
import os
import utils.data as data
import torch.nn.functional as F
import time
import const.constants as const
 
class ClientBase:
    def __init__(self,args,id,train_samples,test_samples,serial_id,**kwargs):
        '''**kwargs是Python中的一种语法,它允许函数接收任意数量的关键字参数。
        这些参数在函数内部作为字典（dictionary）处理，字典的键是参数名，值是传递给函数的参数值。在Python中,
        kwargs是一个通用的名字，代表“keyword arguments”，但你也可以使用其他任何名字。
        '''
        self.model = copy.deepcopy(args['model'])
        self.algorithm = args['algorithm']
        self.dataset = args['dataset']
        self.device = args['device']
        
        self.save_dir = os.path.join(os.getcwd(),args['save_dir'],'ditto')
        self.num_classes = args['num_classes']
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args['batch_size']
        self.learning_rate = args['learning_rate']
        self.epochs = args['epochs']
        self.local_epochs = args['epochs']
        # self.dataset = args.dataset
        # self.device = args.device
        
        # self.save_dir = args.save_dir
        # self.num_classes = args.num_classes
        # self.train_samples = train_samples
        # self.test_samples = test_samples
        # self.batch_size = args.batch_size
        # self.learning_rate = args.learning_rate
        # self.epochs = args.epochs
        # self.local_epochs = args.epochs
        self.dataset_dir = const.DIR_DEPOSITORY
        if 'dataset_dir' in args.keys():
            self.dataset_dir = args['dataset_dir']
        
        # self.dataset_dir = os.getcwd()
        self.id = id
        self.serial_id = serial_id
        self.train_time = {'rounds':0,'total_cost':0.0}
        self.send_time = {'rounds':0,'total_cost':0.0}
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr = self.learning_rate)
        #temporary setting
        self.train_slow = False
        self.late = False
    
    def load_train_data(self,batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        train_data = data.read_client_data(self.dataset,self.serial_id,self.dataset_dir,is_train = True)
        return DataLoader(train_data,batch_size=self.batch_size,drop_last=True, shuffle=True)
    
    def load_test_data(self,batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        test_data = data.read_client_data(self.dataset,self.serial_id,self.dataset_dir,is_train=False)
        return DataLoader(test_data,batch_size = self.batch_size, drop_last=False, shuffle=True)
    
    def train_metrics(self):
        train_loader = self.load_train_data()
        
        self.model.eval()
        train_num = 0
        losses = 0
        with torch.no_grad():
            for x,y in train_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                y_num = y.shape[0]
                output = self.model(x)
                loss = self.loss(output,y)
                train_num += y_num
                losses += loss.item() * y_num
                
        return losses,train_num  

    def test_metrics(self):
        testloader = self.load_test_data()
        # self.model.to(self.device)
        self.model.eval()
        test_accuracy = 0
        test_num = 0
        y_prob = []
        y_true = []

        with torch.no_grad():
            for x,y in testloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                
                test_accuracy += (torch.sum(torch.argmax(output,dim=1) == y)).item()
                test_num += y.shape[0]
                #将 PyTorch tensor output 从计算图中分离并移动到 CPU，然后将其转换为 NumPy 数组。
                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    #如果只有两个标签，我们仍然采用多分类训练？
                    nc += 1
                label = label_binarize(y.detach().cpu().numpy(),classes=np.arange(nc))
                if self.num_classes == 2:
                    label = label[:,:2]
                y_true.append(label)
        
        y_prob = np.concatenate(y_prob,axis = 0)
        y_true = np.concatenate(y_true,axis=0)
        
        
        score = metrics.roc_auc_score(y_true,y_prob,average='micro')
        
        return test_accuracy,test_num,score
    
    def save_item(self,item,item_name,item_path = None):
        if item_path == None:
            item_path = self.save_dir
        if not os.path.exists(item_path):
            os.makedirs(item_path)
        torch.save(item,os.path.join(item_path,'client_'+str(self.id)+'_'+item_name+'.pt'))
        
    def load_item(self,item_name,item_path=None):
        if item_path == None:
            item_path = self.save_dir
        return torch.load(os.path.join(item_path,'client_'+str(self.id)+'_'+item_name+'.pt'))
    
    def train(self):
        trainloader = self.load_train_data()
        # self.model.to(self.device)
        self.model.train()
        
        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for _ in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # self.model.cpu()

        # if self.learning_rate_decay:
        #     self.learning_rate_scheduler.step()

        self.train_time['rounds'] += 1
        self.train_time['total_cost'] += time.time() - start_time
    
    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()
    
    def test_other_model(self,other_personalized_model):
        testloaderfull = self.load_test_data()
      
        other_personalized_model.eval()
      
        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        
        with torch.no_grad():
            for x,y in testloaderfull:
                if isinstance(x, list):
                    x[0] = x[0].to(self.device)
                    x = x[0]
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = other_personalized_model(x)
                
                test_acc += (torch.sum(torch.argmax(output,dim=1) == y)).item()
                test_num += y.shape[0]
                
                y_prob.append(F.softmax(output,dim=1).detach().cpu().numpy())
                y_true.append(label_binarize(y.detach().cpu().numpy(),classes = np.arange(self.num_classes)))
        
        y_prob = np.concatenate(y_prob,axis=0)
        y_true = np.concatenate(y_true,axis=0)
        
        auc = metrics.roc_auc_score(y_true,y_prob,average='micro')
        return test_acc,test_num,auc