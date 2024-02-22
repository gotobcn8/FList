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
 
class ClientBase:
    def __init__(self,args,id,train_samples,test_samples,**kwargs):
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
        self.dataset_dir = args['dataset_dir']
        if args['dataset_dir'] == '':
            self.dataset_dir = "repository/"
        
        # self.dataset_dir = os.getcwd()
        self.id = id
        self.train_time = {'rounds':0,'total_cost':0.0}
        self.send_time = {'rounds':0,'total_cost':0.0}
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr = self.learning_rate)
    
    def load_train_data(self,batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        train_data = data.read_client_data(self.dataset,self.id,self.dataset_dir,is_train = True)
        return DataLoader(train_data,batch_size=self.batch_size,drop_last=True, shuffle=True)
    
    def load_test_data(self,batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        test_data = data.read_client_data(self.dataset,self.id,self.dataset_dir,is_train=False)
        return DataLoader(test_data,batch_size = self.batch_size, drop_last=False, shuffle=True)
    
    def train_model(self):
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

    def test_model(self):
        testloader = self.load_test_data()
        logger.info(len(testloader))
        self.model.eval()
        test_accuracy = 0
        test_num = 0
        y_prob = []
        y_true = []

        with torch.no_grad():
            for x,y in testloader:
                x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                
                test_accuracy += (torch.sum(torch.argmax(output,dim=1) == y)).item()
                test_sum += y.shape[0]
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