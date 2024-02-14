import os
import numpy as np
import torch

def read_data(dataset,dir,suffix,idx):
    '''
    dir = absolute directory, 
    dataset is dataset name
    suffix is train or test data name
    idx is the indicator
    '''
    file_name = suffix+str(idx)+'.npz'
    full_file_name = os.path.join(dir,dataset,file_name)
    # current_file_path = os.path.abspath(__file__)
    # print(current_file_path)
    with open(full_file_name,'rb') as f:
        data = np.load(f,allow_pickle = True)['data'].tolist()
    return data

def read_client_data(dataset,idx,dir,is_train):
    if dataset == 'agnews':
        return read_client_data_text(dataset,idx,dir,is_train)
    
def read_client_data_text(dataset,idx,dir,is_train):
    read_data_info = []
    if is_train:
        read_data_info = read_data(dataset,dir,'train',idx)
    else:
        read_data_info = read_data(dataset,dir,'test',idx)
    X_list,X_list_lens = list(zip(*read_data_info['x']))
    y_list = read_data_info['y']
    
    X_list = torch.Tensor(X_list).type(torch.int64)
    X_list_lens = torch.Tensor(X_list_lens).type(torch.int64)
    y_list = torch.Tensor(read_data_info['y']).type(torch.int64)
    
    predict_data = [((x,lens),y) for x,lens,y in zip(X_list,X_list_lens,y_list)]
    return predict_data