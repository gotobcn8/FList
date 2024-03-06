import os
import numpy as np
import torch

PREFIX_TRAIN = 'train'
PREFIX_TEST = 'test'
SUFFIX_NPZ='.npz'

def read_data(dataset,dir,idx,is_train):
    '''
    dir = absolute directory, 
    dataset is dataset name
    suffix is train or test data name
    idx is the indicator
    '''
    prefix = PREFIX_TRAIN
    if not is_train:
        prefix=PREFIX_TEST
    file_name = str(idx)+SUFFIX_NPZ
    full_file_name = os.path.join(dir,dataset,prefix,file_name)
    # current_file_path = os.path.abspath(__file__)
    # print(current_file_path)
    with open(full_file_name,'rb') as f:
        data = np.load(f,allow_pickle = True)['data'].tolist()
    return data

def read_client_data(dataset,idx,dir,is_train):
    if dataset == 'agnews':
        return read_client_data_text(dataset,idx,dir,is_train)

    if is_train:
        train_data = read_data(dataset, dir,idx, is_train)
        X_train = torch.Tensor(train_data['x']).type(torch.float32)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)

        train_data = [(x, y) for x, y in zip(X_train, y_train)]
        return train_data
    else:
        test_data = read_data(dataset,dir, idx, is_train)
        X_test = torch.Tensor(test_data['x']).type(torch.float32)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)
        test_data = [(x, y) for x, y in zip(X_test, y_test)]
        return test_data

def read_x_data(dataset,idx,dir,is_train):
    # if dataset == 'agnews':
    #     return read_client_data_text(dataset,idx,dir,is_train)

    if is_train:
        train_data = read_data(dataset, dir,idx, is_train)
        # train_x = np.array(train_data['x'])
        train_x = train_data['x']
        for i in range(len(train_x)):
            train_x[i] = torch.tensor(train_x[i])
        # for x in train_x:
        #     print(x.shape)
        # print(train_x.shape)
        return train_x
    else:
        test_data = read_data(dataset,dir, idx, is_train)
        # X_test = torch.Tensor(test_data['x']).type(torch.float32)
        return test_data['x']

def read_client_data_text(dataset,idx,dir,is_train):
    read_data_info = []
    if is_train:
        read_data_info = read_data(dataset,dir,idx,True)
    else:
        read_data_info = read_data(dataset,dir,idx,False)
    X_list,X_list_lens = list(zip(*read_data_info['x']))
    y_list = read_data_info['y']
    
    X_list = torch.Tensor(X_list).type(torch.int64)
    X_list_lens = torch.Tensor(X_list_lens).type(torch.int64)
    y_list = torch.Tensor(read_data_info['y']).type(torch.int64)
    
    predict_data = [((x,lens),y) for x,lens,y in zip(X_list,X_list_lens,y_list)]
    return predict_data

def flatten(source):
    return torch.cat([value.flatten() for value in source.values()])

def copy(target, source):
    for name in target:
        target[name].data = source[name].data.clone()