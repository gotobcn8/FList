import copy
import numpy as np
import time
from clients.ditto import Ditto as ClientDitto
from .serverbase import Server
from .ditto import Ditto
from threading import Thread
from loguru import logger
import torch.nn.functional as F
import utils.dlg as dlg
from algorithm.sim.lsh import SignRandomProjections
from clients.lsh import LSHClient as ClientLshash
import time
from fedlog.logbooker import slogger
from utils.data import read_client_data

class LSHServer(Server):
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
        self.sketches = dict()
        self.set_clients(ClientLshash)
    
    def set_clients(self,clientObj):
        for i in range(self.num_clients):
            train_data = read_client_data(self.dataset,i,self.dataset_dir,is_train=True)
            test_data = read_client_data(self.dataset,i,self.dataset_dir,is_train=False)
            client = clientObj(
                self.args,id = i,
                train_samples = len(train_data),
                test_samples = len(test_data),
            )
            self.clients.append(client)
        #first we need to calculate the sketch for all clients.
        start_time = time.time()
        for client in self.clients:
            client.count_sketch(self.hashF)
            #  = self.hashF.hash(client)
            self.sketches[client.id] = client.minisketch
        
        slogger.info('server :calculating time {:.3f}s'.format(time.time() - start_time))
        