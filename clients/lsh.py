from .client import ClientBase
from models.optimizer.ditto import PersonalizedGradientDescent
import copy
from algorithm.sim.lsh import ReflectSketch
import utils.data as data
import numpy as np
from torch.utils.data import DataLoader
import time
from fedlog.logbooker import clogger

class LSHClient(ClientBase):
    def __init__(self,args,id,train_samples,test_samples,**kwargs):
        super().__init__(args,id,train_samples,test_samples,**kwargs)
        ditto = args['fedAlgorithm']['lsh']
        self.mu = ditto['mu']
        # self.per_local_steps = ditto['per_local_steps']
        self.model_person = copy.deepcopy(self.model)
        self.optimizer_personl = PersonalizedGradientDescent(
                self.model_person.parameters(), lr=self.learning_rate, mu=self.mu)
    
    def count_sketch(self,hashF):
        self.reflector = ReflectSketch(
            hashF=hashF,
            dtype=float,
            data_vol=hashF.data_volume,
            hash_num = hashF.hash_num,
            dimension=hashF.dimension,
        )
        start_time = time.time
        sketch_data = self.load_train_data(hashF.data_volume)
        for x,_ in sketch_data:
            self.reflector.get_sketch(x,self.device)
        self.sketch = self.reflector.sketch
        self.minisketch = self.reflector.sketch / self.reflector.NumberData
        clogger.info(f'{self.id} :calculate sketch time {time.time - start_time}')
        return self.minisketch
        
    def sketch_data_loader(self,data_volume = 1000):
        sketch_data = data.read_client_data(
            self.dataset,self.id,
            self.dataset_dir,
            is_train = True
        )
        sketch_data = np.random.choice(sketch_data,data_volume,)
        return DataLoader(
            dataset = sketch_data,
            batch_size = data_volume,
            shuffle = True
        )
        
        