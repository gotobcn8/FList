import copy
import numpy as np
import time
from threading import Thread
import clients.ditto.Ditto as ClientDitto

class Ditto():
    def __init__(self,args):
        # self.set_slow_clients()
        self.set_clients(ClientDitto)
        print(f"\nJoin ratio / total clients: {args.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")
    
    def set_clients()