from .agnews import generate as generate_agnews
from .mnist import generate as generate_mnist
from .fmnist import generate as generate_fmnist
import os
import ujson
alpha = 0.1
batch_size = 10

def download(name:str,niid,balance,partition,num_clients,num_classes):
    dir_path = name
    if partition == "-":
        partition = None
    if name == 'agnews':
        generate_agnews(dir_path, num_clients, num_classes, niid, balance, partition)
    if name == 'mnist':
        generate_mnist(dir_path,num_clients,num_classes, niid, balance, partition)
    if name == 'fmnist':
        generate_fmnist(dir_path,num_clients,num_classes, niid, balance, partition)