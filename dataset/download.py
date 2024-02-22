from .agnews import generate as generate_agnews
from .mnist import generate as generate_mnist
import os
import ujson
alpha = 0.1
batch_size = 10

def download(name:str,niid,balance,partition,num_clients,num_classes):
    if partition == "-":
        partition = None
    if name == 'agnews':
        dir_path = "agnews/"
        generate_agnews(dir_path, num_clients, num_classes, niid, balance, partition)
    if name == 'mnist':
        dir_path = 'mnist/'
        generate_mnist(dir_path,num_clients,num_classes, niid, balance, partition)
