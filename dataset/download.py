from .agnews import generate as generate_agnews

def download(name:str,niid,balance,partition,num_clients,num_classes):
    if partition == "-":
        partition = None
    if name == 'agnews':
        dir_path = "agnews/"
        generate_agnews(dir_path, num_clients, num_classes, niid, balance, partition)