from .serverbase import Server
from .ditto import Ditto

def get_server(name:str,args)->Server:
    if name == 'Ditto':
        return Ditto(args)