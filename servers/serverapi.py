from .serverbase import Server
from .ditto import Ditto
from .lsh import LSHServer

def get_server(name:str,args)->Server:
    if name == 'Ditto':
        return Ditto(args)
    elif 'lsh' in name:
        return LSHServer(args)