from .serverbase import Server
from .ditto import Ditto
from .lsh import LSHServer
from .dittolsh import LSHDittoServer
from .cfl import ClusterFL


def get_server(name:str,args)->Server:
    if name == 'Ditto':
        return Ditto(args)
    elif name == 'ditto_lsh':
        return LSHDittoServer(args)
    elif 'lsh' in name:
        return LSHServer(args)
    elif name == 'cfl':
        return ClusterFL(args)