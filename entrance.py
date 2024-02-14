from loguru import logger
# import models.models as models_sum
import utils.read as yml
import utils.cmd.parse as parse
import models.transformers as transformers
# from servers.serverapi import get_server
import servers.serverapi as sapi
import torch
from dispatch import Dispatcher
from fedlog.loglite import clogger
from fedlog.loglite import flogger
from dataset import download
import os
# tianzhen = 'tianzhenaa'
# logger.exception(f"couldn't find {tianzhen} in models")

def run(args):
    #select model
    parser = parse.get_cmd_parser().parse_args()
    fparser = yml.yaml_read(parser.file)
    
    #cuda device
    if args['device'] == 'gpu':
        if not torch.cuda.is_available():
            args['device'] = 'cpu'
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args['device_id'])
            args['device'] = 'cuda'
    # download.  
    #initialize a dispatcher
    dispatcher = Dispatcher()
    flogger.trace('starting running')
    dispatcher.model = models_select(args)
    args['model'] = dispatcher.model
    dispatcher.server = algorithm_select(args['algorithm'],args)
    dispatcher.server.train()

def models_select(args):
    model_name = args['model_name']
    clogger.info(model_name)
    parameters = args['models'][model_name]
    clogger.info(model_name)
    dataset = args['dataset']
    dtsparameter = args[dataset]
    args['num_classes'] = parameters['num_classes']
    args['batch_size'] = parameters['batch_size']
    download.download(dataset,dtsparameter['niid'],dtsparameter['balance'],
                      partition=dtsparameter['partition'],
                      num_clients=args['num_clients'],
                      num_classes=parameters['num_classes'])
    if model_name == 'textcnn':
        clogger.info("using textcnn")
        #here should return the nn.Module
        return
    elif model_name == 'transformers':
        return transformers.TransformerModel(ntoken=parameters['vocab_size'],d_model=parameters['embadding_dim'],nhead=8,d_hid=parameters['embadding_dim'],
                                             nlayers=parameters['nlayers'],num_classes=parameters['num_classes']).to(args['device'])
    else:
        clogger.exception(f"couldn't find {model_name} in models")

def algorithm_select(algorithm:str,args):
    return sapi.get_server(algorithm,args)