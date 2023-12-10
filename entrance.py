from loguru import logger
import models.models as models_sum
import models.transformers
import yaml
import utils.read as yml
import utils.cmd.parse as parse
import models.transformers as transformers
# tianzhen = 'tianzhenaa'
# logger.exception(f"couldn't find {tianzhen} in models")

class Entrance():
    def __init__(self) -> None:
        self.clients = 10
        self.learning_rate = 1e-3
        self.dataset = 'IMDB'
        self.algorithm = 'Ditto'

def run(args):
    #select model
    # etce = Entrance()
    parser = parse.get_cmd_parser().parse_args()
    etce = yml.yaml_read(parser.file)
    
    args = Entrance()
    
    
    args.model = models_select(etce)
    args.algorithm = algorithm_select(etce.algorithm)
    args.server = algorithm_select(etce.algorithm)
    
    
    return

def models_select(etce):
    model_name = etce['model_name']
    if model_name not in dir(models_sum):
        logger.exception(f"couldn't find {model_name} in models")
    parameters = etce[model_name]
    if model_name == 'textcnn':
        logger.info("using textcnn")
        #here should return the nn.Module
        return
    elif model_name == 'transformers':
        return transformers.TransformerModel(ntoken=parameters['vocab_size'],d_model=parameters['emb_dim'],nhead=8,d_hid=parameters['emb_dim'],
                                             nlayers=parameters['nlayers'],num_classes=parameters['num_classes']).to(etce['device'])

def algorithm_select(algorithm:str):
    if algorithm == 'Ditto':
        return