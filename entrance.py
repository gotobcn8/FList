from loguru import logger
import models.models
import models.transformers
# tianzhen = 'tianzhenaa'
# logger.exception(f"couldn't find {tianzhen} in models")

class Entrance():
    def __init__(self) -> None:
        self.model_name = 'cnn'
        self.clients = 10
        self.learning_rate = 1e-3
        self.dataset = 'IMDB'
        self.algorithm = 'Ditto'
        
def run(args):
    #select model
    etce = Entrance()
    args.model = models_select(etce.model_name)
    args.algorithm = algorithm_select(etce.algorithm)
    args.server = algorithm_select(etce.algorithm)
    return

def models_select(args):
    model_name = args.model_name
    if model_name not in dir(models):
        logger.exception(f"couldn't find {model_name} in models")
    if model_name == 'textcnn':
        logger.info("using textcnn")
        #here should return the nn.Module
        return
    elif model_name == 'transformers':
        return transformers.TransformerModel(ntoken=vocab_size,d_model=emb_dim,nhead=8,d_hid=emb_dim,nlayers=2,num_classes=args.num_classes).to()
    
def algorithm_select(algorithm:str):
    if algorithm == 'Ditto':
        return