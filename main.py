import utils.read as read
import entrance
from fedlog.loglite import flogger
default_config_path = 'config.yaml'

if __name__ == '__main__':
    args = read.yaml_read(default_config_path)
    flogger.debug(args)
    entrance.run(args)