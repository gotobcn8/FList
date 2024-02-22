import utils.read as read
import entrance
import utils.cmd.parse as parse
from fedlog.loglite import flogger
default_config_path = 'config.yaml'

if __name__ == '__main__':
    parser = parse.get_cmd_parser().parse_args()
    if parser.file == '':
        parser.file = default_config_path
    args = read.yaml_read(parser.file)
    flogger.debug(args)
    entrance.run(args)