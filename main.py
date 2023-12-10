import utils.read as read
import entrance.run as run
default_config_path = 'config.yaml'

if __name__ == '__main__':
    args = read.yaml_read(default_config_path)
    run(args)