import utils.read as read
import pytest
from loguru import logger

def test_read_yaml():
    name = 'config.yaml'
    res = read.yaml_read(name)
    print(res)

def test_case01():
    print('testing')

if __name__ == '__main__':
    test_case01()
    test_read_yaml()