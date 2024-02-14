
from loguru import logger
import sys
clogger = logger.bind(sink = sys.stdout)
clogger.add(sys.stdout, colorize=True)

flogger = logger.bind(sink = 'logfile')
flogger.add('fed_log_{time}.log',rotation='100 MB',compression='zip')