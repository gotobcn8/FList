from loguru import logger
import sys
clogger = logger.bind(sink = sys.stdout)
clogger.add(sys.stdout, colorize=True)

flogger = logger.bind(sink = '../fedlog/logfile/fed_log_{time}.log')
flogger.add(sink = '../fedlog/logfile/fed_log_{}.log',rotation='100 MB',compression='zip')