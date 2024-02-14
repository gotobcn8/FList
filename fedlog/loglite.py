from loguru import logger
import sys
from datetime import datetime

# flogger : Logger
# clogger : Logger
# 创建一个函数来生成包含当前日期的日志文件名  
def get_log_file_name()->str:  
    now = datetime.now()  
    timestamp = now.strftime("%Y-%m-%d")  
    return "logfile/fedlog_"+timestamp+".log"  

clogger = logger.bind(sink = sys.stdout)
clogger.add(sys.stdout, colorize=True)

flogger = logger.bind(sink = 'file.log')
flogger.add(sink = 'file.log',rotation='100 MB',compression='zip')