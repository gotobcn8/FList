import os
import logbook
from logbook import Logger, TimedRotatingFileHandler,NOTSET
from logbook.more import ColorizedStderrHandler

route = 'fedlog/logfiles/client.log'

def logType(record,handler):
    log = "[{date}] [{level}] [{filename}:{func_name}:{lineno}] [identifier] {msg}".format(
        date=record.time,  # 日志时间
        level=record.level_name,  # 日志等级
        filename=os.path.split(record.filename)[-1],  # 文件名
        func_name=record.func_name,  # 函数名
        lineno=record.lineno,  # 行号
        identifier = record.extra['identifier'],
        msg=record.message  # 日志内容
    )
    return log



class Attender(Logger):
    def __init__(self,index='client',filePath = route,name= 'client',level = NOTSET,handlers=None) -> None:
        super().__init__(name,level)
        self.type = type
        #init log handler
        # self.handlers = []
        self.identifier = index
        chandler = ColorizedStderrHandler(bubble=True)
        fhandler = TimedRotatingFileHandler(
            filename=filePath
        )
        # chandler.format_string
        chandler.formatter = logType
        fhandler.formatter = logType
        if handlers is None:
            self.handlers=[chandler,fhandler]
        else:
           self.handlers.extend(handlers)

clogger = Attender(index='client')
slogger = Attender(index='server')
errlogger = Attender()

if __name__ == '__main__':
    chandler = ColorizedStderrHandler()
    fhandler = TimedRotatingFileHandler(
        filename='fedlog/logfiles/client.log'
    )
    clogger = Attender(handlers=[chandler,fhandler])
    clogger.info('this is a info log')
    clogger.warn('this is a warn log')