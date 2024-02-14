from loguru import logger
import sys
# 配置控制台日志记录器
console_logger = logger.bind(sink=sys.stdout)
console_logger.add(sys.stdout, colorize=True)

# 配置文件日志记录器
file_logger = logger.bind(sink="file.log")
file_logger.add("file.log", rotation="500 MB")

# 配置日志记录器格式化程序
console_logger = console_logger.opt(colors=True, record=True)
file_logger = file_logger.opt(record=True)

# 记录日志
console_logger.info("这条日志将输出到控制台")
file_logger.info("这条日志将写入文件")
