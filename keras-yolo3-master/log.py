import os
import logging
import time

def output_log():
        # 项目路径
        prj_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件的上一级的上一级目录（增加一级）
        # 在项目路径下创建一个log文件夹, 拼接成路径格式
        log_path = os.path.join(prj_path, 'log')
        # 在log文件夹下再创建一个以当前日期命名的文件夹
        log_date_path = os.path.join(log_path, time.strftime('%Y%m%d', time.localtime(time.time())))
        current_time = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))  # 返回当前时间
        # 在时间文件夹下创建一个文件，后缀是.log.
        log_name = os.path.join(log_date_path, current_time + '.log')
        isExists = os.path.exists(log_date_path)  # 判断该目录是否存在
        print(prj_path, log_path, log_date_path, log_name)
        # 创建一个logger(初始化logger)
        log1 = logging.getLogger()
        log1.setLevel(logging.DEBUG)
        if not isExists:
            os.makedirs(log_date_path)
            print(log_path + log_date_path + "目录创建成功")
        else:
            # 如果目录存在则不创建，并提示目录已存在
            print(log_path + "目录 %s 已存在" % log_date_path)
        try:
            # 创建一个handler，用于写入日志文件
            current_time = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))  # 返回当前时间
            log_name = os.path.join(log_date_path, current_time + '.log')

            fh = logging.FileHandler(log_name)
            fh.setLevel(logging.INFO)

            # 再创建一个handler，用于输出到控制台
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)

            # 定义handler的输出格式
            formatter = logging.Formatter('[%(asctime)s] - %(name)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)
            # 给logger添加handler
            log1.addHandler(fh)
            log1.addHandler(ch)
        except Exception as e:
            print ( "输出日志失败！ %s" % e )
 import sys



    class Logger ( object ):

        def __init__(self , filename="Default.log"):
            self.terminal = sys.stdout

            self.log = open ( filename , "a" )

        def write(self , message):
            self.terminal.write ( message )

            self.log.write ( message )

        def flush(self):
            pass

    sys.stdout = Logger ( 'log.txt' )