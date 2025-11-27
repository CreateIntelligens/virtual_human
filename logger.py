import logging
import sys
from datetime import datetime
import os

class CustomLogger:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

    def __init__(self, name, log_file='app'):
        self.name = name
        self.log_dir = 'logs'
        self.log_file = log_file
        self.current_log_path = os.path.join(self.log_dir, 'app.log')
        
        # 確保日誌目錄存在
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        # 設置logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # 設置控制台處理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

    def _format_message(self, message, session_id=None):
        return f"[Session {session_id}] {message}" if session_id is not None else message

    def info(self, message, session_id=None):
        self.logger.info(self._format_message(message, session_id))

    def debug(self, message, session_id=None):
        self.logger.debug(self._format_message(message, session_id))

    def warning(self, message, session_id=None):
        self.logger.warning(self._format_message(message, session_id))

    def error(self, message, session_id=None):
        self.logger.error(self._format_message(message, session_id))

    def critical(self, message, session_id=None):
        self.logger.critical(self._format_message(message, session_id))
