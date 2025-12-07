import base64
import logging
import os
import sys

class Logger:
    def __init__(self, name, level=logging.DEBUG):
        self.name = name
        self.level = level
        self.init_logging()

    def init_logging(self):
        # Setup logging config
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(self.level)
        if self.logger.handlers:
            return
        self.file_handler = logging.FileHandler(
            filename=f"./save/log/logging_{self.name}.log",
            encoding="utf-8",
            mode="a"
        )
        self.console_handler = logging.StreamHandler()
        self.logger.addHandler(self.console_handler)
        self.logger.addHandler(self.file_handler)
        
        formatter = logging.Formatter(
           "{asctime} - {levelname} - {message}",
            style="{",
            datefmt="%Y-%m-%d %H:%M",
        )

        self.console_handler.setLevel(self.level)
        self.console_handler.setFormatter(formatter)
    
    def LOG_ERROR(self, messages):
        self.logger.error(messages, exc_info=True)


    def LOG_INFO(self, messages):
        self.logger.info(messages)


    def LOG_WARNING(self, messages):
        self.logger.warning(messages)


    def LOG_DEBUG(self, messages):
        self.logger.debug(messages)