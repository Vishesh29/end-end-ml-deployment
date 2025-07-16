import sys
from src.logger import logging

def error_msg_detail(error,error_detail:sys):
    _,_,exc_tb = error_detail.exc_info()
    filename = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_msg = "Error occurred in script name [{0}] at line number [{1}] with message [{2}]".format(
        filename,
        line_number,
        str(error)
    )
    return error_msg

class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_msg_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message

    def __repr__(self):
        return CustomException.__name__.str() + " " + self.error_message