import sys
import logging 


# if sys is not there

def error_message_detail(error, error_detail:sys):
    _, _, exc_tb = error_detail.exc_info()
    filename = exc_tb.tb_frame.f_code.co_filename

    #this exc_tb gives info about the error and file, line where the error has occured

    error_message = 'Error occured in python script name [{0}] line number [{1}] error message [{2}]'.format(
        filename, exc_tb.tb_lineno, str(error)
    )

    return error_message


#custom exception class inheriting from Exception
class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail = error_detail)

    def __str__(self):
        return self.error_message



if __name__ =="__main__":
    try:
        a = 1/0
    except Exception as e:
        logging.info("Division by Zero")
        raise CustomException(e, sys)