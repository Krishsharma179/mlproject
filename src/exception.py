import sys


def  error_message_detail(error,error_detail:sys):
   c,y,detail=error_detail.exc_info()
   file_name=detail.tb_frame.f_code.co_filename
   error_message="Error occured in python script [{0}] line number [{1}] error message[{2}]".format(file_name,detail.tb_lineno,str(error))

   return error_message
    
class CustomException(Exception):
   def __init__(self,error_message,error_detail:sys):
      super().__init__(error_message)
      self.error_message=error_message_detail(error_message,error_detail=error_detail)

   def __str__(self):
      return self.error_message
       