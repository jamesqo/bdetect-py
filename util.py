import logging as log

from inspect import stack

def log_mcall(level=log.DEBUG):
    method = stack()[1].function
    log.log(level, "%s() called", method)
