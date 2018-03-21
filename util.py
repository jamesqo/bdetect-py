import logging as log
import os

from inspect import stack

def exec_and_check(cmd):
     exit_code = os.system(cmd)
     if exit_code != 0:
         raise RuntimeError("'{}' failed with exit code {}".format(cmd, exit_code))

def log_mcall(level=log.DEBUG):
    method = stack()[1].function
    log.log(level, "%s() called", method)
