#############################################################
############ /Modules/Logger.py
#############################################################
import logging
import os
import sys

#############################################################
################ Function
#############################################################
# allows to return in the stderr the different user messages (warning, info, debug, error)
# takes as input the name of the function using this module.
def get_module_logger(mod_name):
    # set up logger
    logger=logging.getLogger(os.path.basename(mod_name))
    logger.setLevel(logging.DEBUG)
    # create console handler for STDERR
    stderr = logging.StreamHandler(sys.stderr)
    stderr.setLevel(logging.DEBUG)
    #create formatter
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s %(funcName)s(): %(message)s',
                                    '%Y-%m-%d %H:%M:%S')
    #add formatter to stderr handler
    stderr.setFormatter(formatter)
    if not logger.handlers:
        #add stderr handler to logger
        logger.addHandler(stderr)
    return(logger)