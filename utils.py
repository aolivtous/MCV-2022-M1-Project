from main import *

def str_to_bool(v):
    """
    It takes a string and its boolean value
    
    :param v: The value evaluate to a boolean
    :return: A boolean value.
    """
    return str(v).lower() in ("yes", "true", "t", "1")

def refine_results(results):
    new_list = []
    for result in results:
        new_list.append([result[:10]])
    return new_list