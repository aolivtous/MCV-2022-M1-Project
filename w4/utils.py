import global_variables
import numpy as np

def str_to_bool(v):
    """
    It takes a string and its boolean value
    
    :param v: The value evaluate to a boolean
    :return: A boolean value.
    """
    return str(v).lower() in ("yes", "true", "t", "1")

def get_sorted_list_of_lists_from_dict_of_dicts(dictionary, dict_type, num_paintings, two_level = True, may_not_be_in_db = False, k = 10):
    """
    It takes a dictionary of dictionaries and returns a list of lists, where each list is a sorted list
    of the keys of the inner dictionaries
    
    :param dists: a dictionary of dictionaries, where the key of the outer dictionary is the query
    image, and the key of the inner dictionary is the database image. The value of the inner dictionary
    is an object of type Distance, which contains the distance between the query image and the database
    image
    :param distance_type: the type of distance metric to use.
    :return: A list of lists, where each list is the sorted list of image index results for a query.
    """
    list_of_lists = []
    if dict_type in ("eucli", "hellin", "chisq"):
        reverse = False
    else:
        reverse = True 
    
    # Sort dict to assure the order
    dictionary = dict(sorted(dictionary.items(),key=lambda x:x[0]))
    aux_result = []
    for key_query, _ in dictionary.items():
        sorted_key_list = [int(key) for key, _ in sorted(dictionary[key_query].items(), key=lambda item: item[1].dist, reverse=reverse)]
        sorted_key_list_top_k = sorted_key_list[:k]
        if may_not_be_in_db:
            sorted_value_list = [dist.dist for _, dist in sorted(dictionary[key_query].items(), key=lambda item: item[1].dist, reverse=reverse)]
            std_slope_top1_relation = (np.std(sorted_value_list[:2])) / (np.std(sorted_value_list) + 0.0001)
            if std_slope_top1_relation < global_variables.in_db_threshold:
                sorted_key_list_top_k = [-1]
        if(two_level):
            # Check number of keys with the same 5 characters as key_query
            # parts = str(len([key for key in dictionary.keys() if key[:5] == key_query[:5]]))
            parts = str(num_paintings[key_query.split('_')[0]]) # Get the number of the painting without the part
            aux_result.append(sorted_key_list_top_k)
            if key_query.endswith(parts):
                list_of_lists.append(aux_result)
                aux_result = []  
        else:
            list_of_lists.append(sorted_key_list_top_k)

    return list_of_lists

def get_simple_list_from_dict(dictionary):
    # Sort dict to assure the order
    list_ = []
    dictionary = dict(sorted(dictionary.items(),key=lambda x:x[0]))
    for _, coords in dictionary.items():
        list_.append(coords)

    return list_