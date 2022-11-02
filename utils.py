def str_to_bool(v):
    """
    It takes a string and its boolean value
    
    :param v: The value evaluate to a boolean
    :return: A boolean value.
    """
    return str(v).lower() in ("yes", "true", "t", "1")

def get_sorted_list_of_lists_from_dict_of_dicts(dictionary, dict_type, num_paintings, two_level = True, k = 10):
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
        sorted_list = [int(key) for key, _ in sorted(dictionary[key_query].items(), key=lambda item: item[1].dist, reverse=reverse)][:k]
        if(two_level):
            # Check number of keys with the same 5 characters as key_query
            # parts = str(len([key for key in dictionary.keys() if key[:5] == key_query[:5]]))
            parts = str(num_paintings[key_query.split('_')[0]]) # Get the number of the painting without the part
            aux_result.append(sorted_list)
            if key_query.endswith(parts):
                list_of_lists.append(aux_result)
                aux_result = []  
        else:
            list_of_lists.append(sorted_list)

    return list_of_lists

def get_simple_list_of_lists_from_dict_of_dicts(dictionary):
    # Sort dict to assure the order
    list_of_lists = []
    dictionary = dict(sorted(dictionary.items(),key=lambda x:x[0]))
    aux_list = []
    for key_query, coords in dictionary.items():
        list_of_lists.append(coords)

    return list_of_lists