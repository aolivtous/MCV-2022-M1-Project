import cv2

def init(name_query):
    global argument_relations
    global methods_search

    global name_db
    global dir_base
    global results_name
    global aux_name

    global dir_db
    global dir_query
    global dir_results
    global dir_db_aux
    global dir_query_aux
    global new_dirs

    global test_image

    argument_relations = { 
    'corr': cv2.HISTCMP_CORREL,
    'chisq': cv2.HISTCMP_CHISQR,
    'intersect': cv2.HISTCMP_INTERSECT,
    'hellin': cv2.HISTCMP_BHATTACHARYYA,
    'eucli': False
    }

    # color_code = ["RGB", "HSV", "LAB", "YCrCb"]
    # Possible arguments of distance_type at argument_relations
    methods_search = {
        1: {
            'color_code': 'LAB',
            'distance_type': 'chisq'
        },
        2: {
            'color_code': 'LAB',
            'distance_type': 'intersect'
        }
    }

    # Constant arguments
    name_db = 'BBDD'
    dir_base = '../'
    results_name = 'results'
    aux_name = 'aux'

    # Directories assignment (always end with /)
    dir_db = f'{dir_base}{name_db}/' 
    dir_query = f'{dir_base}{name_query}/'
    dir_results = f'{dir_query}{results_name}/'
    dir_db_aux = f'{dir_db}{aux_name}/'
    dir_query_aux = f'{dir_query}{aux_name}/'
    new_dirs = [dir_results, dir_db_aux, dir_query_aux]

    # ONLY TESTING VARIABLES
    test_image = '' # ! Set to '' to iterate all the query