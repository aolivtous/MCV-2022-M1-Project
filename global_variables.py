import cv2

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def init(name_query):
    global argument_relations
    global methods_search
    global weights
    global n_patches
    global n_bins
    global in_db_threshold

    global name_db
    global dir_base
    global results_name
    global aux_name

    global dir_db
    global dir_query
    global dir_museum
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
        'default': {
            'color_code': 'LAB',
            'distance_type': 'hellin',
            'feature_algorithm': 'ORB',   # SIFT, SURF, BRIEF, ORB
            'match_algorithm': 'FLANN'        # BF (very slow), FLANN (faster), if feature_algorithm is ORB or BRIEF we use a specific mathc_algorithm
        }
    }

    n_patches = 5
    n_bins = 40

    weights = {
        'color': 1.0,
        'texture': 0.0,
        'text': 0.0,
        'feature': 0.0
    }
    in_db_threshold = 1.035
    # in_db_threshold = 0.3 * 1e-7 # When unnormalized with db
 
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

    dir_museum = f'{dir_base}museum/'
    
    # ONLY TESTING VARIABLES
    test_image = '10' # ! Set to '' to iterate all the query
