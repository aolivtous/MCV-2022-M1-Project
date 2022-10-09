import pickle

pickle_file = None
try:
    with open( '../qst1_w1/predictions/result.pkl', "rb" ) as f:
        pickle_file = pickle.load(f)
except:
    pass

print(pickle_file)