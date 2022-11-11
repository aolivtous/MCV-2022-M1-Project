import pickle

pickle_file = None
with open( '../museum/precomputed_dct.pkl', "rb" ) as f:
    pickle_file = pickle.load(f)

print(pickle_file)