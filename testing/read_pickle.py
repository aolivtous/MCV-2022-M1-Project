import pickle

pickle_file = None
with open( '../qsd1_w2/gt_corresps.pkl', "rb" ) as f:
    pickle_file = pickle.load(f)

print(pickle_file)