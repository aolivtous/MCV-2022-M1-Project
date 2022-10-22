import pickle

pickle_file = None
with open( '../qsd2_w2/text_boxes.pkl', "rb" ) as f:
    pickle_file = pickle.load(f)

print(pickle_file)