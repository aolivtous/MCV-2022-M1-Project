import pickle

pickle_file = None
with open( '../../qst1_w5/results/result.pkl', "rb" ) as f:
    pickle_file = pickle.load(f)

# To fix the problem with the pickle file (00000 is in DB)
# pickle_file[0][0] = 213

# # Save the file
# with open( '../qsd1_w5/gt_corresps.pkl', "wb" ) as f:
#     pickle.dump(pickle_file, f)

print(pickle_file)