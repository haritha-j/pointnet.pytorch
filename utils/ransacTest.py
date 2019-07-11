import pickle

f = open('ransac_dataset.pickle', 'rb')
collection = pickle.load(f)

print (len(collection[0][0][0]))