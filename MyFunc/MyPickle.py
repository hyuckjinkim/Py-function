import pickle

def to_pickle(data,path):

    with open(path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    
def read_pickle(path):
    
    with open(path, 'rb') as f:
        data = pickle.load(f)
        
    return data