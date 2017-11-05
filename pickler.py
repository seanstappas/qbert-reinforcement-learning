import pickle


def save_to_pickle(data, filename):
    with open('pickle/{}.pkl'.format(filename), 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_from_pickle(filename):
    with open('pickle/{}.pkl'.format(filename), 'rb') as f:
        data = pickle.load(f)
    return data
