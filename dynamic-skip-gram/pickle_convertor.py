
import pickle
import pandas as pd

if __name__ == "__main__":
    # pickle_in = open("/Users/Ganymedian/Desktop/dynamic-rw/datasets/academic.pickle","rb")
    # example_dict = pickle.load(pickle_in, encoding='latin1')
    example_dict = pd.read_pickle("/Users/Ganymedian/Desktop/academic_network/labels.pickle")
    print(example_dict)