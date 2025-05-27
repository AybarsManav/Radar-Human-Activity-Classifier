import os
import pickle

def main():
    # For data pre-processing, make sure datasets are extracted into the 'datasets' directory
    # in the project root. The structure should be:
    # datasets
    #   1 December 2017 Dataset
    #      *.dat
    #
    # Then, run preprocessing.py.

    # Load pre-processed data
    # spectrograms:         list of 2D arrays
    # spectrograms_MTI:     list of 2D arrays
    # labels:               1D array of integers
    with open(os.path.join('preprocessed_data', 'spectrograms.pkl'), 'rb') as f:
        spectrograms, spectrograms_MTI, labels = pickle.load(f)

if __name__ == '__main__':
    main()