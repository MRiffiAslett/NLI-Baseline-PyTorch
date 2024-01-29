# Import the necessary modules
import os as os_module
import sys as system_module
import torch as torch_framework
from torchtext.data import Field as TextField, Iterator as DatasetIterator
from torchtext import datasets as text_datasets
from utility_functions import create_directories as make_dirs
from pdb import set_trace as debug_trace

# Define the modules that will be exported when this script is imported
__all__ = ['stanford_nli']

# Define a class for handling the Stanford Natural Language Inference (SNLI) dataset
class StanfordNLI():
    def __init__(self, config):
        # Initialize text and label fields with specific preprocessing
        self.TextField = TextField(lower=True, tokenize='spacy', batch_first=True)
        self.LabelField = TextField(sequential=False, unk_token=None, is_target=True)
        
        # Split the dataset into training, development, and test sets
        self.training_data, self.development_data, self.test_data = text_datasets.SNLI.splits(self.TextField, self.LabelField)
        
        # Build vocabulary for the text and label fields using the training and development data
        self.TextField.build_vocab(self.training_data, self.development_data)
        self.LabelField.build_vocab(self.training_data)

        # Path for saving/loading pre-trained vectors
        vector_cache_path = '.vector_cache/snli_vectors.pt'
        # Load pre-trained vectors if they exist, else download and save them
        if os_module.path.isfile(vector_cache_path):
            self.TextField.vocab.vectors = torch_framework.load(vector_cache_path)
        else:
            self.TextField.vocab.load_vectors('glove.840B.300d')
            make_dirs(os_module.path.dirname(vector_cache_path))
            torch_framework.save(self.TextField.vocab.vectors, vector_cache_path)

        # Create iterators for the dataset
        self.train_iter, self.dev_iter, self.test_iter = DatasetIterator.splits(
            (self.training_data, self.development_data, self.test_data),
            batch_size=config['batch_size'],
            device=config['device']
        )

    # Method to get the size of the vocabulary
    def vocab_size(self):
        return len(self.TextField.vocab)

    # Method to get the output dimension
    def out_dim(self):
        return len(self.LabelField.vocab)

    # Method to get the label-to-index mapping
    def labels(self):
        return self.LabelField.vocab.stoi

# Function to create an instance of the StanfordNLI
def stanford_nli(config):
    return StanfordNLI(config)
