# Import necessary modules
import torch as torch_lib
import torch.nn as neural_net
from pdb import set_trace as debug_trace

# Define the modules to be exported when this script is imported
__all__ = ['bidirectional_lstm']

# Define a class for the Bi-directional LSTM model
class BidirectionalLSTM(neural_net.Module):
    def __init__(self, config):
        # Initialize the parent class (nn.Module)
        super(BidirectionalLSTM, self).__init__()
        
        # Set fixed embedding dimension, as it matches the pre-trained GloVe model
        self.embedding_dimension = 300
        # Configuration options
        self.hidden_dimension = config['d_hidden']
        self.direction_count = 2
        self.layer_count = 2
        self.concat_factor = 4
        self.execution_device = config['device']

        # Load pre-trained embedding vectors
        self.embedding_layer = neural_net.Embedding.from_pretrained(torch_lib.load('.vector_cache/{}_vectors.pt'.format(config['dataset'])))
        self.linear_projection = neural_net.Linear(self.embedding_dimension, self.hidden_dimension)
        self.lstm_layer = neural_net.LSTM(self.hidden_dimension, self.hidden_dimension, self.layer_count,
                                          bidirectional=True, batch_first=True, dropout=config['dp_ratio'])
        self.activation_relu = neural_net.ReLU()
        self.dropout_layer = neural_net.Dropout(p=config['dp_ratio'])

        # Linear layers for processing
        self.linear1 = neural_net.Linear(self.hidden_dimension * self.direction_count * self.concat_factor, self.hidden_dimension)
        self.linear2 = neural_net.Linear(self.hidden_dimension, self.hidden_dimension)
        self.linear3 = neural_net.Linear(self.hidden_dimension, config['out_dim'])

        # Initialize weights and biases for linear layers
        for linear in [self.linear1, self.linear2, self.linear3]:
            neural_net.init.xavier_uniform_(linear.weight)
            neural_net.init.zeros_(linear.bias)

        # Sequential output layer
        self.output_layer = neural_net.Sequential(
            self.linear1,
            self.activation_relu,
            self.dropout_layer,
            self.linear2,
            self.activation_relu,
            self.dropout_layer,
            self.linear3
        )

    def forward(self, batch):
        # Embedding and projection for premise and hypothesis
        premise_embedding = self.embedding_layer(batch.premise)
        hypothesis_embedding = self.embedding_layer(batch.hypothesis)

        premise_projection = self.activation_relu(self.linear_projection(premise_embedding))
        hypothesis_projection = self.activation_relu(self.linear_projection(hypothesis_embedding))

        # Initialize hidden and cell states
        initial_state = torch_lib.tensor([]).new_zeros((self.layer_count * self.direction_count, batch.batch_size, self.hidden_dimension)).to(self.execution_device)

        # LSTM processing
        _, (premise_ht, _) = self.lstm_layer(premise_projection, (initial_state, initial_state))
        _, (hypothesis_ht, _) = self.lstm_layer(hypothesis_projection, (initial_state, initial_state))
        
        # Combine and reshape the outputs
        premise = premise_ht[-2:].transpose(0, 1).contiguous().view(batch.batch_size, -1)
        hypothesis = hypothesis_ht[-2:].transpose(0, 1).contiguous().view(batch.batch_size, -1)

        # Combine features and pass through the output layer
        combined_features = torch_lib.cat((premise, hypothesis, torch_lib.abs(premise - hypothesis), premise * hypothesis), 1)
        return self.output_layer(combined_features)

# Function to create an instance of BidirectionalLSTM
def bidirectional_lstm(config):
    return BidirectionalLSTM(config)
