import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_size)  # Embedding layer
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)  # LSTM block
        self.fc = nn.Linear(hidden_size, vocab_size)  # Fully connected layer
    
    def forward(self, features, captions):
        # remove '<end>' from captions as it would not be needed
        captions = captions[:, :-1]
        
        # pass captions through embedding layer to generate embeddings
        embeddings = self.embedding(captions)
        
        # concatenate captions embeddings with images features
        lstm_inputs = torch.cat((features.unsqueeze(1), embeddings), 1)
        
        # pass the inputs through lstm, note that I'm ignoring the hidden states
        lstm_outputs, _ = self.lstm(lstm_inputs)
        
        # pass lstm outputs through fully connected layer to estimate predictions
        outputs = self.fc(lstm_outputs)
        
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        outputs = []
        for i in range(max_len):
            lstm_out, states = self.lstm(inputs, states)  # pass throught lstm layer
            out = self.fc(lstm_out)  # pass through fully connected layer
            word = torch.argmax(out, dim=2)  # get the word (vocab index) with max probability
            outputs.append(word.item())  # append the index to the outputs
            
            if word.item() == 1:  # break if end word appears (index=1 corresponds to <end> in our vocabulary)
                break
            
            inputs = self.embedding(word)  # get the word embeddings for next word prediction
        
        return outputs