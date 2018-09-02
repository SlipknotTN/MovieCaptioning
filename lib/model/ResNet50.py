import torch
import torch.nn as nn
import torchvision.models as models


class ResNetEncoderCNN(nn.Module):
    def __init__(self, config)
        super(ResNetEncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)

        # Freeze layers
        for param in resnet.parameters():
            param.requires_grad_(False)

        # Substitute latest fc 2048 -> 1000 (ImageNet classes)
        # with a fc 2048 -> embed_size
        # We train only this layer, the others are frozen
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, config.embed_size)

    def forward(self, images):
        features = self.resnet(images)
        # Flatten for fc
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features


class ResNetDecoderRNN(nn.Module):
    def __init__(self, config, vocab_size):
        super(ResNetDecoderRNN, self).__init__()
        self.embed_size = config.embed_size
        self.hidden_size = config.hidden_size
        self.vocab_size = vocab_size
        self.lstm_layers = config.lstm_layers

        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embed_size)

        # No dropout (not applied to latest layer in any case...)
        self.lstm = nn.LSTM(input_size=self.embed_size, hidden_size=self.hidden_size,
                            num_layers=self.lstm_layers, batch_first=True)

        # dropout layer
        # self.dropout = nn.Dropout(p=self.drop_prob)

        # Final fc
        self.fc = nn.Linear(in_features=self.hidden_size, out_features=self.vocab_size)

        # Initialize fc
        self.init_weights()

    def forward(self, features, captions):
        # batch_size x embed_size
        # print("Features shape: " + str(features.shape))

        # batch_size x captions_length
        # (constant once sampled, but it is not a problem to change it at every forward call)
        # print("Captions shape: " + str(captions.shape))

        # Remove latest token from captions (to use as input)
        input_captions = captions.clone()
        input_captions = input_captions[:, :-1]
        # print("Input captions shape: " + str(input_captions.shape))

        # Pass captions through embedding layer
        x = self.embedding(input_captions)
        # batch_size x (captions_length - 1) x embed_size
        # print("Captions embedded shape: " + str(x.shape))

        # Concat image embedding with captions embedding
        features_r = torch.reshape(features, (features.shape[0], 1, features.shape[1]))
        # print("Features reshaped shape: " + str(features_r.shape))
        x = torch.cat((features_r, x), dim=1)
        # batch_size x (1 + captions_length - 1) x embed_size
        # print("LSTM input shape: " + str(x.shape))

        # Apply lstm with zero hidden and cell state
        x, (h, c) = self.lstm(x)
        # batch_size x captions_length x hidden_size
        # print("LSTM output shape: " + str(x.shape))

        # Apply fc to get softmax inputs
        x = self.fc(x)
        # batch_size x captions_length x vocab_size
        # print("FC output shape: " + str(x.shape))

        return x

    def sample(self, inputs, states=None, max_len=20):
        """
        accepts pre-processed image tensor (inputs) and
        returns predicted sentence (list of tensor ids of length max_len)
        :param inputs:
        :param states:
        :param max_len:
        :return:
        """
        pass

    def init_weights(self):
        """
        Initialize weights for fully connected layer
        """
        initrange = 0.1

        # Set bias tensor to all zeros
        self.fc.bias.data.fill_(0)
        # FC weights as random uniform
        self.fc.weight.data.uniform_(-1, 1)