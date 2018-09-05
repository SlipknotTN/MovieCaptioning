import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from lib.config.ConfigParams import ConfigParams
from lib.data.Preprocessing import Preprocessing
from lib.data.DatasetFactory import DatasetFactory
from lib.data.DataLoaderFactory import DataLoaderFactory
from lib.model.ModelsFactory import ModelsFactory
from lib.vocab.Vocabulary import Vocabulary
from lib.vocab.VocabularyBuilder import VocabularyBuilder


def do_parsing():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="PyTorch training script")
    parser.add_argument("--dataset_dir", required=True, type=str, help="Dataset root directory")
    parser.add_argument("--config_file", required=True, type=str, help="Config file path")
    parser.add_argument("--model_output_dir", required=False, type=str, default="./export/model",
                        help="Directory where to save PyTorch models")
    parser.add_argument("--vocab_file", required=True, type=str, help="Vocabulary pickle file")
    parser.add_argument("--data_source", required=False, type=str, default="coco", help="Vocabulary and images source")
    parser.add_argument("--force_rebuild_vocab", required=False, type=bool, default=False,
                        help="Rebuild vocabulary overwriting the old one")
    args = parser.parse_args()
    return args


def main():

    # Load params
    args = do_parsing()
    print(args)

    # Load config file with model, hyperparameters and preprocessing
    config = ConfigParams(args.config_file)

    # Prepare captions vocabulary
    vocab = Vocabulary(config.vocab_threshold, vocab_file=args.vocab_file, force_rebuild=args.force_rebuild_vocab)
    if vocab.built is False:
        VocabularyBuilder.build(vocab, dataset_dir=args.dataset_dir, source=args.data_source)

    # Prepare preprocessing transform pipeline
    preprocessing_transforms = Preprocessing(config)
    preprocessing_transforms_train = preprocessing_transforms.get_transforms_train()
    preprocessing_transforms_val = preprocessing_transforms.get_transforms_val()

    # Read Dataset
    print("Preparing dataset retrieving image ids and ready to use captions...")
    dataset_train = DatasetFactory.create(dataset_dir=args.dataset_dir, vocabulary=vocab,
                                          transform=preprocessing_transforms_train, source=args.data_source)
    # dataset_train = StandardDataset(args.dataset_train_dir, preprocessing_transforms_train)
    print("Train - Samples: {0}".format(str(len(dataset_train))))
    # dataset_val = StandardDataset(args.dataset_val_dir, preprocessing_transforms_val)
    # print("Validation - Classes: {0}, Samples: {1}".
    #       format(str(len(dataset_val.get_classes())), str(len(dataset_val))))
    # print("Classes " + str(dataset_train.get_classes()))

    # Load model and apply .train() and .cuda()
    enc, dec = ModelsFactory.create(config, len(vocab))
    print(enc)
    print(dec)
    enc.cuda()
    enc.train()
    dec.cuda()
    dec.train()
    params_to_train = list(enc.embed.parameters()) + list(dec.parameters())

    # Create a PyTorch DataLoader for Image+Captions (two of them: train + val)
    train_loader = DataLoaderFactory.create(dataset_train, batch_size=config.batch_size)
    #val_loader = DataLoader(dataset_val, batch_size=config.batch_size, shuffle=False, num_workers=8)

    # Set Optimizer and Loss
    # CrossEntropyLoss add LogSoftmax to the model while NLLLoss doesn't do it
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params_to_train, lr=config.learning_rate, momentum=config.momentum)

    total_step = len(dataset_train) // config.batch_size

    for epoch in range(1, config.epochs + 1):

        # Iterate on train batches and update weights using loss
        for i_step in range(1, total_step + 1):

            # Select a fixed length for captions of the same batch to speedup the training
            train_loader.resample_indices()

            # Obtain the batch.
            data = next(iter(train_loader))

            # get the input images and their corresponding labels
            images = data['image']
            captions = data['caption']

            # Move to GPU
            captions = captions.type(torch.cuda.LongTensor)
            images = images.type(torch.cuda.FloatTensor)

            # Zero the gradients
            dec.zero_grad()
            enc.zero_grad()

            # Forward pass the inputs through the CNN-RNN model
            features = enc(images)
            outputs = dec(features, captions)

            # Calculate the batch loss.
            # Prediction has probabilities for each batch_size x caption_length x vocab_size
            # (reshaped to (batch_size * captions_length) x vocab_size)
            # while target is only batch_size x caption_length (reshaped to batch_size * caption_length)
            loss = criterion(outputs.view(-1, len(vocab)), captions.view(-1))

            # backward pass to calculate the weight gradients
            loss.backward()

            # update the weights
            optimizer.step()

            # Get training statistics.
            stats = 'Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f' \
                    % (epoch, config.epochs, i_step, total_step, loss.item(), np.exp(loss.item()))

            if i_step % 100 == 0:
                print('\r' + stats)

        # TODO: Iterate on validation batches
        # print("Calculating validation accuracy...")

        torch.save(dec.state_dict(), os.path.join(args.model_output_dir, 'decoder-%d.pkl' % epoch))
        torch.save(enc.state_dict(), os.path.join(args.model_output_dir, 'encoder-%d.pkl' % epoch))

    print("End")


if __name__ == "__main__":
    main()
