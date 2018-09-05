import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from lib.config.ConfigParams import ConfigParams
from lib.data.Preprocessing import Preprocessing
from lib.data.SingleDirDataset import SingleDirDataset
from lib.model.ModelsFactory import ModelsFactory
from lib.vocab.Vocabulary import Vocabulary


def clean_sentence(output, vocab):
    words = [vocab.idx2word[x] for x in output]

    trimmed_words = []

    for word in words:
        if word == vocab.start_word:
            continue
        if word == vocab.end_word:
            break
        trimmed_words.append(word)

    return " ".join(trimmed_words)


def do_parsing():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="PyTorch test script, read images from directory")
    parser.add_argument("--test_dir", required=True, type=str, help="Test directory")
    parser.add_argument("--config_file", required=True, type=str, help="Config file path")
    parser.add_argument("--enc_model_path", required=True, type=str, help="Encoder model filepath")
    parser.add_argument("--dec_model_path", required=True, type=str, help="Decoder model filepath")
    parser.add_argument("--vocab_file", required=True, type=str, help="Vocabulary pickle file")
    args = parser.parse_args()
    return args


def main():

    # Load params
    args = do_parsing()
    print(args)

    # Load config file with model and preprocessing (must be the same used in training to be coherent)
    config = ConfigParams(args.config_file)

    # Prepare captions vocabulary
    vocab = Vocabulary(vocab_file=args.vocab_file, force_rebuild=False)
    if vocab.built is False:
        raise Exception("Error: vocabulary not built/not correcly loaded")

    # Prepare preprocessing transform pipeline (same processing of validation dataset)
    preprocessing_transforms = Preprocessing(config)
    preprocessing_transforms_test = preprocessing_transforms.get_transforms_val()

    # Read test Dataset,
    dataset_test = SingleDirDataset(args.test_dir, preprocessing_transforms_test)
    print("Test - Samples: {0}".format(str(len(dataset_test))))

    # Load model and apply .eval() and .cuda()
    enc, dec = ModelsFactory.create(config, len(vocab))
    print(enc)
    print(dec)
    enc.cuda()
    enc.eval()
    dec.cuda()
    dec.eval()

    # Load trained weights
    enc.load_state_dict(torch.load(args.enc_model_path))
    dec.load_state_dict(torch.load(args.dec_model_path))

    # Create a PyTorch DataLoader from CatDogDataset
    test_loader = DataLoader(dataset_test, batch_size=config.batch_size, shuffle=False, num_workers=8)

    print("Evaluating test dataset...")

    for batch_i, data in enumerate(test_loader):

        # Retrieve images
        images = data["image"]

        # Move to GPU
        images = images.type(torch.cuda.FloatTensor)

        # forward pass to get outputs
        for image_index in range(images.shape[0]):
            features = enc(images[image_index].unsqueeze(0)).unsqueeze(1)
            output = dec.sample(features)
            sentence = clean_sentence(output, vocab)

            file = data["file"][image_index]
            print("File " + file + " -> " + sentence)

    print("Test finished")


if __name__ == "__main__":
    main()
