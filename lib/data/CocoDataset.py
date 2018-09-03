import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from pycocotools.coco import COCO
import nltk


class CocoDataset(Dataset):

    def __init__(self, dataset_dir, vocabulary, transform):

        self.dataset_dir = dataset_dir
        self.vocab = vocabulary
        self.samples = []
        self.captions_lengths = []
        self.transform = transform

        self.coco_train = COCO(os.path.join(dataset_dir, "annotations", "captions_train2014.json"))
        ids = self.coco_train.anns.keys()
        for i, id in enumerate(ids):
            caption = str(self.coco_train.anns[id]['caption'])
            image_id = int(self.coco_train.anns[id]["image_id"])
            tokens = nltk.tokenize.word_tokenize(caption.lower())
            tokens_ids = [self.vocab(self.vocab.start_word)]
            for token in tokens:
                tokens_ids.append(self.vocab(token))
            tokens_ids.append(self.vocab(self.vocab.end_word))

            self.captions_lengths.append(len(tokens_ids))

            # Save in samples list image_id and captions tokenized and converted to indexes
            self.samples.append((image_id, tokens_ids))

            if i % 100000 == 0:
                print("[%d/%d] Preparing image_id and captions..." % (i, len(ids)))

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):

        image_id = self.samples[idx][0]
        captions_ids = self.samples[idx][1]

        image_path = self.coco_train.loadImgs(image_id)[0]['file_name']

        # Convert image to tensor and pre-process using transform
        image = Image.open(os.path.join(self.dataset_dir, "images", "train2014", image_path)).convert('RGB')
        image = self.transform(image)

        # Convert to tensor
        captions_ids = torch.Tensor(captions_ids).long()

        return {'image': image, 'caption': captions_ids, 'file': image_path}

    def get_train_indices(self, batch_size):
        # Select a fixed length for captions of the same batch to speedup the training
        sel_length = np.random.choice(self.captions_lengths)
        all_indices = np.where([self.captions_lengths[i] == sel_length for i in np.arange(len(self.captions_lengths))])[0]
        # Return indices of the batch (length: batch_size)
        indices = list(np.random.choice(all_indices, size=batch_size))
        return indices
