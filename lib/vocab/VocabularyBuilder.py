import os
import nltk
from pycocotools.coco import COCO
from collections import Counter


class VocabularyBuilder(object):

    @classmethod
    def build(cls, vocabulary, dataset_dir, source="coco"):

        counter = Counter()

        if source == "coco":

            # TODO: Add also validation annotations to create vocabulary
            coco = COCO(os.path.join(dataset_dir, "annotations", "captions_train2014.json"))
            ids = coco.anns.keys()
            for i, id in enumerate(ids):
                caption = str(coco.anns[id]['caption'])
                tokens = nltk.tokenize.word_tokenize(caption.lower())
                counter.update(tokens)

                if i % 100000 == 0:
                    print("[%d/%d] Tokenizing captions..." % (i, len(ids)))

        else:

            raise Exception("Vocabulary source " + source + " not supported")

        vocabulary.build_vocab(counter)
