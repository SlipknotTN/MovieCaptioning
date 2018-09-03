from .CocoDataset import CocoDataset


class DatasetFactory(object):

    @classmethod
    def create(cls, dataset_dir, vocabulary, transform, source="coco"):

        if source == "coco":

            return CocoDataset(dataset_dir, vocabulary, transform)

        else:

            raise Exception("Dataset source " + source + " not supported")
