#from .SqNet import SqNet
from .ResNet50 import ResNetEncoderCNN, ResNetDecoderRNN


class ModelsFactory(object):

    @classmethod
    def create(cls, config, vocab_size):

        if config.architecture == "sqnet":

            #return SqNet(num_classes)
            raise NotImplementedError

        if config.architecture == "resnet50":

            return ResNetEncoderCNN(config), ResNetDecoderRNN(config, vocab_size)

        else:

            raise Exception("Model architecture " + config.architecture + " not supported")
