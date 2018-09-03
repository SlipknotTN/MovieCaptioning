from torchvision import transforms


class Preprocessing(object):

    def __init__(self, config):

        # TODO: Read params from config

        # From PyTorch doc :
        # All pre-trained models expect input images normalized in the same way,
        # i.e. mini-batches of 3-channel RGB images of shape (3 x H x W),
        # where H and W are expected to be at least 224.
        # The images have to be loaded in to a range of [0, 1] and then normalized using
        # mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]

        # Resize smaller dimension to 256 and then random crop 224x224.
        # ToTensor converts HWC PIL.Image to CHW float tensor.
        self.data_transform_train = transforms.Compose([transforms.Resize(256), transforms.RandomCrop(224),
                                                        transforms.RandomHorizontalFlip(),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                             std=[0.229, 0.224, 0.225])
                                                        ])

        self.data_transform_val = transforms.Compose([transforms.Resize((224, 224)),
                                                      transforms.ToTensor(),
                                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                           std=[0.229, 0.224, 0.225])
                                                      ])

    def get_transforms_train(self):

        return self.data_transform_train

    def get_transforms_val(self):

        return self.data_transform_val
