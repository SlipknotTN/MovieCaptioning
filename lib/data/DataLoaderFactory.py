import torch.utils.data as data
from .CaptionsDataLoader import CaptionsDataLoader


class DataLoaderFactory(object):

    @classmethod
    def create(cls, dataset, batch_size, num_workers=8):

        # Randomly sample a caption length, and sample only the indices with that length
        indices = dataset.get_train_indices(batch_size)

        # Create and assign a batch sampler to retrieve a batch with the sampled indices.
        initial_sampler = data.sampler.SubsetRandomSampler(indices=indices)

        # We set a dummy BatchSampler that will be reassigned at each batch retrieval

        # Data loader for a image+captions dataset (first
        return CaptionsDataLoader(dataset=dataset,
                                  num_workers=num_workers,
                                  batch_sampler=data.sampler.BatchSampler(sampler=initial_sampler,
                                                                          batch_size=batch_size,
                                                                          drop_last=False),
                                  batch_size=batch_size)
