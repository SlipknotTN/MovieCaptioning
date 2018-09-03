import torch.utils.data as data


class CaptionsDataLoader(data.DataLoader):

    def __init__(self, dataset, batch_sampler, batch_size, num_workers=8):
        self.inner_batch_size = batch_size
        super(CaptionsDataLoader, self).__init__(dataset=dataset, num_workers=num_workers, batch_sampler=batch_sampler)

    def resample_indices(self):
        # Randomly sample a caption length, and sample indices with that length.
        indices = self.dataset.get_train_indices(self.inner_batch_size)
        # Create and assign a batch sampler to retrieve a batch with the sampled indices.
        new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
        self.batch_sampler.sampler = new_sampler
