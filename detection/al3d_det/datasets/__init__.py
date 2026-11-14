import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data import DistributedSampler as _DistributedSampler

from al3d_utils import common_utils

from .dataset import DatasetTemplate
from .dsert.dsert_dataset import DSERTTrainingDataset, DSERTInferenceDataset
from .dsert.dsert_single_sensor_dataset import DSERTSingleSensorDataset
from .man_truckscenes.truckscenes_dataset import TruckScenesDataset

__all__ = {
    'DatasetTemplate': DatasetTemplate,
    'DSERTTrainingDataset': DSERTTrainingDataset,
    'DSERTInferenceDataset': DSERTInferenceDataset,
    'DSERTSingleSensorDataset': DSERTSingleSensorDataset,
    'TruckScenesDataset': TruckScenesDataset,
}


class DistributedSampler(_DistributedSampler):

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


def _seed_worker(worker_id):
    # DataLoader seeds `random` and torch in each worker from the main process's torch RNG, but not numpy, which
    # the augmentations use. Module level, so it can be pickled for 'spawn' workers.
    import numpy as np
    np.random.seed(torch.initial_seed() % 2 ** 32)


def build_dataloader(dataset_cfg, class_names, batch_size, dist, root_path=None, workers=4,
                     logger=None, training=True, merge_all_iters_to_one_epoch=False, total_epochs=0, length=0,
                     seed=None):
    """seed: None keeps the old behaviour (shuffle order from DistributedSampler's default seed 0, numpy in the
    workers unseeded); an int gives every seed its own data order and seeds numpy in the workers."""
    dataset = __all__[dataset_cfg.DATASET](
        dataset_cfg=dataset_cfg,
        class_names=class_names,
        root_path=root_path,
        training=training,
        logger=logger,
    )

    if merge_all_iters_to_one_epoch:
        assert hasattr(dataset, 'merge_all_iters_to_one_epoch')
        dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)

    if dist:
        if training:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset, seed=0 if seed is None else seed)
        else:
            rank, world_size = common_utils.get_dist_info()
            sampler = DistributedSampler(dataset, world_size, rank, shuffle=False)
    else:
        sampler = None

    new_dataset = dataset

    if length > 0:
        indices = torch.arange(len(dataset))[:length]
        new_dataset = Subset(dataset, indices)

    dataloader = DataLoader(
        new_dataset, batch_size=batch_size, pin_memory=True, num_workers=workers,
        shuffle=(sampler is None) and training, collate_fn=dataset.collate_batch,
        drop_last=False, sampler=sampler, timeout=0,
        # Workers are started with 'spawn' (see init_dist_pytorch); keep them alive across epochs.
        persistent_workers=workers > 0,
        worker_init_fn=_seed_worker if seed is not None else None
    )

    return dataset, dataloader, sampler
