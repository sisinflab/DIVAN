import numpy as np
import torch
from torch.utils import data
import glob
import logging

class Dataset(data.Dataset):
    def __init__(self, feature_map, data_path):
        self.feature_map = feature_map
        self.darray = self.load_data(data_path)
        self.group_id_col_idx = list(self.feature_map.features.keys()).index(self.feature_map.group_id)

    def __getitem__(self, index):
        return self.darray[index, :]

    def __len__(self):
        return self.darray.shape[0]

    def load_data(self, data_path):
        data_dict = np.load(data_path)  # dict of arrays
        data_arrays = []
        all_cols = list(self.feature_map.features.keys()) + self.feature_map.labels
        for col in all_cols:
            array = data_dict[col]
            if array.ndim == 1:
                array = array[:, np.newaxis]
            data_arrays.append(array)
        data_tensor = torch.from_numpy(np.concatenate(data_arrays, axis=1))
        return data_tensor

class NpzDataLoader(data.DataLoader):
    def __init__(self, feature_map, data_path, batch_size=32, shuffle=False, num_workers=1, **kwargs):
        if not data_path.endswith(".npz"):
            data_path += ".npz"
        self.dataset = Dataset(feature_map, data_path)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(self.dataset)
        self.num_blocks = 1
        self.num_batches = int(np.ceil(self.num_samples * 1.0 / self.batch_size))

        if self.shuffle:
            self.dataset = self.grouped_shuffle(self.dataset)

        super(NpzDataLoader, self).__init__(dataset=self.dataset, batch_size=batch_size,
                                            shuffle=False, num_workers=num_workers)

    def __len__(self):
        return self.num_batches

    def grouped_shuffle(self, dataset):
        data_list = list(dataset)
        group_id_col_idx = dataset.group_id_col_idx

        # Group by group_id
        groups = {}
        for sample in data_list:
            group_id = sample[group_id_col_idx].item()
            if group_id not in groups:
                groups[group_id] = []
            groups[group_id].append(sample)

        # Shuffle groups
        group_ids = list(groups.keys())
        np.random.shuffle(group_ids)

        # Flatten the shuffled groups
        shuffled_data = [sample for group_id in group_ids for sample in groups[group_id]]

        # Convert back to ShuffledDataset
        return ShuffledDataset(shuffled_data)

class NpzBlockDataLoader(data.DataLoader):
    def __init__(self, feature_map, data_path, batch_size=32, shuffle=False,
                 num_workers=1, buffer_size=100000, **kwargs):
        data_blocks = glob.glob(data_path + "/*.npz")
        assert len(data_blocks) > 0, f"invalid data_path: {data_path}"
        if len(data_blocks) > 1:
            data_blocks.sort()  # sort by part name
        self.data_blocks = data_blocks
        self.num_blocks = len(self.data_blocks)
        self.feature_map = feature_map
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_batches, self.num_samples = self.count_batches_and_samples()

        datapipe = BlockDataPipe(self.data_blocks, feature_map, shuffle=shuffle)
        super(NpzBlockDataLoader, self).__init__(dataset=datapipe, batch_size=batch_size,
                                                 num_workers=num_workers)

    def __len__(self):
        return self.num_batches

    def count_batches_and_samples(self):
        num_samples = 0
        for block_path in self.data_blocks:
            block_size = np.load(block_path)[self.feature_map.labels[0]].shape[0]
            num_samples += block_size
        num_batches = int(np.ceil(num_samples / self.batch_size))
        return num_batches, num_samples

class RankDataLoader(object):
    def __init__(self, feature_map, stage="both", train_data=None, valid_data=None, test_data=None,
                 batch_size=32, shuffle=True, streaming=False, **kwargs):
        logging.info("Loading datasets...")
        train_gen = None
        valid_gen = None
        test_gen = None
        DataLoader = NpzBlockDataLoader if streaming else NpzDataLoader
        self.stage = stage
        if stage in ["both", "train"]:
            train_gen = DataLoader(feature_map, train_data, batch_size=batch_size, shuffle=shuffle, **kwargs)
            logging.info("Train samples: total/{:d}, blocks/{:d}".format(train_gen.num_samples, train_gen.num_blocks))
            if valid_data:
                valid_gen = DataLoader(feature_map, valid_data, batch_size=batch_size, shuffle=False, **kwargs)
                logging.info(
                    "Validation samples: total/{:d}, blocks/{:d}".format(valid_gen.num_samples, valid_gen.num_blocks))

        if stage in ["both", "test"]:
            if test_data:
                test_gen = DataLoader(feature_map, test_data, batch_size=batch_size, shuffle=False, **kwargs)
                logging.info("Test samples: total/{:d}, blocks/{:d}".format(test_gen.num_samples, test_gen.num_blocks))
        self.train_gen, self.valid_gen, self.test_gen = train_gen, valid_gen, test_gen

    def make_iterator(self):
        if self.stage == "train":
            logging.info("Loading train and validation data done.")
            return self.train_gen, self.valid_gen
        elif self.stage == "test":
            logging.info("Loading test data done.")
            return self.test_gen
        else:
            logging.info("Loading data done.")
            return self.train_gen, self.valid_gen, self.test_gen

class ShuffledDataset(data.IterableDataset):
    def __init__(self, shuffled_data):
        self.shuffled_data = shuffled_data

    def __iter__(self):
        for sample in self.shuffled_data:
            yield sample

class BlockDataPipe(data.IterDataPipe):
    def __init__(self, block_datapipe, feature_map, shuffle=False):
        self.feature_map = feature_map
        self.block_datapipe = block_datapipe
        self.shuffle = shuffle
        self.incomplete_groups = {}

    def load_data(self, data_path):
        data_dict = np.load(data_path)
        data_arrays = []
        all_cols = list(self.feature_map.features.keys()) + self.feature_map.labels
        for col in all_cols:
            array = data_dict[col]
            if array.ndim == 1:
                array = array[:, np.newaxis]
            data_arrays.append(array)
        data_tensor = torch.from_numpy(np.concatenate(data_arrays, axis=1))
        return data_tensor

    def read_block(self, data_block):
        return self.load_data(data_block)

    def merge_incomplete_groups(self, new_data):
        group_id_col_idx = list(self.feature_map.features.keys()).index(self.feature_map.group_id)
        for sample in new_data:
            group_id = sample[group_id_col_idx].item()
            if group_id in self.incomplete_groups:
                self.incomplete_groups[group_id].append(sample)
            else:
                self.incomplete_groups[group_id] = [sample]

        complete_groups = []
        for group_id, samples in list(self.incomplete_groups.items()):
            if len(samples) == 15:  # Assuming group size is fixed as 15
                complete_groups.extend(samples)
                del self.incomplete_groups[group_id]

        return complete_groups

    def grouped_shuffle(self, data_list):
        group_id_col_idx = list(self.feature_map.features.keys()).index(self.feature_map.group_id)

        # Group by group_id
        groups = {}
        for sample in data_list:
            group_id = sample[group_id_col_idx].item()
            if group_id not in groups:
                groups[group_id] = []
            groups[group_id].append(sample)

        # Shuffle groups
        group_ids = list(groups.keys())
        if self.shuffle:
            np.random.shuffle(group_ids)

        # Flatten the shuffled groups
        shuffled_data = [sample for group_id in group_ids for sample in groups[group_id]]

        return ShuffledDataset(shuffled_data)

    def __iter__(self):
        worker_info = data.get_worker_info()
        if worker_info is None:  # single-process data loading
            block_list = self.block_datapipe
        else:  # in a worker process
            block_list = [
                block
                for idx, block in enumerate(self.block_datapipe)
                if idx % worker_info.num_workers == worker_info.id
            ]

        for block in block_list:
            block_data = self.read_block(block)
            complete_groups = self.merge_incomplete_groups(block_data)
            if self.shuffle:
                shuffled_dataset = self.grouped_shuffle(complete_groups)
                yield from shuffled_dataset
            else:
                yield from complete_groups
