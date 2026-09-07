import json
import os
import random
import warnings
from bisect import bisect_right

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def _normalize_interactions(interactions, user_count, item_count):
    result = {}
    for user, items in interactions.items():
        user = int(user)
        if not 1 <= user <= user_count:
            raise ValueError(f"User ID {user} is outside [1, {user_count}].")
        items = [int(item) for item in items]
        if any(not 1 <= item <= item_count for item in items):
            raise ValueError(f"Invalid item ID for user {user}; item 0 is padding.")
        result[str(user)] = list(dict.fromkeys(items))
    return result


def _merge_interactions(interaction_dicts):
    merged = {}
    for interactions in interaction_dicts:
        for user, items in interactions.items():
            merged.setdefault(user, set()).update(items)
    return {user: sorted(items) for user, items in merged.items()}


def _build_behavior_history(behavior_dict, behaviors):
    # DMR, Sec. 3.2 and Appendix A.2.2: target-masked dispatch.
    history = {}
    for index, behavior in enumerate(behaviors[:-1]):
        for user, items in behavior_dict[behavior].items():
            for item in items:
                key = (int(user), item)
                history.setdefault(key, [0] * len(behaviors))[index] = 1
    return history


class _UniformComplement:
    def __init__(self, item_count, excluded):
        excluded = sorted(set(excluded))
        self.size = item_count - len(excluded)
        self.offsets = tuple(item - index - 1 for index, item in enumerate(excluded))

    def sample(self):
        if not self.size:
            raise ValueError("Cannot sample from an empty negative pool.")
        # Sample the complement without rejection or a dense user-item table.
        rank = random.randrange(self.size)
        return rank + 1 + bisect_right(self.offsets, rank)


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


class TestDataset(Dataset):
    def __init__(self, user_samples):
        self.user_samples = list(user_samples)

    def __getitem__(self, idx):
        return int(self.user_samples[idx])

    def __len__(self):
        return len(self.user_samples)


class MultiBehaviorTrainDataset(Dataset):
    def __init__(self, user_count, item_count, behavior_dict=None, behaviors=None,
                 behavior_history=None, test_dict=None, validation_dict=None,
                 target_negative_ratio=0.5):
        self.user_count = user_count
        self.item_count = item_count
        self.behaviors = list(behaviors or [])
        if len(self.behaviors) < 2 or len(set(self.behaviors)) != len(self.behaviors):
            raise ValueError("Provide distinct auxiliary behaviors followed by the target behavior.")
        if not 0.0 <= target_negative_ratio <= 1.0:
            raise ValueError("target_negative_ratio must be in [0, 1].")
        self.target_negative_ratio = target_negative_ratio
        self.n_behaviors = len(self.behaviors)
        self.target_behavior = self.behaviors[-1]
        self.behavior_dict = {
            behavior: _normalize_interactions(behavior_dict[behavior], user_count, item_count)
            for behavior in self.behaviors
        }
        self.test_dict = _normalize_interactions(test_dict or {}, user_count, item_count)
        self.validation_dict = _normalize_interactions(validation_dict or {}, user_count, item_count)
        held_out = _merge_interactions([self.validation_dict, self.test_dict])
        for user, items in self.behavior_dict[self.target_behavior].items():
            if set(items).intersection(held_out.get(user, [])):
                raise ValueError(f"Target training and held-out positives overlap for user {user}.")
        self.behavior_dict['all'] = _merge_interactions(
            self.behavior_dict[behavior] for behavior in self.behaviors
        )
        self.behavior_history = (
            behavior_history if behavior_history is not None
            else _build_behavior_history(self.behavior_dict, self.behaviors)
        )
        self.aux_negative_pools = {}
        self.zero_negative_pools = {}
        self.users = []
        self.unsampleable_positive_count = 0

        # Sec. 3.4 and Appendix B.1.2: mixed target negatives.
        for user, interacted in self.behavior_dict['all'].items():
            target_positives = set(self.behavior_dict[self.target_behavior].get(user, []))
            target_positives.update(held_out.get(user, []))
            auxiliary_items = set()
            for behavior in self.behaviors[:-1]:
                auxiliary_items.update(self.behavior_dict[behavior].get(user, []))
            aux_pool = tuple(sorted(auxiliary_items - target_positives))
            zero_pool = _UniformComplement(item_count, auxiliary_items | target_positives)
            self.aux_negative_pools[user] = aux_pool
            self.zero_negative_pools[user] = zero_pool
            has_pair = False

            for behavior in self.behaviors:
                positives = self.behavior_dict[behavior].get(user, [])
                has_negative = zero_pool.size > 0 or (
                        behavior == self.target_behavior and bool(aux_pool)
                )

                has_pair |= bool(positives) and has_negative

                if positives and not has_negative:
                    self.unsampleable_positive_count += len(positives)

            if has_pair:
                self.users.append(user)
        if self.unsampleable_positive_count:
            warnings.warn(
                f"{self.unsampleable_positive_count} positive interactions have no valid "
                "negative pool; their training slots are skipped.", RuntimeWarning
            )

    def _sample_negative(self, user, behavior):
        zero_pool = self.zero_negative_pools[user]
        if behavior == self.target_behavior:
            aux_pool = self.aux_negative_pools[user]
            if aux_pool and (not zero_pool.size or random.random() < self.target_negative_ratio):
                return random.choice(aux_pool)
        return zero_pool.sample() if zero_pool.size else 0

    def _history(self, user, item):
        history = list(self.behavior_history.get((user, item), [0] * self.n_behaviors))
        history[-1] = 0
        return history

    def __getitem__(self, idx):
        user = self.users[idx]
        user_id = int(user)

        positives = []
        negatives = []
        positive_histories = []
        negative_histories = []

        for behavior in self.behaviors:
            items = self.behavior_dict[behavior].get(user, [])

            negative = (
                self._sample_negative(user, behavior)
                if items else 0
            )
            positive = random.choice(items) if negative else 0

            positives.append(positive)
            negatives.append(negative)
            positive_histories.append(
                self._history(user_id, positive)
            )
            negative_histories.append(
                self._history(user_id, negative)
            )

        return (
            torch.tensor(user_id, dtype=torch.long),
            torch.tensor(positives, dtype=torch.long),
            torch.tensor(negatives, dtype=torch.long),
            torch.tensor(positive_histories, dtype=torch.float32),
            torch.tensor(negative_histories, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.users)


def custom_collate_fn(batch):
    return tuple(torch.stack(values) for values in zip(*batch))


class DataSet:
    def __init__(self, args):
        self.args = args
        self.behaviors = list(args.behaviors)
        if len(self.behaviors) < 2 or len(set(self.behaviors)) != len(self.behaviors):
            raise ValueError("Provide distinct auxiliary behaviors followed by the target behavior.")
        self.target_behavior = self.behaviors[-1]
        self.path = args.data_path
        self.__get_count()
        self.validation_interacts = self._read_interactions('validation_dict.txt')
        self.test_interacts = self._read_interactions('test_dict.txt')
        self.__get_behavior_items()
        self.__get_sparse_interact_dict()
        self.behavior_history = _build_behavior_history(self.train_behavior_dict, self.behaviors)
        self.__check_auxiliary_coverage()
        self.val_users = [user for user, items in self.validation_interacts.items() if items]
        self.test_users = [user for user, items in self.test_interacts.items() if items]
        self.validation_gt_length = np.array(
            [len(self.validation_interacts[user]) for user in self.val_users], dtype=np.int64
        )
        self.test_gt_length = np.array(
            [len(self.test_interacts[user]) for user in self.test_users], dtype=np.int64
        )

    def __get_count(self):
        with open(os.path.join(self.path, 'count.txt'), encoding='utf-8') as stream:
            count = json.load(stream)
        self.user_count = int(count['user'])
        self.item_count = int(count['item'])
        if self.user_count < 1 or self.item_count < 1:
            raise ValueError("User and item counts must be positive.")

    def _read_interactions(self, filename):
        with open(os.path.join(self.path, filename), encoding='utf-8') as stream:
            return _normalize_interactions(json.load(stream), self.user_count, self.item_count)

    def __get_behavior_items(self):
        self.train_behavior_dict = {
            behavior: self._read_interactions(behavior + '_dict.txt')
            for behavior in self.behaviors
        }
        for user, items in self.validation_interacts.items():
            if set(items).intersection(self.test_interacts.get(user, [])):
                raise ValueError(f"Validation and test target positives overlap for user {user}.")
        held_out = _merge_interactions([self.validation_interacts, self.test_interacts])
        removed = 0
        # MID, Sec. 3.1 and Appendix B.1.2: keep only target-training edges.
        for user, items in self.train_behavior_dict[self.target_behavior].items():
            excluded = set(held_out.get(user, []))
            retained = [item for item in items if item not in excluded]
            removed += len(items) - len(retained)
            self.train_behavior_dict[self.target_behavior][user] = retained
        if removed:
            warnings.warn(f"Removed {removed} held-out target edges from training inputs.", RuntimeWarning)
        self.train_behavior_dict['all'] = _merge_interactions(
            self.train_behavior_dict[behavior] for behavior in self.behaviors
        )

    def __check_auxiliary_coverage(self):
        target_positives = _merge_interactions([
            self.train_behavior_dict[self.target_behavior],
            self.validation_interacts, self.test_interacts,
        ])
        self.target_positives_without_aux = sum(
            (int(user), item) not in self.behavior_history
            for user, items in target_positives.items() for item in items
        )
        if self.target_positives_without_aux:
            warnings.warn(
                f"{self.target_positives_without_aux} target positives have no auxiliary evidence; "
                "the data do not satisfy the coverage assumption in Appendix B.1.2.", RuntimeWarning
            )

    def _edge_index(self, interactions):
        users, items = [], []
        for user, user_items in interactions.items():
            users.extend([int(user)] * len(user_items))
            items.extend(user_items)
        user_nodes = torch.tensor(users, dtype=torch.long)
        item_nodes = torch.tensor(items, dtype=torch.long) + self.user_count + 1
        return torch.stack([
            torch.cat([user_nodes, item_nodes]), torch.cat([item_nodes, user_nodes])
        ])

    def __get_sparse_interact_dict(self):
        self.edge_index = {
            behavior: self._edge_index(self.train_behavior_dict[behavior])
            for behavior in self.behaviors
        }
        self.all_edge_index = self._edge_index(self.train_behavior_dict['all'])

    def get_behavior_histories(self, user_ids, item_ids):
        users, items = torch.broadcast_tensors(
            torch.as_tensor(user_ids, dtype=torch.long).detach().cpu(),
            torch.as_tensor(item_ids, dtype=torch.long).detach().cpu(),
        )
        histories = [
            self.behavior_history.get((user, item), [0] * len(self.behaviors))
            for user, item in zip(users.reshape(-1).tolist(), items.reshape(-1).tolist())
        ]
        return torch.tensor(histories, dtype=torch.float32).reshape(*users.shape, len(self.behaviors))

    def get_train_loader(self, batch_size):
        dataset = MultiBehaviorTrainDataset(
            self.user_count, self.item_count, self.train_behavior_dict,
            self.behaviors, self.behavior_history, self.test_interacts,
            self.validation_interacts, getattr(self.args, 'target_negative_ratio', 0.5),
        )
        if not len(dataset):
            raise ValueError("No training user has a valid positive-negative pair.")
        generator = torch.Generator().manual_seed(getattr(self.args, 'seed', 2020))
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            num_workers=getattr(self.args, 'num_workers', 0),
            collate_fn=custom_collate_fn, worker_init_fn=seed_worker, generator=generator,
        )

    def test_dataset(self):
        return TestDataset(self.test_users)

    def get_val_loader(self, batch_size=None):
        return DataLoader(
            TestDataset(self.val_users), batch_size=batch_size or self.args.test_batch_size,
            shuffle=False,
        )

    def get_test_loader(self, batch_size=None):
        return DataLoader(
            self.test_dataset(), batch_size=batch_size or self.args.test_batch_size,
            shuffle=False,
        )
