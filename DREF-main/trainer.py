import math
import os
from pathlib import Path

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_setp import TestDataset


class Trainer:
    def __init__(self, model, dataset, args):
        self.args = args
        self.device = torch.device(args.device)
        self.model = model.to(self.device)
        self.dataset = dataset
        self.behaviors = list(args.behaviors)
        self.topk = sorted(set(int(k) for k in args.topk))
        self.metrics = list(dict.fromkeys(
            'hit' if name.lower() == 'hr' else name.lower() for name in args.metrics
        ))
        if not self.topk or self.topk[0] < 1:
            raise ValueError("topk must contain positive integers.")
        if not self.metrics or set(self.metrics) - {'hit', 'ndcg'}:
            raise ValueError("The paper evaluation supports hit/HR and NDCG.")
        self.learning_rate = float(args.lr)
        self.batch_size = int(args.batch_size)
        self.test_batch_size = int(args.test_batch_size)
        self.eval_pair_batch_size = int(getattr(args, 'eval_pair_batch_size', 4096))
        self.epochs = int(getattr(args, 'epochs', 200))
        self.patience = int(getattr(args, 'patience', 0))
        self.grad_clip = float(getattr(args, 'grad_clip', 0.0))
        self.show_progress = bool(getattr(args, 'show_progress', True))
        if min(self.batch_size, self.test_batch_size, self.eval_pair_batch_size) < 1:
            raise ValueError("Batch sizes must be positive.")
        if not 1 <= self.epochs <= 200:
            raise ValueError("Appendix B.1.2 specifies at most 200 epochs.")
        if self.patience < 0 or self.grad_clip < 0:
            raise ValueError("patience and grad_clip must be nonnegative.")
        default_k = 10 if 10 in self.topk else self.topk[0]
        default_metric = 'ndcg' if 'ndcg' in self.metrics else 'hit'
        self.selection_metric = getattr(args, 'selection_metric', f'{default_metric}@{default_k}').lower()
        if self.selection_metric.startswith('hr@'):
            self.selection_metric = 'hit@' + self.selection_metric.split('@')[1]
        metric_keys = {f'{metric}@{k}' for metric in self.metrics for k in self.topk}
        if self.selection_metric not in metric_keys:
            raise ValueError("selection_metric must be one of the reported metrics.")
        for method in ('update_faiss_indexes', 'score_candidates'):
            if not callable(getattr(self.model, method, None)):
                raise TypeError(f"The DREF model must implement {method}().")
        seed = int(getattr(args, 'seed', 2020))
        model_name = getattr(args, 'model_name', 'DREF')
        self.checkpoint_path = Path(args.model_path) / f'{model_name}_seed{seed}_best.pt'
        self.optimizer = self.get_optimizer(self.model)
        self.best_epoch = None
        self.best_validation = None
        self.history = []

    def get_optimizer(self, model):
        # Eq. (10): L2 belongs to the model objective, without duplicate weight decay.
        return optim.Adam(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=self.learning_rate, weight_decay=0.0,
        )

    @torch.no_grad()
    def _refresh_indexes(self):
        # Appendix B.1.3: refresh indices and inference caches from current weights.
        was_training = self.model.training
        self.model.eval()
        try:
            self.model.update_faiss_indexes()
        finally:
            self.model.train(was_training)

    def train_model(self):
        if not self.dataset.val_users or not self.dataset.test_users:
            raise ValueError("Nonempty validation and test splits are required.")
        train_loader = self.dataset.get_train_loader(self.batch_size)
        self.best_epoch = None
        self.best_validation = None
        self.history = []
        best_value = -math.inf
        stale_epochs = 0
        for epoch in range(1, self.epochs + 1):
            loss = self._train_one_epoch(train_loader, epoch)
            validation = self.evaluate('validation')
            value = validation[self.selection_metric]
            self.history.append({
                'epoch': epoch, 'mean_batch_loss': loss, 'validation': validation,
            })
            if value > best_value:
                best_value = value
                self.best_epoch = epoch
                self.best_validation = dict(validation)
                self.save_model()
                stale_epochs = 0
            else:
                stale_epochs += 1
            tqdm.write(f"Epoch {epoch}: mean batch loss {loss:.6f}; validation {validation}")
            if self.patience and stale_epochs >= self.patience:
                break

        # Appendix B.1.2: select on validation, then evaluate the selected model on test.
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()
        test_metrics = self.evaluate('test')
        tqdm.write(f"Best epoch {self.best_epoch}; test {test_metrics}")
        return {
            'best_epoch': self.best_epoch,
            'validation': self.best_validation,
            'test': test_metrics,
            'checkpoint': str(self.checkpoint_path),
            'history': self.history,
        }

    def _train_one_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0.0
        batches = 0
        for batch in tqdm(train_loader, desc=f'Train {epoch}', disable=not self.show_progress):
            if len(batch) != 5:
                raise ValueError("Expected users, positives, negatives, positive histories, negative histories.")
            if batch[3][..., -1].any() or batch[4][..., -1].any():
                raise ValueError("Target behavior must be masked in both routing histories.")
            batch = tuple(value.to(self.device) for value in batch)
            self.optimizer.zero_grad(set_to_none=True)
            # Sec. 3.4, Eq. (10): the model returns the complete training objective.
            loss = self.model(batch)
            if not torch.is_tensor(loss) or loss.ndim != 0 or not torch.isfinite(loss).item():
                raise ValueError("The model must return one finite scalar training loss.")
            loss.backward()
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip, error_if_nonfinite=True,
                )
            self.optimizer.step()
            total_loss += loss.detach().item()
            batches += 1
        if not batches:
            raise ValueError("The training loader is empty.")
        return total_loss / batches

    def save_model(self):
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            'model_state': {key: value.detach().cpu().clone()
                            for key, value in self.model.state_dict().items()},
            'epoch': self.best_epoch,
            'validation': self.best_validation,
            'selection_metric': self.selection_metric,
            'behaviors': self.behaviors,
            'seed': int(getattr(self.args, 'seed', 2020)),
        }
        temporary = self.checkpoint_path.with_suffix('.pt.tmp')
        torch.save(checkpoint, temporary)
        os.replace(temporary, self.checkpoint_path)

    @torch.no_grad()
    def _rank_users(self, users, ground_truth):
        users = users.detach().cpu()
        batch_size = len(users)
        chunk_size = max(1, self.eval_pair_batch_size // batch_size)
        best_scores = torch.empty((batch_size, 0), device=self.device)
        best_items = torch.empty(
            (batch_size, 0), dtype=torch.long, device=self.device
        )

        train_items = self.dataset.train_behavior_dict[self.behaviors[-1]]
        validation_items = self.dataset.validation_interacts
        test_items = self.dataset.test_interacts

        excluded = []
        for user in users.tolist():
            user_key = str(user)

            # Rank against the full catalog while filtering every other
            # known target-positive item.
            known_positives = set(train_items.get(user_key, []))
            known_positives.update(validation_items.get(user_key, []))
            known_positives.update(test_items.get(user_key, []))

            # The ground-truth items for the current split must remain rankable.
            known_positives.difference_update(
                ground_truth.get(user_key, [])
            )
            excluded.append(known_positives)

        # DMR, Eq. (9): score every real item under its natural configuration.
        for start in range(1, self.dataset.item_count + 1, chunk_size):
            items = torch.arange(start, min(start + chunk_size, self.dataset.item_count + 1))
            histories = self.dataset.get_behavior_histories(users[:, None], items[None, :])
            if histories[..., -1].any():
                raise ValueError("Evaluation routing histories must be target-masked.")
            pair_users = users[:, None].expand(-1, len(items)).reshape(-1).to(self.device)
            pair_items = items[None, :].expand(batch_size, -1).reshape(-1).to(self.device)
            scores = self.model.score_candidates(
                pair_users, pair_items, histories.reshape(-1, len(self.behaviors)).to(self.device),
            )
            if (not torch.is_tensor(scores) or scores.shape != pair_items.shape
                    or not scores.is_floating_point() or not torch.isfinite(scores).all().item()):
                raise ValueError("score_candidates must return one finite score per candidate pair.")
            scores = scores.to(self.device).reshape(batch_size, -1).clone()
            for row, user_excluded in enumerate(excluded):
                local = [item - start for item in user_excluded if start <= item < start + len(items)]
                if local:
                    scores[row, local] = -torch.inf
            item_grid = items.to(self.device)[None, :].expand(batch_size, -1)
            merged_scores = torch.cat([best_scores, scores], dim=1)
            merged_items = torch.cat([best_items, item_grid], dim=1)
            order = torch.argsort(merged_scores, dim=1, descending=True, stable=True)[:, :self.topk[-1]]
            best_scores = merged_scores.gather(1, order)
            best_items = merged_items.gather(1, order)
        return best_items.masked_fill(~torch.isfinite(best_scores), 0).cpu().numpy()

    @torch.no_grad()
    def evaluate(self, split='validation'):
        if split == 'validation':
            users, ground_truth = self.dataset.val_users, self.dataset.validation_interacts
        elif split == 'test':
            users, ground_truth = self.dataset.test_users, self.dataset.test_interacts
        else:
            raise ValueError("split must be 'validation' or 'test'.")
        if not users:
            raise ValueError(f"The {split} split has no evaluation users.")
        was_training = self.model.training
        self.model.eval()
        try:
            self._refresh_indexes()
            loader = DataLoader(
                TestDataset(users), batch_size=min(self.test_batch_size, self.eval_pair_batch_size),
                shuffle=False,
            )
            totals = {f'{metric}@{k}': 0.0 for metric in self.metrics for k in self.topk}
            user_count = 0
            train_items = self.dataset.train_behavior_dict[self.behaviors[-1]]
            for batch_users in tqdm(loader, desc=split.title(), disable=not self.show_progress):
                labels = [ground_truth[str(user)] for user in batch_users.tolist()]
                if any(not items for items in labels):
                    raise ValueError("Each evaluation user must have a positive ground truth.")
                for user, items in zip(batch_users.tolist(), labels):
                    if set(items).intersection(train_items.get(str(user), [])):
                        raise ValueError("Evaluation positives must not appear in target training edges.")
                ranked_items = self._rank_users(
                    batch_users,
                    ground_truth,
                )
                hits = np.zeros((len(batch_users), self.topk[-1]), dtype=bool)
                for row, items in enumerate(labels):
                    hits[row, :ranked_items.shape[1]] = np.isin(ranked_items[row], items)
                lengths = np.array([len(set(items)) for items in labels], dtype=np.int64)
                values = self.calculate_result(hits, lengths)
                for key, value in values.items():
                    totals[key] += value * len(batch_users)
                user_count += len(batch_users)
            return {key: value / user_count for key, value in totals.items()}
        finally:
            self.model.train(was_training)

    def calculate_result(self, topk_list, gt_len):
        hits = np.asarray(topk_list, dtype=bool)
        lengths = np.asarray(gt_len, dtype=np.int64)
        if hits.ndim != 2 or hits.shape[0] == 0 or hits.shape[1] < self.topk[-1]:
            raise ValueError("Expected nonempty hit rows through max(topk).")
        if lengths.shape != (hits.shape[0],) or np.any(lengths <= 0):
            raise ValueError("Ground-truth lengths must match the evaluated user order.")
        discounts = 1.0 / np.log2(np.arange(2, hits.shape[1] + 2))
        dcg = np.cumsum(hits * discounts, axis=1)
        ideal_dcg = np.cumsum(discounts)
        result = {}
        for k in self.topk:
            if 'hit' in self.metrics:
                result[f'hit@{k}'] = float(hits[:, :k].any(axis=1).mean())
            if 'ndcg' in self.metrics:
                denominator = ideal_dcg[np.minimum(lengths, k) - 1]
                result[f'ndcg@{k}'] = float((dcg[:, k - 1] / denominator).mean())
        return result
