import ast
import weakref
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class MIDState:
    user_c: list
    item_c: list
    user_s: list
    item_s: list
    user: list
    item: list
    gates: list


# MID: LightGCN, Sec. 3.1.1, Eq. (1).
class LightGCN(nn.Module):
    def __init__(self, edge_index, node_count, depth):
        super().__init__()
        self.depth = depth
        edges = torch.unique(edge_index.long(), dim=1)
        degree = torch.bincount(edges[0], minlength=node_count).float()
        inverse = degree.clamp_min(1).rsqrt()
        values = inverse[edges[0]] * inverse[edges[1]]
        adjacency = torch.sparse_coo_tensor(edges.flip(0), values, (node_count, node_count)).coalesce()
        self.register_buffer('adjacency', adjacency, persistent=False)

    def forward(self, embeddings):
        current = embeddings
        output = embeddings
        for _ in range(self.depth):
            current = torch.sparse.mm(self.adjacency, current)
            output = output + current
        return output


# MID: separate user/item hypergraphs, Sec. 3.1.1, Eq. (2).
class HypergraphView(nn.Module):
    def __init__(self, dim, hyperedges):
        super().__init__()
        self.assignment = nn.Parameter(torch.empty(dim, hyperedges))
        nn.init.xavier_uniform_(self.assignment)

    def forward(self, collaborative):
        real = collaborative[1:]
        incidence = torch.softmax(real @ self.assignment, dim=-1)
        semantic = incidence @ (incidence.T @ real)
        return torch.cat([semantic.new_zeros(1, semantic.shape[1]), semantic], dim=0)


# MID: Interaction-Aware Gated Filtration, Sec. 3.1.2, Eq. (3).
class InteractionAwareGatedFiltration(nn.Module):
    def __init__(self, dim, has_predecessor):
        super().__init__()
        self.balance_gate = nn.Linear(3 * dim, dim)
        self.propagation_gate = nn.Linear(2 * dim, dim) if has_predecessor else None
        nn.init.zeros_(self.balance_gate.bias)
        if self.propagation_gate is not None:
            nn.init.zeros_(self.propagation_gate.bias)

    def forward(self, collaborative, semantic, previous=None):
        interaction = torch.cat([collaborative, semantic, collaborative * semantic], dim=-1)
        z = torch.sigmoid(self.balance_gate(interaction))
        balanced = z * collaborative + (1 - z) * semantic
        f = None
        output = balanced
        if previous is not None:
            f = torch.sigmoid(self.propagation_gate(torch.cat([balanced, previous], dim=-1)))
            output = f * previous + (1 - f) * balanced
        return output, {'z': z, 'f': f, 'balanced': balanced}


# DMR: four concatenated attention heads, Sec. 3.2.1, Eq. (6).
class SparseGraphAttention(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        if dim % heads:
            raise ValueError("The embedding dimension must be divisible by four GAT heads.")
        self.heads = heads
        self.head_dim = dim // heads
        self.projection = nn.Linear(dim, dim, bias=False)
        self.attention = nn.Parameter(torch.empty(heads, 2 * self.head_dim))
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.xavier_uniform_(self.attention)

    def forward(self, nodes, edge_index):
        source, destination = edge_index
        projected = self.projection(nodes).reshape(-1, self.heads, self.head_dim)
        receiver = (projected * self.attention[:, :self.head_dim]).sum(-1)
        sender = (projected * self.attention[:, self.head_dim:]).sum(-1)
        logits = F.leaky_relu(receiver[destination] + sender[source], negative_slope=0.2)
        maxima = logits.new_full((len(nodes), self.heads), -torch.inf)
        maxima.scatter_reduce_(
            0, destination[:, None].expand_as(logits), logits.detach(), reduce='amax', include_self=True,
        )
        numerators = torch.exp(logits - maxima[destination])
        denominators = logits.new_zeros((len(nodes), self.heads)).index_add(0, destination, numerators)
        alpha = numerators / denominators[destination]
        messages = projected[source] * alpha.unsqueeze(-1)
        output = projected.new_zeros(projected.shape).index_add(0, destination, messages)
        return output.reshape(len(nodes), -1), alpha


# DMR: Progressive Neural Execution, Sec. 3.2.1-3.2.2, Eq. (6).
class GraphExecutor(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layers = nn.ModuleList([SparseGraphAttention(dim), SparseGraphAttention(dim)])

    def forward(self, nodes, edge_index, return_trace=False):
        batch_size, node_count, dim = nodes.shape
        flat = nodes.reshape(-1, dim)
        states, attentions = ([nodes], []) if return_trace else (None, None)
        for index, layer in enumerate(self.layers):
            flat, alpha = layer(flat, edge_index)
            if index == 0:
                flat = F.elu(flat)
            if return_trace:
                states.append(flat.reshape(batch_size, node_count, dim))
                attentions.append(alpha)
        candidate = flat.reshape(batch_size, node_count, dim)[:, 1]
        return candidate, {'states': states, 'attention': attentions} if return_trace else None


@torch.no_grad()
def context_knn(contexts, degree):
    count = contexts.shape[1]
    degree = min(degree, max(0, count - 1))
    if degree == 0:
        return torch.empty((*contexts.shape[:2], 0), dtype=torch.long, device=contexts.device)
    similarity = contexts @ contexts.transpose(1, 2)
    diagonal = torch.arange(count, device=contexts.device)
    similarity[:, diagonal, diagonal] = -torch.inf
    return similarity.topk(degree, dim=-1).indices


# DMR: user/candidate stars and symmetric context KNN, Eq. (5).
def local_logic_edges(knn, extra_nodes=0):
    batch_size, context_count, degree = knn.shape
    device = knn.device
    node_count = context_count + 2 + extra_nodes
    contexts = torch.arange(2, context_count + 2, device=device)
    source = torch.cat([
        torch.tensor([0, 1], device=device),
        torch.zeros_like(contexts), contexts, torch.ones_like(contexts), contexts,
    ])
    destination = torch.cat([
        torch.tensor([1, 0], device=device), contexts, torch.zeros_like(contexts),
        contexts, torch.ones_like(contexts),
    ])
    if extra_nodes:
        anchors = torch.arange(context_count + 2, node_count, device=device)
        source = torch.cat([source, anchors, torch.ones_like(anchors)])
        destination = torch.cat([destination, torch.ones_like(anchors), anchors])
    nodes = torch.arange(node_count, device=device)
    source = torch.cat([source, nodes]).expand(batch_size, -1)
    destination = torch.cat([destination, nodes]).expand(batch_size, -1)
    if degree:
        context_source = contexts[None, :, None].expand(batch_size, -1, degree).reshape(batch_size, -1)
        context_destination = (knn + 2).reshape(batch_size, -1)
        source = torch.cat([source, context_source, context_destination], dim=1)
        destination = torch.cat([destination, context_destination, context_source], dim=1)
    offsets = torch.arange(batch_size, device=device)[:, None] * node_count
    edges = torch.stack([(source + offsets).reshape(-1), (destination + offsets).reshape(-1)])
    return torch.unique(edges, dim=1)


# DMR: exact inner-product Top-R retrieval, Eq. (4).
@torch.no_grad()
def retrieve_contexts(item_embeddings, candidates, count, query_batch_size=128, item_chunk_size=4096):
    item_count = len(item_embeddings) - 1
    count = min(count, item_count - 1)
    rows = []
    for queries in candidates.split(query_batch_size):
        best_scores = item_embeddings.new_empty((len(queries), 0))
        best_ids = candidates.new_empty((len(queries), 0))
        for start in range(1, item_count + 1, item_chunk_size):
            end = min(start + item_chunk_size, item_count + 1)
            ids = torch.arange(start, end, device=item_embeddings.device)
            scores = item_embeddings[queries] @ item_embeddings[start:end].T
            scores.masked_fill_(queries[:, None] == ids[None, :], -torch.inf)
            joined_scores = torch.cat([best_scores, scores], dim=1)
            joined_ids = torch.cat([best_ids, ids[None, :].expand(len(queries), -1)], dim=1)
            best_scores, positions = joined_scores.topk(min(count, joined_scores.shape[1]), dim=1)
            best_ids = joined_ids.gather(1, positions)
        rows.append(best_ids)
    return torch.cat(rows) if rows else candidates.new_empty((0, count))


# DREF: MID, DMR, and Dual Regularization, Sec. 3.
class Model(nn.Module):
    def __init__(self, args, dataset):
        super().__init__()
        self.behaviors = list(args.behaviors)
        self.n_behaviors = len(self.behaviors)
        self.n_users, self.n_items = dataset.user_count, dataset.item_count
        self.dim = int(getattr(args, 'embedding_size', 64))
        depths = getattr(args, 'layers_nums', None)
        if isinstance(depths, str):
            depths = ast.literal_eval(depths)
        if depths is None:
            depths = [1] * (self.n_behaviors - 1) + [3]
        self.depths = list(depths)
        if (self.n_behaviors < 2 or len(set(self.behaviors)) != self.n_behaviors
                or self.behaviors != list(dataset.behaviors)):
            raise ValueError("Model and dataset must use the same ordered behavior chain.")
        if len(self.depths) != self.n_behaviors or any(not isinstance(x, int) or x < 0 for x in self.depths):
            raise ValueError("layers_nums must specify a nonnegative depth for each behavior.")
        if self.dim < 4 or self.dim % 4 or self.n_items < 2:
            raise ValueError("Require dim divisible by four and at least two real items.")
        self.context_size = int(getattr(args, 'context_size', 20))
        self.knn_degree = int(getattr(args, 'knn_degree', 5))
        self.retrieval_query_batch_size = int(getattr(args, 'retrieval_query_batch_size', 128))
        self.retrieval_item_chunk_size = int(getattr(args, 'retrieval_item_chunk_size', 4096))
        self.max_context_cache_entries = int(getattr(args, 'max_context_cache_entries', 100000))
        self.retrieval_backend = getattr(args, 'retrieval_backend', 'faiss_flat')
        if self.retrieval_backend not in {'faiss_flat', 'faiss_hnsw', 'torch'}:
            raise ValueError("retrieval_backend must be faiss_flat, faiss_hnsw, or torch.")
        self.faiss_hnsw_m = int(getattr(args, 'faiss_hnsw_m', 32))
        self.faiss_ef_construction = int(getattr(args, 'faiss_ef_construction', 200))
        self.faiss_ef_search = int(getattr(args, 'faiss_ef_search', 64))
        if min(self.faiss_hnsw_m, self.faiss_ef_construction, self.faiss_ef_search) < 1:
            raise ValueError("FAISS HNSW parameters must be positive.")
        self._faiss = None
        if self.retrieval_backend != 'torch':
            try:
                import faiss
            except ImportError as error:
                raise ImportError("FAISS retrieval requires faiss-cpu in the active Python environment. "
                                  "Set retrieval_backend='torch' only for explicit exact-reference runs.") from error
            self._faiss = faiss
        self.faiss_indexes_gcn = {}
        self.faiss_indexes_hyper = {}
        self._indexed_banks = {}
        hyperedges = int(getattr(args, 'hyper_nums', 32))
        if min(self.context_size, hyperedges, self.retrieval_query_batch_size, self.retrieval_item_chunk_size) < 1:
            raise ValueError("Retrieval batch sizes, context size and hyperedge count must be positive.")
        if self.knn_degree < 0 or self.max_context_cache_entries < 0:
            raise ValueError("KNN degree and cache capacity must be nonnegative.")
        self.lambda_r = float(getattr(args, 'lambda_r', 0.01))
        self.lambda_s = float(getattr(args, 'lambda_s', 0.05))
        self.lambda_cal = float(getattr(args, 'lambda_cal', 1.0))
        self.lambda_2 = float(getattr(args, 'lambda_2', getattr(args, 'reg_weight', 1e-4)))
        if any(not 0 <= weight < float('inf') for weight in
               [self.lambda_r, self.lambda_s, self.lambda_cal, self.lambda_2]):
            raise ValueError("Loss weights must be finite and nonnegative.")

        self.user_embedding = nn.Embedding(self.n_users + 1, self.dim, padding_idx=0)
        self.item_embedding = nn.Embedding(self.n_items + 1, self.dim, padding_idx=0)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        with torch.no_grad():
            self.user_embedding.weight[0].zero_()
            self.item_embedding.weight[0].zero_()
        node_count = self.n_users + self.n_items + 2
        self.collaborative_encoders = nn.ModuleList([
            LightGCN(dataset.edge_index[behavior], node_count, self.depths[index])
            for index, behavior in enumerate(self.behaviors)
        ])
        self.user_hypergraphs = nn.ModuleList([HypergraphView(self.dim, hyperedges) for _ in self.behaviors])
        self.item_hypergraphs = nn.ModuleList([HypergraphView(self.dim, hyperedges) for _ in self.behaviors])
        self.user_igf = nn.ModuleList([
            InteractionAwareGatedFiltration(self.dim, index > 0) for index in range(self.n_behaviors)
        ])
        self.item_igf = nn.ModuleList([
            InteractionAwareGatedFiltration(self.dim, index > 0) for index in range(self.n_behaviors)
        ])
        self.user_transitions = nn.ModuleList([
            nn.Linear(self.dim, self.dim, bias=False) for _ in self.behaviors[:-1]
        ])
        self.item_transitions = nn.ModuleList([
            nn.Linear(self.dim, self.dim, bias=False) for _ in self.behaviors[:-1]
        ])
        self.gat_c = GraphExecutor(self.dim)
        self.gat_s = GraphExecutor(self.dim)
        self.e_T = nn.Parameter(F.normalize(torch.randn(self.dim), dim=0))
        self._inference_state = None
        self._inference_signature = None
        self._context_cache = OrderedDict()
        self.last_losses = {}

    @property
    def device(self):
        return self.user_embedding.weight.device

    def encode(self):
        state = MIDState([], [], [], [], [], [], [])
        user, item = self.user_embedding.weight, self.item_embedding.weight
        previous_user = previous_item = None
        for index in range(self.n_behaviors):
            if index:
                # MID: inter-behavior cascading initialization, Sec. 3.1.3.
                user = self.user_transitions[index - 1](previous_user)
                item = self.item_transitions[index - 1](previous_item)
            collaborative = self.collaborative_encoders[index](torch.cat([user, item]))
            user_c, item_c = collaborative.split([self.n_users + 1, self.n_items + 1])
            user_s = self.user_hypergraphs[index](user_c)
            item_s = self.item_hypergraphs[index](item_c)
            user, user_gates = self.user_igf[index](user_c, user_s, previous_user)
            item, item_gates = self.item_igf[index](item_c, item_s, previous_item)
            for name, value in [('user_c', user_c), ('item_c', item_c), ('user_s', user_s),
                                ('item_s', item_s), ('user', user), ('item', item)]:
                getattr(state, name).append(value)
            state.gates.append({'user': user_gates, 'item': item_gates})
            previous_user, previous_item = user, item
        return state

    # DMR: fixed dispatcher, Sec. 3.2 and Appendix A.2.2.
    def dispatch(self, histories, allow_strong=False):
        if histories.ndim != 2 or histories.shape[1] != self.n_behaviors:
            raise ValueError("histories must have shape [pairs, behaviors].")
        if not ((histories == 0) | (histories == 1)).all().item():
            raise ValueError("Dispatcher inputs must be binary behavior-presence indicators.")
        target_observed = histories[:, -1].bool()
        if target_observed.any().item() and not allow_strong:
            raise ValueError("Target prediction requires target-masked histories.")
        observed = histories[:, :-1].bool()
        counts = observed.sum(-1)
        levels = torch.arange(self.n_behaviors - 1, device=histories.device)
        downstream = torch.where(observed, levels, -1).amax(dim=1)
        # Configurations: 0=fallback, 1+b=collaborative, 1+T+b=semantic, -1=Strong.
        configs = torch.zeros(len(histories), dtype=torch.long, device=histories.device)
        configs = torch.where(counts >= 2, 1 + downstream, configs)
        configs = torch.where(counts == 1, 1 + self.n_behaviors + downstream, configs)
        return torch.where(target_observed, -1, configs)

    def _configuration(self, config):
        if config == 0:
            return 's', self.n_behaviors - 1
        if 1 <= config <= self.n_behaviors:
            return 'c', config - 1
        if self.n_behaviors < config <= 2 * self.n_behaviors:
            return 's', config - self.n_behaviors - 1
        raise ValueError(f"Invalid dynamic configuration {config}.")

    # DMR: raw inner-product indices excluding padding, Eq. (4), Appendix B.1.3.
    @torch.no_grad()
    def _ensure_faiss_index(self, bank, space, behavior):
        key = (space, behavior)
        registry = self.faiss_indexes_gcn if space == 'c' else self.faiss_indexes_hyper
        name = self.behaviors[behavior]
        previous = self._indexed_banks.get(key)
        if previous is not None and previous[0]() is bank and previous[1] == bank._version:
            return registry[name]
        vectors = bank[1:].detach().to(device='cpu', dtype=torch.float32).contiguous().numpy()
        if not np.isfinite(vectors).all():
            raise FloatingPointError("FAISS item embeddings must be finite.")
        if self.retrieval_backend == 'faiss_flat':
            index = self._faiss.IndexFlatIP(self.dim)
        else:
            index = self._faiss.IndexHNSWFlat(self.dim, self.faiss_hnsw_m, self._faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = self.faiss_ef_construction
            index.hnsw.efSearch = max(self.faiss_ef_search, min(self.context_size + 1, self.n_items))
        index.add(vectors)
        registry[name] = index
        self._indexed_banks[key] = (weakref.ref(bank), bank._version)
        return index

    @torch.no_grad()
    def _retrieve(self, bank, space, behavior, items):
        if self.retrieval_backend == 'torch':
            return retrieve_contexts(bank, items, self.context_size,
                                     self.retrieval_query_batch_size, self.retrieval_item_chunk_size)
        index = self._ensure_faiss_index(bank, space, behavior)
        count = min(self.context_size, self.n_items - 1)
        rows = []
        for candidates in items.split(self.retrieval_query_batch_size):
            queries = bank[candidates].detach().to(device='cpu', dtype=torch.float32).contiguous().numpy()
            if not np.isfinite(queries).all():
                raise FloatingPointError("FAISS queries must be finite.")
            _, neighbors = index.search(queries, count + 1)
            for candidate, neighbors_row in zip(candidates.tolist(), neighbors):
                valid = neighbors_row[(neighbors_row >= 0) & (neighbors_row != candidate - 1)] + 1
                if len(valid) < count:
                    raise RuntimeError("FAISS returned too few contexts; increase faiss_ef_search or use faiss_flat.")
                rows.append(torch.from_numpy(valid[:count].copy()))
        return torch.stack(rows).to(items.device) if rows else items.new_empty((0, count))

    def _contexts(self, state, space, behavior, items, use_cache=False):
        bank = state.item_c[behavior] if space == 'c' else state.item_s[behavior]
        unique, inverse = items.unique(sorted=True, return_inverse=True)
        if not use_cache or not self.max_context_cache_entries:
            indices = self._retrieve(bank, space, behavior, unique)
            knn = context_knn(bank[indices].detach(), self.knn_degree)
            return indices[inverse], knn[inverse]
        keys = [(space, behavior, item) for item in unique.tolist()]
        entries = [self._context_cache.get(key) for key in keys]
        missing = [index for index, entry in enumerate(entries) if entry is None]
        if missing:
            missing_indices = self._retrieve(bank, space, behavior, unique[missing])
            missing_knn = context_knn(bank[missing_indices], self.knn_degree)
            for row, index in enumerate(missing):
                entries[index] = (missing_indices[row].cpu().clone(), missing_knn[row].cpu().clone())
        for key, entry in zip(keys, entries):
            self._context_cache[key] = entry
            self._context_cache.move_to_end(key)
        while len(self._context_cache) > self.max_context_cache_entries:
            self._context_cache.popitem(last=False)
        indices = torch.stack([entry[0] for entry in entries]).to(items.device)
        knn = torch.stack([entry[1] for entry in entries]).to(items.device)
        return indices[inverse], knn[inverse]

    def _logic_space(self, state, users, items, space, behavior, use_cache=False):
        indices, knn = self._contexts(state, space, behavior, items, use_cache)
        user_bank = state.user_c[behavior] if space == 'c' else state.user_s[behavior]
        item_bank = state.item_c[behavior] if space == 'c' else state.item_s[behavior]
        nodes = torch.cat([
            user_bank[users, None], state.item[-1][items, None], item_bank[indices],
        ], dim=1)
        return nodes, indices, knn

    def _score_configurations(self, state, users, items, configs, use_cache=False, return_trace=False):
        scores = state.user[-1].new_zeros(len(users))
        traces = []
        for config in configs.unique(sorted=True).tolist():
            positions = (configs == config).nonzero(as_tuple=True)[0]
            selected_users, selected_items = users[positions], items[positions]
            if config == -1:
                candidate = state.item[-1][selected_items]
                trace = {'rule': 'Strong', 'rule_grounded': False}
            else:
                space, behavior = self._configuration(config)
                nodes, contexts, knn = self._logic_space(
                    state, selected_users, selected_items, space, behavior, use_cache,
                )
                edges = local_logic_edges(knn)
                executor = self.gat_c if space == 'c' else self.gat_s
                candidate, execution = executor(nodes, edges, return_trace)
                if return_trace:
                    bank = state.item_c[behavior] if space == 'c' else state.item_s[behavior]
                    trace = {
                        'rule': 'fallback' if config == 0 else ('Moderate' if space == 'c' else 'Weak'),
                        'rule_grounded': config != 0, 'executor': space,
                        'retrieval_backend': self.retrieval_backend,
                        'behavior': self.behaviors[behavior], 'context_ids': contexts,
                        'retrieval_similarities': (bank[selected_items, None] * bank[contexts]).sum(-1),
                        'knn': knn, 'edge_index': edges, 'node_states': execution['states'],
                        'attention': execution['attention'],
                    }
            # Prediction: target MID user dotted with the updated candidate, Eq. (9).
            group_scores = (state.user[-1][selected_users] * candidate).sum(-1)
            scores = scores.index_copy(0, positions, group_scores)
            if return_trace:
                trace.update({'pair_indices': positions, 'users': selected_users, 'items': selected_items,
                              'configuration': config, 'candidate_output': candidate, 'scores': group_scores})
                traces.append(trace)
        return scores, traces

    def _parameter_signature(self):
        return tuple((parameter._version, str(parameter.device), parameter.dtype) for parameter in self.parameters())

    @torch.no_grad()
    def update_faiss_indexes(self):
        self._inference_state = self.encode()
        self.faiss_indexes_gcn.clear()
        self.faiss_indexes_hyper.clear()
        self._indexed_banks.clear()
        if self.retrieval_backend != 'torch':
            for behavior in range(self.n_behaviors):
                self._ensure_faiss_index(self._inference_state.item_c[behavior], 'c', behavior)
                self._ensure_faiss_index(self._inference_state.item_s[behavior], 's', behavior)
        self._inference_signature = self._parameter_signature()
        self._context_cache.clear()

    @torch.no_grad()
    def score_candidates(self, users, items, histories, return_trace=False):
        if self.training:
            raise RuntimeError("score_candidates requires eval mode.")
        if users.ndim != 1 or items.shape != users.shape or len(histories) != len(users):
            raise ValueError("Expected aligned one-dimensional users and items.")
        if users.dtype != torch.long or items.dtype != torch.long:
            raise ValueError("User and item IDs must have dtype torch.long.")
        if ((users < 1) | (users > self.n_users) | (items < 1) | (items > self.n_items)).any().item():
            raise ValueError("Candidate scoring requires real user/item IDs, excluding padding.")
        configs = self.dispatch(histories)
        if self._inference_state is None or self._inference_signature != self._parameter_signature():
            self.update_faiss_indexes()
        scores, traces = self._score_configurations(
            self._inference_state, users, items, configs, use_cache=True, return_trace=return_trace,
        )
        return (scores, traces) if return_trace else scores

    @staticmethod
    def _bpr(positive, negative):
        return F.softplus(negative - positive)

    def _target_objective(self, state, users, positives, negatives, positive_history, negative_history):
        positive_config = self.dispatch(positive_history)
        negative_config = self.dispatch(negative_history)
        positive_score, _ = self._score_configurations(state, users, positives, positive_config)
        negative_score, _ = self._score_configurations(state, users, negatives, negative_config)
        natural = self._bpr(positive_score, negative_score).sum()
        calibration = natural.new_zeros(())
        different = positive_config != negative_config
        if self.lambda_cal and different.any().item():
            # Sec. 3.4: three comparisons, each weighted by lambda_cal / 3.
            cross_positive, _ = self._score_configurations(
                state, users[different], positives[different], negative_config[different],
            )
            cross_negative, _ = self._score_configurations(
                state, users[different], negatives[different], positive_config[different],
            )
            calibration = self.lambda_cal / 3 * (
                self._bpr(positive_score[different], cross_negative)
                + self._bpr(cross_positive, negative_score[different])
                + self._bpr(cross_positive, cross_negative)
            ).sum()
        return natural, calibration

    @staticmethod
    def _cosine_distance(first, second):
        return (1 - F.cosine_similarity(first, second, dim=-1, eps=1e-8)).clamp(0, 2)

    # Dual Regularization: six equally weighted responses, Table 1 and Eq. (7).
    def logical_regularization(self, state, users, items, histories):
        if len(users) == 0:
            return self.e_T.new_zeros(()), self.e_T.new_zeros(6)
        if not torch.isfinite(self.e_T).all().item() or self.e_T.detach().norm().item() < 1e-8:
            raise FloatingPointError("The learnable true anchor must remain finite and nonzero.")
        self.dispatch(histories)
        observed = histories[:, :-1].bool()
        levels = torch.arange(self.n_behaviors - 1, device=users.device)
        downstream = torch.where(observed, levels, -1).amax(dim=-1)
        downstream = torch.where(downstream >= 0, downstream, self.n_behaviors - 1)
        penalties = self.e_T.new_zeros(6)
        for behavior in downstream.unique().tolist():
            select = downstream == behavior
            selected_users, selected_items = users[select], items[select]
            for offset, space in enumerate(['c', 's']):
                nodes, _, knn = self._logic_space(state, selected_users, selected_items, space, behavior)
                executor = self.gat_c if space == 'c' else self.gat_s
                base, _ = executor(nodes, local_logic_edges(knn))
                true_anchor = self.e_T.expand(len(nodes), -1)
                false_anchor = -true_anchor
                identity_anchor = true_anchor if space == 'c' else false_anchor
                annihilator_anchor = false_anchor if space == 'c' else true_anchor
                anchor_edges = local_logic_edges(knn, extra_nodes=1)
                identity, _ = executor(torch.cat([nodes, identity_anchor[:, None]], dim=1), anchor_edges)
                annihilator, _ = executor(torch.cat([nodes, annihilator_anchor[:, None]], dim=1), anchor_edges)
                predicate = state.item[-1][selected_items]
                complement_nodes = torch.cat([nodes[:, :2], predicate[:, None], -predicate[:, None]], dim=1)
                complement_edges = local_logic_edges(context_knn(complement_nodes[:, 2:], self.knn_degree))
                complement, _ = executor(complement_nodes, complement_edges)
                values = [self._cosine_distance(identity, base),
                          self._cosine_distance(annihilator, annihilator_anchor),
                          self._cosine_distance(complement, annihilator_anchor)]
                for law, value in enumerate(values):
                    penalties = penalties.index_add(
                        0, penalties.new_tensor([2 * law + offset], dtype=torch.long), value.sum().reshape(1),
                    )
        penalties = penalties / len(users)
        return penalties.mean(), penalties

    def forward(self, batch):
        if len(batch) != 5:
            raise ValueError("Expected five batch fields, including both routing histories.")
        users, positives, negatives, positive_history, negative_history = batch
        if users.ndim != 1 or any(value.dtype != torch.long for value in [users, positives, negatives]):
            raise ValueError("Training IDs must be torch.long, with users of shape [batch].")
        if ((users < 1) | (users > self.n_users)).any().item():
            raise ValueError("Training user IDs must exclude padding and stay in range.")
        if any(((value < 0) | (value > self.n_items)).any().item() for value in [positives, negatives]):
            raise ValueError("Training item IDs must stay in [0, item_count].")
        expected = (len(users), self.n_behaviors)
        if positives.shape != expected or negatives.shape != expected:
            raise ValueError("Positive and negative items must have shape [batch, behaviors].")
        if positive_history.shape != (*expected, self.n_behaviors) or negative_history.shape != positive_history.shape:
            raise ValueError("Routing histories must have shape [batch, behaviors, behaviors].")
        self.dispatch(positive_history.reshape(-1, self.n_behaviors))
        self.dispatch(negative_history.reshape(-1, self.n_behaviors))
        if ((positives == 0) != (negatives == 0)).any().item():
            raise ValueError("Padding must be paired in positive and negative slots.")
        self._inference_state = None
        self._context_cache.clear()
        state = self.encode()
        zero = self.e_T.new_zeros(())
        auxiliary = zero
        for behavior in range(self.n_behaviors - 1):
            valid = positives[:, behavior] > 0
            user = state.user[behavior][users[valid]]
            positive_score = (user * state.item[behavior][positives[valid, behavior]]).sum(-1)
            negative_score = (user * state.item[behavior][negatives[valid, behavior]]).sum(-1)
            auxiliary = auxiliary + self._bpr(positive_score, negative_score).sum()
        valid = positives[:, -1] > 0
        target_users, target_positive, target_negative = users[valid], positives[valid, -1], negatives[valid, -1]
        positive_hist, negative_hist = positive_history[valid, -1], negative_history[valid, -1]
        natural, calibration = self._target_objective(
            state, target_users, target_positive, target_negative, positive_hist, negative_hist,
        ) if valid.any().item() else (zero, zero)
        # Semantic consistency on the same mixed target triplets, Eq. (8).
        target_user = state.user[-1][target_users]
        semantic = self._bpr(
            (target_user * state.item[-1][target_positive]).sum(-1),
            (target_user * state.item[-1][target_negative]).sum(-1),
        ).sum()
        logical, axioms = self.logical_regularization(
            state, target_users, target_positive, positive_hist,
        ) if self.lambda_r else (zero, zero.expand(6))
        l2 = sum(parameter.square().sum() for parameter in self.parameters()) if self.lambda_2 else zero
        # Prediction and Optimization: complete objective, Eq. (10).
        loss = natural + calibration + auxiliary + self.lambda_r * logical + self.lambda_s * semantic + self.lambda_2 * l2
        self.last_losses = {name: value.detach() for name, value in {
            'target_natural': natural, 'calibration': calibration, 'auxiliary': auxiliary,
            'semantic': semantic, 'logical': logical, 'axioms': axioms, 'l2': l2, 'total': loss,
        }.items()}
        return loss
