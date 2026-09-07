import argparse
import copy
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch

from data_setp import DataSet
from model import Model
from trainer import Trainer


ROOT = Path(__file__).resolve().parent

# Appendix B.1.2 and B.3: dataset-specific settings; Beibei depths are provisional.
DATASET_DEFAULTS = {
    'taobao': {'behaviors': ['view', 'cart', 'buy'], 'layers_nums': [1, 1, 3], 'lr': 1e-3},
    'tmall': {'behaviors': ['view', 'collect', 'cart', 'buy'], 'layers_nums': [1, 1, 1, 3], 'lr': 1e-4},
    'beibei': {'behaviors': ['view', 'cart', 'buy'], 'layers_nums': [1, 1, 3], 'lr': 1e-3},
}


def build_parser():
    parser = argparse.ArgumentParser(
        description='DREF: Dynamic Reasoning for Multi-Behavior Recommendation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    data = parser.add_argument_group('Data and experiment')
    data.add_argument('--data_name', choices=DATASET_DEFAULTS, default='taobao', help='Dataset profile')
    data.add_argument('--data_path', default=None, help='Interaction directory; default: data/<data_name> next to main.py')
    data.add_argument('--behaviors', nargs='+', default=None, help='Ordered chain; default: dataset profile')
    data.add_argument('--model_path', default=str(ROOT / 'check_point'), help='Root for separate experiment directories')
    data.add_argument('--model_name', default='DREF', help='Checkpoint filename prefix')
    seeds = data.add_mutually_exclusive_group()
    seeds.add_argument('--seed', type=int, default=None, help='Run one seed for debugging or a single experiment')
    seeds.add_argument('--seeds', type=int, nargs='+', default=[2020, 2021, 2022, 2023, 2024],
                       help='Run seeds independently; exact seed values are an implementation choice')
    data.add_argument('--num_workers', type=int, default=0, help='Training DataLoader workers')
    data.add_argument('--target_negative_ratio', type=float, default=0.5,
                      help='Probability of auxiliary-observed target negatives, Sec. 3.4')

    # MID: collaborative/semantic encoding and cascading, Eq. (1)-(3).
    mid = parser.add_argument_group('MID')
    mid.add_argument('--embedding_size', type=int, default=64, help='Embedding dimension, divisible by four')
    mid.add_argument('--hyper_nums', type=int, default=32, help='Latent hyperedges per hypergraph')
    mid.add_argument('--layers_nums', type=int, nargs='+', default=None, help='GCN depths; default: dataset profile')

    # DMR: context retrieval and Local Logic Space, Eq. (4)-(6), Appendix B.1.3.
    dmr = parser.add_argument_group('DMR and retrieval')
    dmr.add_argument('--context_size', type=int, default=20, help='Retrieved contexts per candidate')
    dmr.add_argument('--knn_degree', type=int, default=5, help='Context KNN degree; zero disables KNN edges')
    dmr.add_argument('--retrieval_backend', choices=['faiss_flat', 'faiss_hnsw', 'torch'], default='faiss_flat',
                     help='Exact FAISS IP, approximate HNSW IP, or exact Torch reference')
    dmr.add_argument('--faiss_hnsw_m', type=int, default=32, help='HNSW links parameter')
    dmr.add_argument('--faiss_ef_construction', type=int, default=200, help='HNSW construction search width')
    dmr.add_argument('--faiss_ef_search', type=int, default=64, help='HNSW query search width')
    dmr.add_argument('--faiss_threads', type=int, default=1, help='CPU FAISS threads')
    dmr.add_argument('--retrieval_query_batch_size', type=int, default=128, help='Candidate queries per retrieval chunk')
    dmr.add_argument('--retrieval_item_chunk_size', type=int, default=4096, help='Item chunk size for Torch retrieval only')
    dmr.add_argument('--max_context_cache_entries', type=int, default=100000, help='Inference context/KNN cache capacity; zero disables')

    # Dual Regularization and Optimization: Eq. (7)-(10).
    train = parser.add_argument_group('Optimization and losses')
    train.add_argument('--epochs', type=int, default=200, help='Maximum training epochs, at most 200')
    train.add_argument('--batch_size', type=int, default=1024, help='Training users per batch')
    train.add_argument('--lr', type=float, default=None, help='Adam learning rate; default: dataset profile')
    train.add_argument('--lambda_r', type=float, default=0.01, help='Logical regularization weight')
    train.add_argument('--lambda_s', type=float, default=0.05, help='Semantic consistency weight')
    train.add_argument('--lambda_cal', type=float, default=1.0, help='Cross-configuration calibration weight')
    train.add_argument('--lambda_2', '--reg_weight', dest='lambda_2', type=float, default=1e-4,
                       help='L2 regularization weight')
    train.add_argument('--patience', type=int, default=0, help='Validation early stopping patience; zero disables')
    train.add_argument('--grad_clip', type=float, default=0.0, help='Gradient norm limit; zero disables')

    evaluation = parser.add_argument_group('Full-catalog evaluation')
    evaluation.add_argument('--test_batch_size', type=int, default=256, help='Evaluation users per batch')
    evaluation.add_argument('--eval_pair_batch_size', type=int, default=4096, help='Maximum user-item pairs per scoring chunk')
    evaluation.add_argument('--topk', type=int, nargs='+', default=[10, 20, 50], help='Ranking cutoffs')
    evaluation.add_argument('--metrics', type=str.lower, nargs='+', choices=['hit', 'hr', 'ndcg'],
                            default=['hit', 'ndcg'], help='Ranking metrics')
    evaluation.add_argument('--selection_metric', default=None,
                            help='Validation checkpoint metric; default: ndcg@10 if available, else first metric/smallest K')

    runtime = parser.add_argument_group('Runtime')
    runtime.add_argument('--device', default='auto', help='auto, cpu, or cuda:N; auto selects cuda:0 when available')
    runtime.add_argument('--torch_threads', type=int, default=0, help='CPU Torch threads; zero preserves the runtime setting')
    runtime.add_argument('--deterministic', action=argparse.BooleanOptionalAction, default=False,
                         help='Require deterministic Torch algorithms; unsupported operations raise errors')
    runtime.add_argument('--detect_anomaly', action=argparse.BooleanOptionalAction, default=False,
                         help='Enable autograd anomaly detection for debugging')
    runtime.add_argument('--show_progress', action=argparse.BooleanOptionalAction, default=True, help='Show batch progress bars')
    runtime.add_argument('--print_config', action='store_true', help='Print resolved parameters and exit without loading data')
    return parser


def parse_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    profile = DATASET_DEFAULTS[args.data_name]
    for key in ['behaviors', 'layers_nums', 'lr']:
        if getattr(args, key) is None:
            setattr(args, key, copy.deepcopy(profile[key]))
    if args.behaviors != profile['behaviors']:
        parser.error(f"--behaviors must follow the {args.data_name} profile: {profile['behaviors']}")
    if args.seed is not None:
        args.seeds = [args.seed]
    if len(set(args.seeds)) != len(args.seeds) or any(not 0 <= seed < 2 ** 32 for seed in args.seeds):
        parser.error('Seeds must be distinct integers in [0, 2**32)')
    args.seed = args.seeds[0]
    args.topk = sorted(set(args.topk))
    args.metrics = list(dict.fromkeys('hit' if metric == 'hr' else metric for metric in args.metrics))
    default_k = 10 if 10 in args.topk else args.topk[0]
    default_metric = 'ndcg' if 'ndcg' in args.metrics else args.metrics[0]
    args.selection_metric = (args.selection_metric or f'{default_metric}@{default_k}').lower()
    if args.selection_metric.startswith('hr@'):
        args.selection_metric = 'hit@' + args.selection_metric.split('@', 1)[1]
    if not args.model_name or any(character in args.model_name for character in '<>:"/\\|?*'):
        parser.error('--model_name must be a nonempty filename prefix without reserved path characters')
    args.data_path = str(Path(args.data_path).expanduser().resolve()) if args.data_path else str(ROOT / 'data' / args.data_name)
    args.model_path = str(Path(args.model_path).expanduser().resolve())
    if args.device == 'auto':
        args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    return args


def seed_everything(seed, deterministic=False):
    if deterministic:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(deterministic)


def write_json(path, payload):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def run_seed(args):
    seed_everything(args.seed, args.deterministic)
    if args.torch_threads < 0 or args.faiss_threads < 1:
        raise ValueError('Require torch_threads >= 0 and faiss_threads >= 1.')
    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    start = time.perf_counter()
    dataset = DataSet(args)
    model = Model(args, dataset)
    if model._faiss is not None:
        model._faiss.omp_set_num_threads(args.faiss_threads)
    trainer = Trainer(model, dataset, args)
    result = trainer.train_model()
    if torch.device(args.device).type == 'cuda':
        torch.cuda.synchronize(torch.device(args.device))
    result.update({
        'seed': args.seed,
        'elapsed_seconds': time.perf_counter() - start,
        'runtime': {'torch': torch.__version__, 'numpy': np.__version__,
                    'faiss': getattr(model._faiss, '__version__', None), 'device': args.device,
                    'torch_threads': torch.get_num_threads(), 'retrieval_backend': args.retrieval_backend},
    })
    return result


def run_experiment(args):
    required = ['count.txt', 'validation_dict.txt', 'test_dict.txt']
    required.extend(f'{behavior}_dict.txt' for behavior in args.behaviors)
    missing = [name for name in required if not (Path(args.data_path) / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing data files in {args.data_path}: {', '.join(missing)}")
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid4().hex[:8]
    output = Path(args.model_path) / args.data_name / run_id
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'config.json', vars(args))
    print(json.dumps(vars(args), ensure_ascii=False, indent=2), flush=True)
    print(f'Experiment directory: {output}', flush=True)
    results = []
    for seed in args.seeds:
        seed_args = copy.deepcopy(args)
        seed_args.seed = seed
        seed_args.seeds = [seed]
        seed_args.model_path = str(output / f'seed_{seed}')
        seed_directory = Path(seed_args.model_path)
        seed_directory.mkdir()
        write_json(seed_directory / 'config.json', vars(seed_args))
        print(f'Running seed {seed}', flush=True)
        result = run_seed(seed_args)
        write_json(seed_directory / 'result.json', result)
        results.append(result)
        metrics = {
            key: {'mean': float(np.mean([run['test'][key] for run in results])),
                  'std_population': float(np.std([run['test'][key] for run in results], ddof=0))}
            for key in result['test']
        }
        summary = {'completed_seeds': [run['seed'] for run in results], 'requested_seeds': args.seeds,
                   'test_metrics': metrics, 'runs': results, 'output_directory': str(output)}
        write_json(output / 'summary.json', summary)
        if torch.device(args.device).type == 'cuda':
            torch.cuda.empty_cache()
    print(json.dumps(summary['test_metrics'], ensure_ascii=False, indent=2), flush=True)
    print(f'Results saved to {output / "summary.json"}', flush=True)
    return summary


def main(argv=None):
    args = parse_args(argv)
    if args.print_config:
        print(json.dumps(vars(args), ensure_ascii=False, indent=2))
        return vars(args)
    return run_experiment(args)


if __name__ == '__main__':
    main()
