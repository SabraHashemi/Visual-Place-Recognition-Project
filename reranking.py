import numpy as np
from tqdm import tqdm
import os, argparse
from glob import glob
from pathlib import Path
import torch

from util import get_list_distances_from_preds

# Import wandb logger (optional)
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from wandb_logging import WandBLogger
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

def parse_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--preds-dir", type=str, help="directory with predictions of a VPR model")
    parser.add_argument("--inliers-dir", type=str, help="directory with image matching results")
    parser.add_argument("--num-preds", type=int, default=100, help="number of predictions to re-rank")
    parser.add_argument(
        "--positive-dist-threshold",
        type=int,
        default=25,
        help="distance (in meters) for a prediction to be considered a positive",
    )
    parser.add_argument(
        "--recall-values",
        type=int,
        nargs="+",
        default=[1, 5, 10, 20, 100],
        help="values for recall (e.g. recall@1, recall@5)",
    )

    return parser.parse_args()

def main(args):
    preds_folder = args.preds_dir
    inliers_folder = Path(args.inliers_dir)
    num_preds = args.num_preds
    threshold = args.positive_dist_threshold
    recall_values = args.recall_values

    # Initialize wandb logger
    wandb_logger = None
    if WANDB_AVAILABLE:
        # Try to infer matcher name from inliers folder name (e.g., "log_dir_superpoint-lg")
        matcher_name = "unknown"
        if "_" in str(inliers_folder.name):
            matcher_name = "_".join(str(inliers_folder.name).split("_")[1:])
        
        config = {
            "matcher": matcher_name,
            "num_preds": num_preds,
            "positive_dist_threshold": threshold,
            "recall_values": recall_values,
            "preds_dir": str(preds_folder),
            "inliers_dir": str(inliers_folder),
        }
        experiment_name = f"reranking_{matcher_name}"
        wandb_logger = WandBLogger(
            project_name="vpr-evaluation",
            experiment_name=experiment_name,
            config=config
        )

    txt_files = glob(os.path.join(preds_folder, "*.txt"))
    txt_files.sort(key=lambda x: int(Path(x).stem))

    total_queries = len(txt_files)
    recalls = np.zeros(len(recall_values))

    for txt_file_query in tqdm(txt_files):
        geo_dists = torch.tensor(get_list_distances_from_preds(txt_file_query))[:num_preds]
        torch_file_query = inliers_folder.joinpath(Path(txt_file_query).name.replace('txt', 'torch'))
        query_results = torch.load(torch_file_query, weights_only=False)
        query_db_inliers = torch.zeros(num_preds, dtype=torch.float32)
        for i in range(num_preds):
            query_db_inliers[i] = query_results[i]['num_inliers']
        query_db_inliers, indices = torch.sort(query_db_inliers, descending=True)
        geo_dists = geo_dists[indices]
        
        for i, n in enumerate(recall_values):
            if torch.any(geo_dists[:n] <= threshold):
                recalls[i:] += 1
                break

    recalls = recalls / total_queries * 100
    recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(recall_values, recalls)])

    print(recalls_str)
    
    # Log reranking recalls to wandb
    if wandb_logger:
        metrics = {f"rerank_recall@{val}": float(rec) for val, rec in zip(recall_values, recalls)}
        wandb_logger.log_metrics(metrics)
        wandb_logger.finish()

if __name__ == "__main__":
    args = parse_arguments()
    main(args)