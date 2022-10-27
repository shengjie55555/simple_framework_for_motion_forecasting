import os
import argparse
import time
import multiprocessing
import numpy as np
import pandas as pd
import pickle as pkl

from typing import Any, Dict, List, Tuple
from tqdm import tqdm
from joblib import Parallel, delayed
from data_preprocess import lanegcn_preprocess, vectornet_preprocess


def get_args() -> Any:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="", type=str, help="Directory where the sequences (csv file) are saved.")
    parser.add_argument("--save_dir", default="", type=str,
                        help="Directory where the processed data (pickle file) are saved.")
    parser.add_argument("--model", default="LaneGCN", type=str, help="LaneGCN/VectorNet")
    parser.add_argument("--mode", required=True, type=str, help="train/val/test")
    parser.add_argument("--obs_len", default=20, type=int, help="Observed length of the historical trajectory.")
    parser.add_argument("--pred_len", default=30, type=int, help="Predicted horizon.")
    parser.add_argument("--debug", default=False, action="store_true", help="If True, debug mode.")
    parser.add_argument("--viz", default=False, action="store_true", help="If True, viz.")
    return parser.parse_args()


def load_seq_save_features(args: Any, start_idx: int, batch_size: int, sequences: List[str],
                           save_dir: str) -> None:
    if args.model == "LaneGCN":
        dataset = lanegcn_preprocess.PreProcess(args)
    else:
        dataset = vectornet_preprocess.PreProcess(args)

    sample_loop = tqdm(enumerate(sequences[start_idx: start_idx + batch_size]),
                       total=len(sequences[start_idx: start_idx + batch_size]),
                       desc="Processing", leave=False)
    for _, seq in sample_loop:
        if not seq.endswith(".csv"):
            continue

        seq_id = int(seq.split(".")[0])
        seq_path = f"{args.data_dir}/{seq}"

        df = pd.read_csv(seq_path, dtype={"TIMESTAMP": str})
        data, headers = dataset.process(seq_id, df)

        if not args.debug:
            data_df = pd.DataFrame(data, columns=headers)
            filename = "{}".format(data[0][0])
            data_df.to_pickle(f"{save_dir}/{filename}.pkl")

    print("Finish computing {} - {}".format(start_idx, start_idx + batch_size))


def main():
    start = time.time()
    args = get_args()

    sequences = os.listdir(args.data_dir)
    num_sequences = len(sequences)
    print("num of sequences: ", num_sequences)

    n_proc = multiprocessing.cpu_count() - 2 if not args.debug else 1

    batch_size = np.max([int(np.ceil(num_sequences / n_proc)), 1])
    print("n_proc: {}, batch_size: {}".format(n_proc, batch_size))

    save_dir = os.path.join(args.save_dir, args.mode)
    os.makedirs(save_dir, exist_ok=True)
    print("save processed dataset to {}".format(save_dir))

    Parallel(n_jobs=n_proc)(delayed(load_seq_save_features)(args, i, batch_size, sequences, save_dir)
                            for i in range(0, num_sequences, batch_size))

    print("Preprocess for {} set completed in {} minutes".format(args.mode, (time.time() - start) / 60.0))


if __name__ == "__main__":
    main()
