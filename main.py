import argparse
import os
import random
import sys
import time
import warnings

import numpy as np
import torch
from torch.backends import cudnn

from solver import Solver
from utils.utils import mkdir

warnings.filterwarnings("ignore")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def format_elapsed(seconds):
    seconds = int(round(float(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class Logger(object):
    def __init__(self, filename="default.log", add_flag=True, stream=sys.stdout):
        self.terminal = stream
        self.filename = filename
        self.add_flag = add_flag

    def write(self, message):
        mode = "a+" if self.add_flag else "w"
        with open(self.filename, mode, encoding="utf-8") as log:
            self.terminal.write(message)
            log.write(message)

    def flush(self):
        pass


def resolve_dataset_result_path(result_path, dataset_name):
    result_path = os.path.normpath(result_path)
    dataset_name = str(dataset_name)
    if os.path.normcase(os.path.basename(result_path)) == os.path.normcase(dataset_name):
        return result_path
    return os.path.join(result_path, dataset_name)


def main(config):
    cudnn.benchmark = True
    experiment_start = time.time()
    solver = None

    try:
        mkdir(config.model_save_path)
        mkdir(config.result_path)

        solver = Solver(vars(config))
        solver.print_runtime_device_info()
        set_seed(config.seed)

        if config.mode == "train":
            solver.train()
            solver.test()
        else:
            solver.test()

        return solver
    finally:
        if solver is not None and solver.is_using_gpu():
            try:
                torch.cuda.synchronize(solver.device)
            except Exception:
                pass

        elapsed = time.time() - experiment_start
        print("\n================ Experiment Runtime ================")
        print(f"Total elapsed time     : {elapsed:.2f}s")
        print(f"Total elapsed hms      : {format_elapsed(elapsed)}")
        print("====================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--win_size", type=int, default=15)  
    parser.add_argument("--step", type=int, default=1)  
    parser.add_argument("--anormly_ratio", type=float, default=0.24)  
    parser.add_argument("--batch_size", type=int, default=128)  
    parser.add_argument("--epochs", type=int, default=10)  
    parser.add_argument("--stage23_lr", type=float, default=1e-3)  
    parser.add_argument("--use_gpu", type=bool, default=True)  
    parser.add_argument("--gpu", type=int, default=0)  
    parser.add_argument("--use_multi_gpu", action="store_true", default=False)  
    parser.add_argument("--devices", type=str, default="0,1,2,3")  
    parser.add_argument("--index", type=int, default=137)  
    parser.add_argument("--input_c", type=int, default=0)  
    parser.add_argument("--output_c", type=int, default=0)  
    parser.add_argument("--dataset", type=str, default="Genesis")  
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])  
    parser.add_argument("--data_path", type=str, default="Genesis")  
    parser.add_argument("--model_save_path", type=str, default="checkpoints")  
    parser.add_argument("--result_path", type=str, default="result")  

    parser.add_argument("--stage1_lr", type=float, default=1e-3)  
    parser.add_argument("--stage1_weight_decay", type=float, default=1e-4)  
    parser.add_argument("--stage1_mask_ratio", type=float, default=0.3)  
    parser.add_argument("--stage1_center_init_std", type=float, default=0.5)  
    parser.add_argument("--stage1_sigma_init", type=float, default=1.0) 
    parser.add_argument("--stage1_sigma_min", type=float, default=0.05)  
    parser.add_argument("--stage1_peak_amp_init", type=float, default=1.0)  
    parser.add_argument("--stage1_checkpoint", type=str, default="auto")  

    parser.add_argument("--stage2_num_patterns", type=int, default=5)  
    parser.add_argument("--top_q", type=int, default=1)  
    parser.add_argument("--stage2_center_init_std", type=float, default=0.5)  
    parser.add_argument("--stage2_sigma_init", type=float, default=1.0)  
    parser.add_argument("--stage2_sigma_min", type=float, default=0.05)  
    parser.add_argument("--stage2_peak_amp_init", type=float, default=1.0)  

    parser.add_argument("--stage3_center_init", type=float, default=0.0) 
    parser.add_argument("--stage3_one_hot_gamma", type=float, default=0.3)  
    parser.add_argument("--stage3_peak_amp_init", type=float, default=1.0) 
    parser.add_argument("--score_type", type=str, default="abs", choices=["abs", "square"])  
    parser.add_argument("--threshold_source", type=str, default="combined", choices=["train", "combined"])  
    parser.add_argument("--vus_sliding_window", type=int, default=100)  

    parser.add_argument("--seed", type=int, default=42)  
    parser.add_argument("--num_workers", type=int, default=0)  
    parser.add_argument("--save_name", type=str, default=None)  

    config = parser.parse_args()
    config.result_path = resolve_dataset_result_path(config.result_path, config.dataset)
    args = vars(config)

    set_seed(config.seed)
    config.use_gpu = True if torch.cuda.is_available() and config.use_gpu else False
    if config.use_gpu and config.use_multi_gpu:
        config.devices = config.devices.replace(" ", "")
        device_ids = config.devices.split(",")
        config.device_ids = [int(id_) for id_ in device_ids]
        config.gpu = config.device_ids[0]

    mkdir(config.result_path)
    sys.stdout = Logger(os.path.join(config.result_path, f"result_{config.data_path}.log"), stream=sys.stdout)

    print("\n\n")
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print("================ Hyperparameters ===============")
    for k, v in sorted(args.items()):
        print("%s: %s" % (str(k), str(v)))
    print("================================================")

    main(config)
