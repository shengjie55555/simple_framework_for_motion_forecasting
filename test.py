import os
import argparse
import torch
import warnings
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.data_utils import collate_fn
from utils.train_utils import init_seeds, load_prev_weights
from utils.train_utils import Loader, update_cfg
from argoverse.evaluation.competition_util import generate_forecasting_h5

warnings.filterwarnings("ignore")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="", type=str, help="Path to the pre-trained weights.")
    parser.add_argument("--weights_name", default=1, type=int, help="Which weights for loading.")
    parser.add_argument("--test_dir", required=True, default="", type=str,
                        help="Path to the file which has features for test.")
    parser.add_argument("--test_batch_size", default=1, type=int, help="Test batch size.")
    parser.add_argument("--model", default="VectorNet", type=str, help="Name of the selected model.")
    parser.add_argument("--use_cuda", default=True, type=bool, help="Use CUDA for acceleration.")
    parser.add_argument("--resume", action="store_true", default=False, help="Resume training.")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    loader = Loader(args.model)
    cfg, dataset_cls, model_cls, loss_cls, al_cls, am_cls, vis_cls = loader.load()
    cfg = update_cfg(args, cfg, is_test=True)
    print("Args: ", args)
    print("Config: ", cfg)
    print(dataset_cls, model_cls, loss_cls, al_cls, am_cls, vis_cls)

    # set seed
    init_seeds(0)

    # select device
    if args.use_cuda and torch.cuda.is_available():
        device = torch.device("cuda", 0)
    else:
        device = torch.device('cpu')

    # dataset & dataloader
    test_set = dataset_cls(cfg["processed_test"], mode="test")
    test_loader = DataLoader(test_set,
                             batch_size=cfg["test_batch_size"],
                             num_workers=cfg['test_workers'],
                             shuffle=False,
                             pin_memory=True,
                             collate_fn=collate_fn)

    # model & training strategy
    net = model_cls(cfg, device).to(device)
    loss_net = loss_cls(cfg, device).to(device)

    # resume
    if args.resume:
        model_path = os.path.join(cfg["save_dir"], str(args.weights_name) + ".pth")
        if not os.path.isfile(model_path):
            assert False, "Model path error - {}".format(model_path)
        else:
            load_prev_weights(net, model_path, cfg)
    else:
        print("Test without pre-trained weights!")

    # test
    preds = {}
    prob_preds = {}
    loop = tqdm(enumerate(test_loader), total=len(test_loader), desc="Test", leave=True)
    net.eval()
    with torch.no_grad():
        for i, batch in loop:
            out = net(batch)
            results = [x[0:1].detach().cpu().numpy() for x in out['reg']]
            prob_results = [x[0:1].detach().cpu().numpy() for x in out['cls']]
            for j, idx in enumerate(batch['argo_id']):
                preds[idx] = results[j].squeeze()
                prob_preds[idx] = [x for x in prob_results[j].squeeze()]
    generate_forecasting_h5(preds, cfg['competition_files'], probabilities=prob_preds)


if __name__ == "__main__":
    main()
