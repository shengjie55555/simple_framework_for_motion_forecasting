import os
import argparse
import torch
import warnings
from tqdm import tqdm
from torch.utils.data import DataLoader
from visualize.vis_vectornet import Vis
from utils.data_utils import collate_fn
from utils.train_utils import init_seeds, load_prev_weights
from utils.train_utils import Loader, update_cfg

warnings.filterwarnings("ignore")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="", type=str, help="Path to the pre-trained weights.")
    parser.add_argument("--weights_name", default=1, type=int, help="Which weights for loading.")
    parser.add_argument("--val_dir", required=True, default="", type=str,
                        help="Path to the file which has features for evaluation.")
    parser.add_argument("--val_batch_size", default=1, type=int, help="Val batch size.")
    parser.add_argument("--model", default="VectorNet", type=str, help="Name of the selected model.")
    parser.add_argument("--use_cuda", default=True, type=bool, help="Use CUDA for acceleration.")
    parser.add_argument("--resume", action="store_true", default=False, help="Resume training.")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    loader = Loader(args.model)
    cfg, dataset_cls, model_cls, loss_cls, al_cls, am_cls, vis_cls = loader.load()
    cfg = update_cfg(args, cfg, is_eval=True)
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
    val_set = dataset_cls(cfg["processed_val"], mode="val")
    val_loader = DataLoader(val_set,
                            batch_size=cfg["val_batch_size"],
                            num_workers=cfg['val_workers'],
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
        print("Evaluation without pre-trained weights!")

    # evaluation
    vis = vis_cls()
    average_loss = al_cls()
    average_metrics = am_cls(cfg)
    loop = tqdm(enumerate(val_loader), total=len(val_loader), desc="Val", leave=True)
    net.eval()
    with torch.no_grad():
        for i, batch in loop:
            out = net(batch)
            loss_out = loss_net(out, batch)

            average_loss.update(loss_out)
            average_metrics.update(out, batch)

            if (i + 1) % 50 == 0:
                post_loss = average_loss.get()
                post_metrics = average_metrics.get()
                if args.resume:
                    vis.draw(batch, out, cfg, save=True, show=False)
                loop.set_postfix_str(f'total_loss={post_loss["loss"]:.3f}, minFDE={post_metrics["minfde_k"]:.3f}')

        post_loss = average_loss.get()
        post_metrics = average_metrics.get()
        for k, v in post_loss.items():
            print(f"{k}: {v:.3f}")
        for k, v in post_metrics.items():
            print(f"{k}: {v:.3f}")


if __name__ == "__main__":
    main()
