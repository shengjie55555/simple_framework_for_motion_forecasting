import os
import argparse
import torch
import warnings
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.data_utils import collate_fn, create_dirs, save_log
from utils.log_utils import Logger
from utils.train_utils import worker_init_fn, init_seeds, load_prev_weights, save_ckpt
from utils.train_utils import Loader, update_cfg

warnings.filterwarnings("ignore")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="train", type=str, help="train/val/test")
    parser.add_argument("--train_dir", required=True, default="", type=str,
                        help="Path to the file which has features for training.")
    parser.add_argument("--val_dir", required=True, default="", type=str,
                        help="Path to the file which has features for evaluation.")
    parser.add_argument("--obs_len", default=20, type=int, help="Observed length of the historical trajectory.")
    parser.add_argument("--pred_len", default=30, type=int, help="Predicted horizon.")
    parser.add_argument("--train_batch_size", default=1, type=int, help="Training batch size.")
    parser.add_argument("--val_batch_size", default=1, type=int, help="Val batch size.")
    parser.add_argument("--train_epochs", default=32, type=int, help="Number of epochs for training.")
    parser.add_argument("--num_val", default=2, type=int, help="Validation intervals.")
    parser.add_argument("--num_display", default=10, type=int, help="Logger interval.")
    parser.add_argument("--workers", default=0, type=int, help="Number of workers for loading data.")
    parser.add_argument("--model", default="VectorNet", type=str, help="Name of the selected model.")
    parser.add_argument("--use_cuda", default=True, type=bool, help="Use CUDA for acceleration.")
    parser.add_argument("--logger_writer", action="store_true", default=False, help="Enable Tensorboard.")
    parser.add_argument("--resume", action="store_true", default=False, help="Resume training.")
    parser.add_argument("--model_path", required=False, type=str, help="Path to the saved model")
    args = parser.parse_args()
    return args


def val(config, data_loader, net, loss_net, logger, average_loss, average_metrics, vis, epoch, rank=0):
    average_loss.reset()
    average_metrics.reset()
    loop = tqdm(enumerate(data_loader), total=len(data_loader), desc="Val", leave=False, disable=rank)
    net.eval()
    with torch.no_grad():
        for i, batch in loop:
            out = net(batch)
            loss_out = loss_net(out, batch)

            average_loss.update(loss_out)
            average_metrics.update(out, batch)

            if (i + 1) % 100 == 0 and rank == 0:
                vis.draw(batch, out, config, save=True, show=False)

        post_loss = average_loss.get()
        post_metrics = average_metrics.get()
        if rank == 0:
            logger.add_dict(post_loss, epoch, "val_")
            logger.add_dict(post_metrics, epoch, "val_")
            save_log(epoch, post_loss, post_metrics, config, mode="val")
    net.train()
    return post_loss, post_metrics


def main():
    args = get_args()
    loader = Loader(args.model)
    cfg, dataset_cls, model_cls, loss_cls, al_cls, am_cls, vis_cls = loader.load()
    cfg = update_cfg(args, cfg)
    print("Args: ", args)
    print("Config: ", cfg)
    print(dataset_cls, model_cls, loss_cls, al_cls, am_cls, vis_cls)

    # set seed
    init_seeds(0)

    # create directories
    create_dirs(cfg)

    # select device
    if args.use_cuda and torch.cuda.is_available():
        device = torch.device("cuda", 0)
    else:
        device = torch.device('cpu')

    # tensorboard logger
    logger = Logger(enable=args.logger_writer, log_dir=cfg["log_dir"])

    # dataset & dataloader
    train_set = dataset_cls(cfg["processed_train"], mode="train")
    train_loader = DataLoader(train_set,
                              batch_size=cfg["train_batch_size"],
                              num_workers=cfg['train_workers'],
                              shuffle=True,
                              pin_memory=True,
                              collate_fn=collate_fn,
                              worker_init_fn=worker_init_fn)
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
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg["milestones"],
                                                     gamma=cfg["gamma"], last_epoch=-1)

    # resume
    start_epoch = 0
    n_iter = 0
    if args.resume:
        if not args.model_path.endswith(".pth"):
            assert False, "Model path error - {}".format(args.model_path)
        else:
            start_epoch = int(os.path.basename(args.model_path).split(".")[0])
            n_iter = len(train_loader) * start_epoch
            load_prev_weights(net, args.model_path, cfg, optimizer)
            print("Training from checkpoint: {} - {}".format(start_epoch, n_iter))
    else:
        print("Training from beginning!")

    # training
    vis = vis_cls()
    average_loss = al_cls()
    average_metrics = am_cls(cfg)
    epoch_loop = tqdm(range(start_epoch, cfg["epoch"]), leave=True)
    for epoch in epoch_loop:
        average_loss.reset()
        average_metrics.reset()
        net.train()
        num_batches = len(train_loader)
        epoch_per_batch = 1.0 / num_batches

        iter_loop = tqdm(enumerate(train_loader), total=len(train_loader), desc="Train", leave=False)
        iter_loop.set_postfix_str(f'lr={optimizer.param_groups[0]["lr"]:.5f}, total_loss="???", minFDE="???"')
        for i, batch in iter_loop:
            n_iter += cfg["train_batch_size"]
            epoch += epoch_per_batch
            epoch_loop.set_description(f'Process [{epoch:.3f}/{cfg["epoch"]}]')

            out = net(batch)
            loss_out = loss_net(out, batch)

            optimizer.zero_grad()
            loss_out['loss'].backward()
            optimizer.step()
            scheduler.step(epoch)

            average_loss.update(loss_out)
            average_metrics.update(out, batch)

            if (i + 1) % (num_batches // cfg["num_display"]) == 0 or (i + 1) == len(train_loader):
                post_loss = average_loss.get()
                post_metrics = average_metrics.get()
                logger.add_dict(post_loss, n_iter)
                logger.add_dict(post_metrics, n_iter)
                save_log(epoch, post_loss, post_metrics, cfg, mode="train")
                iter_loop.set_postfix_str(
                    f'lr={optimizer.param_groups[0]["lr"]:.5f}, total_loss={post_loss["loss"]:.3f}, '
                    f'minFDE={post_metrics["minfde_k"]:.3f}')

        save_ckpt(net, optimizer, round(epoch), cfg["save_dir"])

        if round(epoch) % cfg['num_val'] == 0:
            val(cfg, val_loader, net, loss_net, logger, average_loss, average_metrics, vis, round(epoch))


if __name__ == "__main__":
    main()
