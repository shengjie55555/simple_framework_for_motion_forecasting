import os


config = dict()
model_name = "mhl"

# net
config["n_agent"] = 5            # (x, y, dx, dy, mask)
config["n_feature"] = 128        # feature dimension
config["n_agent_layer"] = 3      # number of layers in agent encoder's backbone
config["map2map_dist"] = 7.5     # map to map threshold
config["map2agent_dist"] = 15.0  # map to agent threshold
config["num_scales"] = 6         # dilated lane

config["num_mode"] = 6           # num of predicted trajectories
config["pred_len"] = 30          # predicted horizon
config["obs_len"] = 20           # observed horizon

config["cls_th"] = 10.0          # Calculate cls loss for those samples whose minFDE is lower than this threshold.
config["cls_ignore"] = 0.2       # nms for different goals
config["mgn"] = 0.2              # margin

config["cls_coef"] = 1.0         # cls loss coefficient
config["reg_coef"] = 1.0         # reg loss coefficient

# weights
config["ignored_modules"] = []   # Do not load weights of these modules.

# Processed Dataset
config["processed_train"] = "./processed/train/"
config["processed_val"] = "./processed/val/"
config["processed_test"] = "./processed/test/"

# train
config["scaling_ratio"] = [0.8, 1.25]   # set [1, 1] to stop
config["past_motion_dropout"] = 11      # set 1 to stop
config["flip"] = 0.2                    # set 0 to stop

config["epoch"] = 50
config["lr"] = 0.001
config["optimizer"] = "Adam"
config["weight_decay"] = 0.005
config["scheduler"] = "MultiStepLR"
config["T_0"] = 2
config["T_mult"] = 1
config["milestones"] = [30, 48]
config["gamma"] = 0.1
config["train_batch_size"] = 1
config["train_workers"] = 0

config["num_display"] = 100
config["num_val"] = 2
config["save_dir"] = os.path.join("../results", model_name, "weights/")
config["cfg"] = os.path.join("../results", model_name, "cfg.txt")
config["images"] = os.path.join("../results", model_name, "images/")
config["train_log"] = os.path.join("../results", model_name, "train_log.csv")
config["val_log"] = os.path.join("../results", model_name, "val_log.csv")
config["log_dir"] = "logs/" + model_name
config["log_columns"] = ["epoch", "cls_loss", "reg_loss", "loss",
                         "minade_1", "minfde_1", "mr_1", "brier_fde_1",
                         "minade_k", "minfde_k", "mr_k", "brier_fde_k"]

# val
config["val_batch_size"] = 32
config["val_workers"] = config["train_workers"]

# test
config["test_batch_size"] = 32
config["test_workers"] = config["train_workers"]
config["competition_files"] = os.path.join("../results", model_name, "competition/")
