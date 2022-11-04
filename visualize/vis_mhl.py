import gc
import os
import numpy as np
from matplotlib import pyplot as plt


class Vis(object):
    def __init__(self):
        super(Vis, self).__init__()
        self.color_dict = {
            "AGENT": "#d33e4c",
            "OTHERS": "#d3e8ef",
            "AV": "#007672",
            "LANE": "deepskyblue"
        }

    def plot_obs_trajs(self, ax, trajs_obs, pad_obs):
        masks = pad_obs.astype(np.bool_)
        for i in range(trajs_obs.shape[0]):
            traj_obs = trajs_obs[i, masks[i]]
            ax.plot(traj_obs[:, 0], traj_obs[:, 1], "-", color=self.color_dict["OTHERS"], linewidth=1)
            ax.plot(traj_obs[-1, 0], traj_obs[-1, 1], "o", color=self.color_dict["OTHERS"], linewidth=1)

    def plot_av_trajs(self, ax, trajs_obs):
        ax.plot(trajs_obs[:, 0], trajs_obs[:, 1], "-", color=self.color_dict["AV"], linewidth=1)
        ax.plot(trajs_obs[-1, 0], trajs_obs[-1, 1], "o", color=self.color_dict["AV"], linewidth=1)

    def plot_agent_trajs(self, ax, trajs_fut, trajs_obs, preds, cls):
        ax.plot(trajs_obs[:, 0], trajs_obs[:, 1], "-", color=self.color_dict["AGENT"], linewidth=1)
        ax.plot(trajs_obs[-1, 0], trajs_obs[-1, 1], "o", color=self.color_dict["AGENT"], linewidth=1)
        ax.plot(trajs_fut[:, 0], trajs_fut[:, 1], "-", color=self.color_dict["AGENT"], linewidth=1)
        ax.scatter(trajs_fut[-1, 0], trajs_fut[-1, 1], marker="X", color=self.color_dict["AGENT"])
        for i in range(preds.shape[0]):
            ax.plot(preds[i, :, 0], preds[i, :, 1], "-", alpha=0.5, color='b', linewidth=1)
            ax.scatter(preds[i, -1, 0], preds[i, -1, 1], marker="X", alpha=cls[i], color='b')

    def plot_lane_graph(self, ax, ctrs, feats):
        ax.scatter(ctrs[:, 0], ctrs[:, 1], color="b", s=2, alpha=0.5)
        for j in range(feats.shape[0]):
            vec = feats[j]
            pt0 = ctrs[j] - vec / 2
            pt1 = ctrs[j] + vec / 2
            ax.arrow(pt0[0], pt0[1], (pt1 - pt0)[0], (pt1 - pt0)[1],
                     edgecolor=None, color=self.color_dict["LANE"], alpha=0.3)

    def draw(self, data, post_out, cfg, save=True, show=False):
        orig = data["orig"][0].detach().cpu().numpy()
        rot = data["rot"][0].detach().cpu().numpy()

        trajs_obs = data["locs"][0][1:, :, :2].detach().cpu().numpy()
        trajs_obs = np.matmul(trajs_obs, rot) + orig
        pad_obs = data["feats"][0][1:, :, -1].detach().cpu().numpy().astype(np.bool_)

        trajs_agent_fut = data["gt_preds"][0][0].detach().cpu().numpy()
        trajs_agent_obs = data["locs"][0][0, :, :2].detach().cpu().numpy()
        trajs_agent_obs = np.matmul(trajs_agent_obs, rot) + orig
        preds = post_out['reg'][0][0].detach().cpu().numpy()
        cls = post_out["cls"][0][0].detach().cpu().numpy()

        ctrs = data["graph"][0]["ctrs"].detach().cpu().numpy()
        ctrs = np.matmul(ctrs, rot) + orig
        feats = data["graph"][0]["feats"].detach().cpu().numpy()
        feats = np.matmul(feats, rot)

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_title(data["argo_id"][0], fontweight="bold")
        ax.axis("equal")

        x_min = orig[0] - 100
        x_max = orig[0] + 100
        y_min = orig[1] - 100
        y_max = orig[1] + 100
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        self.plot_lane_graph(ax, ctrs, feats)
        self.plot_agent_trajs(ax, trajs_agent_fut, trajs_agent_obs, preds, cls)
        self.plot_obs_trajs(ax, trajs_obs, pad_obs)
        if save:
            plt.savefig(os.path.join(cfg["images"], str(data["argo_id"][0]) + '.png'), dpi=250)
        if show:
            plt.show()
        plt.close(fig)
