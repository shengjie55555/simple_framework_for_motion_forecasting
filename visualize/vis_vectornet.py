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

    def plot_agent_trajs(self, ax, trajs_fut, trajs_obs, preds):
        ax.plot(trajs_obs[:, 0], trajs_obs[:, 1], "-", color=self.color_dict["AGENT"], linewidth=1)
        ax.plot(trajs_obs[-1, 0], trajs_obs[-1, 1], "o", color=self.color_dict["AGENT"], linewidth=1)
        ax.plot(trajs_fut[:, 0], trajs_fut[:, 1], "-", color=self.color_dict["AGENT"], linewidth=1)
        for i in range(preds.shape[0]):
            ax.plot(preds[i, :, 0], preds[i, :, 1], "-", alpha=0.5, color='b', linewidth=1)

    def plot_lane_graph(self, ax, lanes):
        for cl in lanes:
            for j in range(cl.shape[0] - 1):
                pt0 = cl[j]
                pt1 = cl[j + 1]
                ax.arrow(pt0[0], pt0[1], (pt1 - pt0)[0], (pt1 - pt0)[1], edgecolor=None,
                         color=self.color_dict["LANE"], alpha=0.3)

    def draw(self, data, post_out, cfg, save=True, show=False):
        orig = data["orig"][0].detach().cpu().numpy()
        rot = data["rot"][0].detach().cpu().numpy()

        trajs_obs = data["trajs_obs"][0][2:].detach().cpu().numpy().dot(rot.T) + orig
        pad_obs = data["pad_obs"][0][2:].detach().cpu().numpy()

        trajs_av = data["trajs_obs"][0][1].detach().cpu().numpy().dot(rot.T) + orig

        trajs_agent_fut = data["trajs_fut"][0][0].detach().cpu().numpy().dot(rot.T) + orig
        trajs_agent_obs = data["trajs_obs"][0][0].detach().cpu().numpy().dot(rot.T) + orig
        preds = post_out['reg'][0][0].detach().cpu().numpy().dot(rot.T) + orig

        lanes = data["graph"][0]["feats"].detach().cpu().numpy()
        lanes[:, :, :2] = lanes[:, :, :2].dot(rot.T) + np.expand_dims(orig, axis=0)

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_title(data["seq_id"][0], fontweight="bold")
        ax.axis("equal")

        x_min = orig[0] - 100
        x_max = orig[0] + 100
        y_min = orig[1] - 100
        y_max = orig[1] + 100
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        self.plot_lane_graph(ax, lanes)
        self.plot_agent_trajs(ax, trajs_agent_fut, trajs_agent_obs, preds)
        self.plot_av_trajs(ax, trajs_av)
        self.plot_obs_trajs(ax, trajs_obs, pad_obs)
        if save:
            plt.savefig(os.path.join(cfg["images"], str(data["seq_id"][0]) + '.png'), dpi=250)
        if show:
            plt.show()
        plt.close(fig)
