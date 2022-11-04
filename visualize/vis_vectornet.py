import gc
import os
import numpy as np
from matplotlib import pyplot as plt
from visualize.vis_utils import Visualizer


class Vis(Visualizer):
    def __init__(self):
        super(Vis, self).__init__()

    def draw(self, data, post_out, cfg, save=True, show=False):
        orig = data["orig"][0].detach().cpu().numpy()
        rot = data["rot"][0].detach().cpu().numpy()

        trajs_obs = data["trajs_obs"][0][2:].detach().cpu().numpy().dot(rot.T) + orig
        pad_obs = data["pad_obs"][0][2:].detach().cpu().numpy()

        trajs_av = data["trajs_obs"][0][1].detach().cpu().numpy().dot(rot.T) + orig

        trajs_agent_fut = data["trajs_fut"][0][0].detach().cpu().numpy().dot(rot.T) + orig
        trajs_agent_obs = data["trajs_obs"][0][0].detach().cpu().numpy().dot(rot.T) + orig
        preds = post_out['reg'][0][0].detach().cpu().numpy().dot(rot.T) + orig
        cls = post_out["cls"][0][0].detach().cpu().numpy()

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_title(data["seq_id"][0], fontweight="bold")
        ax.axis("equal")

        x_min = orig[0] - 50
        x_max = orig[0] + 50
        y_min = orig[1] - 50
        y_max = orig[1] + 50
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        self.argo_vis.show_surrounding_elements(ax, data["city"][0], x_min, x_max, y_min, y_max)
        self.plot_obs_trajs(ax, trajs_obs, pad_obs)
        self.plot_av_trajs(ax, trajs_av)
        self.plot_agent_trajs(ax, trajs_agent_fut, trajs_agent_obs, preds, cls)

        # plt.legend()
        # ax.axes.xaxis.set_visible(False)
        # ax.axes.yaxis.set_visible(False)
        if save:
            plt.savefig(os.path.join(cfg["images"], str(data["seq_id"][0]) + '.png'), dpi=250)
        if show:
            plt.show()
        plt.close(fig)
