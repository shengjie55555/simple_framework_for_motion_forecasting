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

        trajs_obs = data["locs"][0][1:, :, :2].detach().cpu().numpy()
        trajs_obs = np.matmul(trajs_obs, rot) + orig
        pad_obs = data["feats"][0][1:, :, -1].detach().cpu().numpy().astype(np.bool_)

        trajs_agent_fut = data["gt_preds"][0][0].detach().cpu().numpy()
        trajs_agent_obs = data["locs"][0][0, :, :2].detach().cpu().numpy()
        trajs_agent_obs = np.matmul(trajs_agent_obs, rot) + orig
        preds = post_out['reg'][0][0].detach().cpu().numpy()
        cls = post_out["cls"][0][0].detach().cpu().numpy()

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_title(data["argo_id"][0], fontweight="bold")
        ax.axis("equal")

        x_min = orig[0] - 50
        x_max = orig[0] + 50
        y_min = orig[1] - 50
        y_max = orig[1] + 50
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        self.argo_vis.show_surrounding_elements(ax, data["city"][0], x_min, x_max, y_min, y_max)
        self.plot_obs_trajs(ax, trajs_obs, pad_obs)
        self.plot_agent_trajs(ax, trajs_agent_fut, trajs_agent_obs, preds, cls)

        # plt.legend()
        # ax.axes.xaxis.set_visible(False)
        # ax.axes.yaxis.set_visible(False)
        if save:
            plt.savefig(os.path.join(cfg["images"], str(data["argo_id"][0]) + '.png'), dpi=250)
        if show:
            plt.show()
        plt.close(fig)
