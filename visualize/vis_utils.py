import abc
import numpy as np
from matplotlib.patches import Polygon
from argoverse.map_representation.map_api import ArgoverseMap


class Visualizer(object):
    def __init__(self):
        super(Visualizer, self).__init__()
        self.color_dict = {
            "AGENT_OBS": "#d33e4c",
            "AGENT_FUT": "#269626",
            "OTHERS": "#FFD500",
            "AV": "#007672",
        }
        self.argo_vis = ArgoMapVisualizer()

    def plot_obs_trajs(self, ax, trajs_obs, pad_obs):
        masks = pad_obs.astype(np.bool_)
        for i in range(trajs_obs.shape[0]):
            traj_obs = trajs_obs[i, masks[i]]
            ax.plot(traj_obs[:, 0], traj_obs[:, 1], "-", color=self.color_dict["OTHERS"], linewidth=1,
                    label="historical trajectory (surrounding agents)" if i == 0 else "")
            ax.plot(traj_obs[-1, 0], traj_obs[-1, 1], "o", color=self.color_dict["OTHERS"], linewidth=1)

    def plot_av_trajs(self, ax, trajs_obs):
        ax.plot(trajs_obs[:, 0], trajs_obs[:, 1], "-", color=self.color_dict["AV"], linewidth=1,
                label="historical trajectory (autonomous vehicle)")
        ax.plot(trajs_obs[-1, 0], trajs_obs[-1, 1], "o", color=self.color_dict["AV"], linewidth=1)

    def plot_agent_trajs(self, ax, trajs_fut, trajs_obs, preds, cls):
        ax.plot(trajs_obs[:, 0], trajs_obs[:, 1], "-", color=self.color_dict["AGENT_OBS"], linewidth=1,
                label="historical trajectory (target vehicle)")
        ax.plot(trajs_obs[-1, 0], trajs_obs[-1, 1], "o", color=self.color_dict["AGENT_OBS"], linewidth=1)
        ax.plot(trajs_fut[:, 0], trajs_fut[:, 1], "-", color=self.color_dict["AGENT_FUT"], linewidth=1,
                label="future trajectory (target vehicle)")
        ax.scatter(trajs_fut[-1, 0], trajs_fut[-1, 1], marker="X", color=self.color_dict["AGENT_FUT"])
        for i in range(preds.shape[0]):
            ax.plot(preds[i, :, 0], preds[i, :, 1], "-", alpha=0.5, color='b', linewidth=1,
                    label="predicted trajectories (target vehicle)" if i == 0 else "")
            ax.scatter(preds[i, -1, 0], preds[i, -1, 1], marker="X", alpha=1, color='b')
            ax.text(preds[i, -1, 0], preds[i, -1, 1], '{:.2f}'.format(cls[i]), zorder=10)

    @abc.abstractmethod
    def draw(self, data, post_out, cfg, save=True, show=False):
        pass


class ArgoMapVisualizer:
    def __init__(self):
        self.argo_map = ArgoverseMap()

    def show_lane_center_lines(self, ax, city_name, lane_ids, clr='g', alpha=0.2, show_lane_ids=False):
        seq_lane_props = self.argo_map.city_lane_centerlines_dict[city_name]

        for idx in lane_ids:
            lane_cl = seq_lane_props[idx].centerline
            ax.plot(lane_cl[:, 0], lane_cl[:, 1], color=clr, alpha=alpha, linewidth=5)

            if show_lane_ids:
                m_pt = lane_cl[int(lane_cl.shape[0] / 2)]
                ax.text(m_pt[0], m_pt[1], idx, color='b')

    def show_lanes(self, ax, city_name, lane_ids, show_lane_ids=False):
        seq_lane_props = self.argo_map.city_lane_centerlines_dict[city_name]

        for idx in lane_ids:
            lane_cl = seq_lane_props[idx].centerline
            lane_polygon = self.argo_map.get_lane_segment_polygon(idx, city_name)
            ax.add_patch(Polygon(lane_polygon[:, 0:2], color='gray', alpha=1, edgecolor=None))

            pt = lane_cl[0]
            vec = lane_cl[1] - lane_cl[0]
            ax.arrow(pt[0], pt[1], vec[0], vec[1], alpha=0.5, color='grey', width=0.1, zorder=1)
            if show_lane_ids:
                m_pt = lane_cl[int(lane_cl.shape[0] / 2)]
                ax.text(m_pt[0], m_pt[1], idx, color='b')

    def show_surrounding_elements(self, ax, city_name, x_min, x_max, y_min, y_max,
                                  show_lane_ids=False):
        seq_lane_props = self.argo_map.city_lane_centerlines_dict[city_name]
        surrounding_lanes = {}
        for lane_id, lane_props in seq_lane_props.items():
            lane_cl = lane_props.centerline
            if (
                    np.min(lane_cl[:, 0]) < x_max and
                    np.min(lane_cl[:, 1]) < y_max and
                    np.max(lane_cl[:, 0]) > x_min and
                    np.max(lane_cl[:, 1]) > y_min
            ):
                surrounding_lanes[lane_id] = lane_cl

        for idx, lane_cl in surrounding_lanes.items():
            lane_polygon = self.argo_map.get_lane_segment_polygon(idx, city_name)
            ax.add_patch(Polygon(lane_polygon[:, 0:2], color='gray', alpha=0.2, edgecolor="grey"))

            pt = lane_cl[0]
            vec = lane_cl[1] - lane_cl[0]
            vec = vec / np.linalg.norm(vec) * 1.0
            ax.arrow(pt[0], pt[1], vec[0], vec[1], alpha=0.5, color='grey', width=0.1, zorder=1)
            if show_lane_ids:
                m_pt = lane_cl[int(lane_cl.shape[0] / 2)]
                ax.text(m_pt[0], m_pt[1], idx, color='b')

    def show_all_map(self, ax, city_name):
        seq_lane_props = self.argo_map.city_lane_centerlines_dict[city_name]

        for lane_id, lane_props in seq_lane_props.items():
            lane_cl = lane_props.centerline

            pt = lane_cl[0]
            vec = lane_cl[1] - lane_cl[0]

            under_control = self.argo_map.lane_has_traffic_control_measure(
                lane_id, city_name)

            in_intersection = self.argo_map.lane_is_in_intersection(
                lane_id, city_name)

            turn_dir = self.argo_map.get_lane_turn_direction(
                lane_id, city_name)

            cl_clr = 'grey'

            if in_intersection:
                cl_clr = 'orange'

            if turn_dir == 'LEFT':
                cl_clr = 'blue'
            elif turn_dir == 'RIGHT':
                cl_clr = 'green'

            ax.arrow(pt[0],
                     pt[1],
                     vec[0],
                     vec[1],
                     alpha=0.5,
                     color=cl_clr,
                     width=0.1,
                     zorder=1)

            if under_control:
                p_vec = vec / np.linalg.norm(vec) * 1.5
                pt1 = pt + np.array([-p_vec[1], p_vec[0]])
                pt2 = pt + np.array([p_vec[1], -p_vec[0]])
                ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]],
                        color='tomato',
                        alpha=0.5,
                        linewidth=2)

            lane_polygon = self.argo_map.get_lane_segment_polygon(
                lane_id, city_name)
            ax.add_patch(
                Polygon(lane_polygon[:, 0:2],
                        color=cl_clr,
                        alpha=0.1,
                        edgecolor=None))
