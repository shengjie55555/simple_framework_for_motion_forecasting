import os
from os.path import expanduser
import time
from typing import Any, Dict, List, Tuple
import random
import copy
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt

from matplotlib import cm
from matplotlib.path import Path
from shapely.geometry import LineString, Point, Polygon, MultiPolygon
from shapely.ops import cascaded_union, nearest_points, unary_union
from scipy import sparse, spatial
from argoverse.map_representation.map_api import ArgoverseMap


class PreProcess(object):
    def __init__(self, args):
        self.args = args
        self.debug = args.debug
        self.viz = args.viz
        self.mode = args.mode

        self.MIN_OBS_FRAMES = 10
        self.MAP_RADIUS = 120.0
        self.COMPL_RANGE = 30.0
        ''' DA: drivable area '''
        self.DA_RESOLUTION = 1.0
        self.DA_NUM_SCALES = 4
        ''' LS: lane segment '''
        self.LS_SEG_LENGTH = 2.0
        self.LS_NUM_SCALES = 6
        self.LS_CROSS_DIST = 6.0

        self.DALS_DIST_THRES = 2.25

        self.argo_map = ArgoverseMap()

    def process(self, seq_id, df):
        city_name = df['CITY_NAME'].values[0]

        # get trajectories
        ts, trajs, pad_flags = self.get_trajectories(df, city_name)

        # get origin and rot
        orig, rot = self.get_origin_rotation(trajs[0])
        trajs = (trajs - orig).dot(rot)

        # get lane graph
        lane_ids = self.get_related_lanes(seq_id, city_name, orig, expand_dist=self.MAP_RADIUS)
        graph = self.construct_lane_segment_layer(city_name, lane_ids, orig, rot)

        # collect data
        data = [[seq_id, city_name, orig, rot, ts, trajs, pad_flags, graph]]
        headers = ["SEQ_ID", "CITY_NAME", "ORIG", "ROT", "TIMESTAMP", "TRAJS", "PAD_FLAGS", "GRAPH"]

        # ! For debug
        if self.debug and self.viz:
            _, ax = plt.subplots(figsize=(10, 10))
            ax.axis('equal')
            vis_map = True
            self.plot_trajs(ax, trajs, pad_flags, orig, rot, vis_map=vis_map)
            self.plot_lane_graph(ax, lane_ids, graph, orig, rot, city_name, vis_map=vis_map)
            ax.set_title("{} {}".format(seq_id, city_name))
            plt.show()

        return data, headers

    def get_origin_rotation(self, traj):
        orig = traj[self.args.obs_len - 1]

        # vec = orig - traj[self.args.obs_len - 2]
        vec = orig - traj[0]
        theta = np.arctan2(vec[1], vec[0])
        rot = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])

        return orig, rot

    def get_trajectories(self,
                         df: pd.DataFrame,
                         city_name: str):
        ts = np.sort(np.unique(df['TIMESTAMP'].values)).astype(np.float)
        t_obs = ts[self.args.obs_len - 1]

        agent_traj = df[df["OBJECT_TYPE"] == "AGENT"]
        agent_traj = np.stack((agent_traj['X'].values, agent_traj['Y'].values), axis=1)
        agent_traj[:, 0:2] = agent_traj[:, 0:2]

        av_traj = df[df["OBJECT_TYPE"] == "AV"]
        av_traj = np.stack((av_traj['X'].values, av_traj['Y'].values), axis=1)
        av_traj[:, 0:2] = av_traj[:, 0:2]

        assert len(agent_traj) == len(av_traj), "Shape error for AGENT and AV, AGENT: {}, AV: {}".format(
            agent_traj.shape, av_traj.shape)

        trajs = [agent_traj, av_traj]
        pad_flags = [np.zeros_like(ts), np.zeros_like(ts)]

        track_ids = np.unique(df["TRACK_ID"].values)
        for idx in track_ids:
            mot_traj = df[df["TRACK_ID"] == idx]

            if mot_traj['OBJECT_TYPE'].values[0] == 'AGENT' or mot_traj['OBJECT_TYPE'].values[0] == 'AV':
                continue

            ts_mot = np.array(mot_traj['TIMESTAMP'].values).astype(np.float)
            mot_traj = np.stack((mot_traj['X'].values, mot_traj['Y'].values), axis=1)

            # ~ remove traj after t_obs
            if np.all(ts_mot > t_obs):
                continue

            _, idcs, _ = np.intersect1d(ts, ts_mot, return_indices=True)
            padded = np.ones_like(ts)
            padded[idcs] = 0

            mot_traj_pad = np.full(agent_traj[:, :2].shape, None)
            mot_traj_pad[idcs] = mot_traj

            mot_traj_pad = self.padding_traj_nn(mot_traj_pad)
            assert np.all(mot_traj_pad[idcs] == mot_traj), "Padding error"

            mot_traj = np.stack((mot_traj_pad[:, 0], mot_traj_pad[:, 1]), axis=1)
            mot_traj[:, 0:2] = mot_traj[:, 0:2]
            trajs.append(mot_traj)
            pad_flags.append(padded)

        ts = (ts - ts[0]).astype(np.float32)
        trajs = np.array(trajs).astype(np.float32)  # [N, 50(20), 2]
        pad_flags = np.array(pad_flags).astype(np.int16)  # [N, 50(20)]

        return ts, trajs, pad_flags

    @staticmethod
    def padding_traj_nn(traj):
        n = len(traj)

        # forward
        buff = None
        for i in range(n):
            if np.all(buff == None) and np.all(traj[i] != None):
                buff = traj[i]
            if np.all(buff != None) and np.all(traj[i] == None):
                traj[i] = buff
            if np.all(buff != None) and np.all(traj[i] != None):
                buff = traj[i]

        # backward
        buff = None
        for i in reversed(range(n)):
            if np.all(buff == None) and np.all(traj[i] != None):
                buff = traj[i]
            if np.all(buff != None) and np.all(traj[i] == None):
                traj[i] = buff
            if np.all(buff != None) and np.all(traj[i] != None):
                buff = traj[i]

        return traj

    def get_related_lanes(self, seq_id, city_name, orig, expand_dist):
        # Get all lane candidates within a bubble
        # Keep expanding the bubble until at least 1 lane is found
        curr_lane_candidates = []
        NEARBY_DIST = 5.0
        while len(curr_lane_candidates) < 1:
            curr_lane_candidates = self.argo_map.get_lane_ids_in_xy_bbox(orig[0], orig[1], city_name, NEARBY_DIST)
            NEARBY_DIST *= 2
        assert len(curr_lane_candidates) > 0, "No nearby lanes found!"

        init_lane_ids = []
        init_lane_dist = []
        orig_shp = Point(orig)
        for lane_id in curr_lane_candidates:
            cl = self.argo_map.get_lane_segment_centerline(lane_id, city_name)
            dist_2_cl = LineString(cl).distance(orig_shp)
            init_lane_dist.append(dist_2_cl)
            if dist_2_cl <= NEARBY_DIST:
                init_lane_ids.append((lane_id, dist_2_cl))
        if not init_lane_ids:
            nearest_id = np.array(init_lane_dist).argmin()
            init_lane_ids.append((curr_lane_candidates[nearest_id], init_lane_dist[nearest_id]))

        # ~ graph search until reaching a certain distance
        lanes = self.argo_map.city_lane_centerlines_dict[city_name]

        closed_set = []
        open_set = dict()  # (idx, cost)
        node_list = dict()

        def get_process_cost(idx_s, idx_e, action):
            cost = 0.0
            if action in ['pre', 'suc']:
                cost += (LineString(lanes[idx_s].centerline).length + LineString(lanes[idx_e].centerline).length) / 2
                # cost += LineString(lanes[idx_e].centerline).length
            if action in ['left', 'right']:
                cost += 3.0
            if lanes[idx_e].turn_direction in ['RIGHT', 'LEFT']:
                cost += LineString(lanes[idx_e].centerline).length
            return cost

        for lane_id, dist in init_lane_ids:
            open_set[lane_id] = dist
            node_list[lane_id] = (-1, dist)  # (parent_id, cost)
        open_set = {k: v for k, v in sorted(open_set.items(), key=lambda item: item[1])}  # sort by value

        while len(open_set) != 0:
            idx_expand, cost_expand = next(iter(open_set.items()))
            if cost_expand > expand_dist:
                break

            open_set.pop(idx_expand)
            closed_set.append(idx_expand)

            # get child and cost
            child_nodes = []
            lane_expand = lanes[idx_expand]
            if lane_expand.predecessors is not None:
                for nbr_idx in lane_expand.predecessors:
                    child_nodes.append((nbr_idx, 'pre'))
            if lane_expand.successors is not None:
                for nbr_idx in lane_expand.successors:
                    child_nodes.append((nbr_idx, 'suc'))
            if lane_expand.l_neighbor_id is not None:
                child_nodes.append((lane_expand.l_neighbor_id, 'left'))
            if lane_expand.r_neighbor_id is not None:
                child_nodes.append((lane_expand.r_neighbor_id, 'right'))

            for nbr_idx, act in child_nodes:
                cost_proc = get_process_cost(idx_expand, nbr_idx, action=act)
                cost_new = cost_proc + node_list[idx_expand][1]

                if nbr_idx in node_list:
                    # visited
                    if nbr_idx in closed_set:
                        continue
                    if cost_new < node_list[nbr_idx][1]:
                        # update node info
                        node_list[nbr_idx] = (idx_expand, cost_new)
                        # update open set
                        open_set[nbr_idx] = cost_new
                        open_set = {k: v for k, v in sorted(
                            open_set.items(), key=lambda item: item[1])}  # sort by value
                else:
                    # not visited, add new node
                    node_list[nbr_idx] = (idx_expand, cost_new)
                    open_set[nbr_idx] = cost_new
                    open_set = {k: v for k, v in sorted(open_set.items(), key=lambda item: item[1])}  # sort by value

        # near in geometry
        nearby_tmp = self.argo_map.get_lane_ids_in_xy_bbox(orig[0], orig[1], city_name, self.COMPL_RANGE)
        nearby_ids = []
        for lane_id in nearby_tmp:
            cl = self.argo_map.get_lane_segment_centerline(lane_id, city_name)
            dist_2_cl = LineString(cl).distance(orig_shp)
            if dist_2_cl <= self.COMPL_RANGE:
                nearby_ids.append(lane_id)
        selected_lane_ids = copy.deepcopy(list(node_list.keys()))  # return all visited lanes
        selected_lane_ids += nearby_ids
        selected_lane_ids = list(set(selected_lane_ids))

        return copy.deepcopy(selected_lane_ids)

    def construct_lane_segment_layer(self, city_name, lane_ids, orig, rot):
        """
            ctrs            torch.Tensor        (N_l, 2)            center of lane node
            feats           torch.Tensor        (N_l, 10, 2)        lane centerline
            turn            torch.Tensor        (N_l, 2)            flag for turning
            control         torch.Tensor        (N_l, )             flag for traffic controlled
            intersection    torch.Tensor        (N_l, )             flag for intersection
        """
        ctrs, feats, turn, control, intersect = [], [], [], [], []
        lanes = self.argo_map.city_lane_centerlines_dict[city_name]
        for lane_id in lane_ids:
            lane = lanes[lane_id]

            # ~ resample centerline
            cl_raw = lane.centerline
            cl_raw[:, 0:2] = (cl_raw[:, 0:2] - orig).dot(rot)

            ctrs.append(cl_raw.mean(axis=0, keepdims=True))  # middle point
            feats.append(np.expand_dims(cl_raw, 0))

            x = np.zeros((1, 2), np.float32)
            if lane.turn_direction == 'LEFT':
                x[:, 0] = 1
            elif lane.turn_direction == 'RIGHT':
                x[:, 1] = 1
            else:
                pass
            turn.append(x)

            control.append(lane.has_traffic_control * np.ones(1, np.float32))
            intersect.append(lane.is_intersection * np.ones(1, np.float32))

        node_idcs = []  # List of range
        count = 0
        for i, ctr in enumerate(ctrs):
            node_idcs.append(range(count, count + len(ctr)))
            count += len(ctr)
        num_nodes = count

        ctrs = np.concatenate(ctrs, 0)
        feats = np.concatenate(feats, 0)

        graph = dict()
        graph['num_nodes'] = num_nodes
        graph['ctrs'] = ctrs.astype(np.float32)
        graph['feats'] = feats.astype(np.float32)
        graph['turn'] = np.concatenate(turn, 0).astype(np.int16)
        graph['control'] = np.concatenate(control, 0).astype(np.int16)
        graph['intersect'] = np.concatenate(intersect, 0).astype(np.int16)

        return graph

    def plot_trajs(self, ax, trajs, pad_flags, orig, rot, vis_map=True):
        if not vis_map:
            rot = np.eye(2)
            orig = np.zeros(2)

        for i, traj in enumerate(trajs):
            zorder = 10
            if i == 0:  # agent
                clr = "r"
                zorder = 20
            else:
                clr = "orange"

            traj = traj.dot(rot.T) + orig
            ax.plot(traj[:, 0], traj[:, 1], marker=".", alpha=0.5, color=clr, zorder=zorder)
            ax.text(traj[self.args.obs_len, 0], traj[self.args.obs_len, 1], str(i))
            ax.scatter(traj[:, 0], traj[:, 1], s=list((1 - pad_flags[i]) * 50 + 1), color="b")

    def plot_lane_graph(self, ax, lane_ids, graph, orig, rot, city_name, vis_map=True):
        x_min = orig[0] - 100
        x_max = orig[0] + 100
        y_min = orig[1] - 100
        y_max = orig[1] + 100
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        if vis_map:
            seq_lane_props = self.argo_map.city_lane_centerlines_dict[city_name]

            for idx in lane_ids:
                lane_cl = seq_lane_props[idx].centerline
                lane_polygon = self.argo_map.get_lane_segment_polygon(
                    idx, city_name)
                # ax.add_patch(
                #     Polygon(lane_polygon[:, 0:2], color='gray', alpha=0.1, edgecolor=None))

                pt = lane_cl[0]
                vec = lane_cl[1] - lane_cl[0]
                ax.arrow(pt[0],
                         pt[1],
                         vec[0],
                         vec[1],
                         alpha=0.5,
                         color='grey',
                         width=0.1,
                         zorder=1)
        else:
            rot = np.eye(2)
            orig = np.zeros(2)

        ctrs = graph["ctrs"]
        ctrs[:, :2] = ctrs[:, :2].dot(rot.T) + orig
        ax.scatter(ctrs[:, 0], ctrs[:, 1], color="b", s=2, alpha=0.5)

        feats = graph["feats"]
        feats[:, :, :2] = feats[:, :, :2].dot(rot.T) + np.expand_dims(orig, axis=0)
        for cl in feats:
            for j in range(cl.shape[0] - 1):
                pt0 = cl[j]
                pt1 = cl[j + 1]
                ax.arrow(pt0[0], pt0[1], (pt1 - pt0)[0], (pt1 - pt0)[1], edgecolor=None, color="deepskyblue", alpha=0.3)
