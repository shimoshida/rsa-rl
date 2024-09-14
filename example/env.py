import numpy as np
import networkx as nx
from bitarray import bitarray

from rsarl.envs import Env
from rsarl.data import Observation
from rsarl.utils import path_to_edges, k_consecutive_available_slot
from rsarl.utils.fragmentation import entropy


def memory_fragment(path_slot: bitarray):
    """Metric for Heap Fragmentation

    See https://github.com/rhempel/umm_malloc/issues/14

    """
    num, _, veclen_list = k_consecutive_available_slot(path_slot, 1)
    if num == 0:
        metric = 1
    else:
        np_veclen = np.array(veclen_list)
        metric = np.sqrt(np.sum(np_veclen * np_veclen)) / np.sum(np_veclen)

    return metric


class MultiMetricEnv(Env):

    def __init__(self, net, requester, episode_step=None):
        super().__init__(net, requester, episode_step)

    def utilization_reward(self):
        return 1 - self.net.resource_util()

    def current_fragment_reward(self, act):
        edges = path_to_edges(act.path)
        slot_dict = self.net.slot
        # calc
        reward = 0
        for e in edges:
            # current slot
            slot = bitarray(slot_dict[e])
            # fragmentation reward
            reward += memory_fragment(slot)

        # post-processing
        reward = reward / len(edges)
        return reward

    def neighbor_fragment_reward(self):
        """ """
        whole_slot = np.array(list(self.net.slot.values()))
        trans_whole_slot = whole_slot.T
        reward = 0
        for slot_vec in trans_whole_slot:
            bv = bitarray(list(slot_vec))
            reward += memory_fragment(bv)

        reward = reward / self.net.n_slot
        return reward

    def compute_reward(self, action, is_assignable: bool) -> float:
        """ """
        reward = -1
        if is_assignable:
            # blocking-reward
            reward = 0

            crrnt_reward = self.current_fragment_reward(action)
            reward += crrnt_reward

            neighbor_reward = self.neighbor_fragment_reward()
            reward += neighbor_reward

            util_reward = self.utilization_reward()
            reward += util_reward

        return reward

    def step(self, action):
        """Override"""
        self.n_step += 1

        # ------------------------------------------------------------
        # NOTE that reward function executes after path assignment

        # assign path
        is_assignable = self.is_assignable(action)
        if is_assignable:
            self.assign_path(action)

        # reward
        reward = self.compute_reward(action, is_assignable)
        # ------------------------------------------------------------

        # Spend time until next request
        time_interval = self.requester.time_interval()
        self.net.spend_time(time_interval)

        # Generate next path request
        req = self.requester.request()

        # Generate next observation
        self.last_obs = Observation(request=req, net=self.net)

        # check eposode end
        done = self.is_terminate()

        info = {}
        info["is_success"] = is_assignable
        return self.last_obs, reward, done, info
