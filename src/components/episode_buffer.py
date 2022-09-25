import torch as th
import numpy as np
from types import SimpleNamespace as SN


class EfficientStore:
    """
    Stores variable length trajectories of data without padding out each
    episode to the maximum length. Used as a drop in replacement for a
    PyTorch Tensor. Much more memory efficient, but you have to allocate
    memory as you go instead of all at once.
    """
    def __init__(self, size, shape, dtype, device):
        # (batch_size, max_seq_length, *shape), dtype=dtype, device=self.device
        self.size = size
        self.shape = shape
        self.dtype = dtype
        self.device = device

        self.data = [None for _ in range(size)]

    def to(self, device):
        """
        Moves data to specified device. You probably don't want to call this
        after filling the buffer (just call it to start and data will be placed
        on the desired device as it's added)
        """
        self.device = device
        self.data = list(map(lambda x: x.to(device) if x is not None else x,
                             self.data))

    def __getitem__(self, index):
        if type(index) in (slice, int):
            pad_data = self.data[index]
        else:  # assume some sort of iterable
            pad_data = [self.data[i] for i in index]
        return th.nn.utils.rnn.pad_sequence(pad_data, batch_first=True,
                                            padding_value=0)

    def __setitem__(self, index, value):
        self.data[index] = value


class EpisodeBatch:
    def __init__(self,
                 scheme,
                 groups,
                 batch_size,
                 max_seq_length,
                 data=None,
                 preprocess=None,
                 efficient_store=False,
                 device="cpu"):
        self.scheme = scheme.copy()
        self.groups = groups
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.preprocess = {} if preprocess is None else preprocess
        self.efficient_store = efficient_store
        self.device = device

        if data is not None:
            self.data = data
        else:
            self.data = SN()
            self.data.transition_data = {}
            self.data.episode_data = {}
            self._setup_data(self.scheme, self.groups, batch_size, max_seq_length, self.preprocess)

    def _setup_data(self, scheme, groups, batch_size, max_seq_length, preprocess):
        if preprocess is not None:
            for k in preprocess:
                assert k in scheme
                new_k = preprocess[k][0]
                transforms = preprocess[k][1]

                vshape = self.scheme[k]["vshape"]
                dtype = self.scheme[k]["dtype"]
                for transform in transforms:
                    vshape, dtype = transform.infer_output_info(vshape, dtype)

                self.scheme[new_k] = {
                    "vshape": vshape,
                    "dtype": dtype
                }
                if "group" in self.scheme[k]:
                    self.scheme[new_k]["group"] = self.scheme[k]["group"]
                if "episode_const" in self.scheme[k]:
                    self.scheme[new_k]["episode_const"] = self.scheme[k]["episode_const"]

        assert "filled" not in scheme, '"filled" is a reserved key for masking.'
        scheme.update({
            "filled": {"vshape": (1,), "dtype": th.long},
        })

        for field_key, field_info in scheme.items():
            assert "vshape" in field_info, "Scheme must define vshape for {}".format(field_key)
            vshape = field_info["vshape"]
            episode_const = field_info.get("episode_const", False)
            group = field_info.get("group", None)
            dtype = field_info.get("dtype", th.float32)

            if isinstance(vshape, int):
                vshape = (vshape,)

            if group:
                assert group in groups, "Group {} must have its number of members defined in _groups_".format(group)
                shape = (groups[group], *vshape)
            else:
                shape = vshape

            if episode_const:
                self.data.episode_data[field_key] = th.zeros((batch_size, *shape), dtype=dtype, device=self.device)
            else:
                if self.efficient_store:
                    self.data.transition_data[field_key] = EfficientStore(batch_size, shape, dtype, self.device)
                else:
                    self.data.transition_data[field_key] = th.zeros((batch_size, max_seq_length, *shape), dtype=dtype, device=self.device)

    def extend(self, scheme, groups=None):
        self._setup_data(scheme, self.groups if groups is None else groups, self.batch_size, self.max_seq_length)

    def to(self, device):
        for k, v in self.data.transition_data.items():
            self.data.transition_data[k] = v.to(device)
        for k, v in self.data.episode_data.items():
            self.data.episode_data[k] = v.to(device)
        self.device = device

    def update(self, data, bs=slice(None), ts=slice(None), mark_filled=True):
        slices = self._parse_slices((bs, ts))
        for k, v in data.items():
            if k in self.data.transition_data:
                target = self.data.transition_data
                if mark_filled and not self.efficient_store:
                    target["filled"][slices] = 1
                    mark_filled = False
                _slices = slices
            elif k in self.data.episode_data:
                target = self.data.episode_data
                _slices = slices[0]
            else:
                raise KeyError("{} not found in transition or episode data".format(k))

            dtype = self.scheme[k].get("dtype", th.float32)
            if self.efficient_store and k in self.data.transition_data:
                _bs, _ts = _slices
                assert (((_ts.start, _ts.stop, _ts.step) == (0, self.max_seq_length, 1) or
                         _ts == slice(None)),
                        "Efficient store only supports replacing entire trajectories")

                trunc_v = [th.tensor(v[i, :int(data["filled"][i].sum())],
                                     dtype=dtype,
                                     device=self.device)
                           for i in range(v.shape[0])]
                target[k][_bs] = trunc_v

                if k in self.preprocess:
                    new_k = self.preprocess[k][0]
                    v = target[k][_bs]  # returns a padded tensor
                    for transform in self.preprocess[k][1]:
                        v = transform.transform(v)
                    trunc_v = [th.tensor(v[i, :int(data["filled"][i].sum())],
                                         dtype=dtype,
                                         device=self.device)
                               for i in range(v.shape[0])]
                    target[new_k][_bs] = trunc_v
            else:
                v = th.tensor(v, dtype=dtype, device=self.device)
                self._check_safe_view(v, target[k][_slices])
                target[k][_slices] = v.view_as(target[k][_slices])

                if k in self.preprocess:
                    new_k = self.preprocess[k][0]
                    v = target[k][_slices]
                    for transform in self.preprocess[k][1]:
                        v = transform.transform(v)
                    target[new_k][_slices] = v.view_as(target[new_k][_slices])

    def _check_safe_view(self, v, dest):
        idx = len(v.shape) - 1
        for s in dest.shape[::-1]:
            if v.shape[idx] != s:
                if s != 1:
                    raise ValueError("Unsafe reshape of {} to {}".format(v.shape, dest.shape))
            else:
                idx -= 1

    def __contains__(self, key):
        if key in self.data.episode_data or key in self.data.transition_data:
            return True
        return False

    def __getitem__(self, item):
        # Return data that matches the key (e.g. 'states')
        if isinstance(item, str):
            if item in self.data.episode_data:
                return self.data.episode_data[item]
            elif item in self.data.transition_data:
                return self.data.transition_data[item][:]
            else:
                raise ValueError
        # Return all data that matches the keys passed in a tuple
        elif isinstance(item, tuple) and all([isinstance(it, str) for it in item]):
            new_data = self._new_data_sn()
            for key in item:
                if key in self.data.transition_data:
                    new_data.transition_data[key] = self.data.transition_data[key][:]
                elif key in self.data.episode_data:
                    new_data.episode_data[key] = self.data.episode_data[key]
                else:
                    raise KeyError("Unrecognised key {}".format(key))

            # Update the scheme to only have the requested keys
            new_scheme = {key: self.scheme[key] for key in item}
            new_groups = {self.scheme[key]["group"]: self.groups[self.scheme[key]["group"]]
                          for key in item if "group" in self.scheme[key]}
            ret = EpisodeBatch(new_scheme, new_groups, self.batch_size, self.max_seq_length, data=new_data, device=self.device)
            return ret
        # Return data from all keys but cut time to slices
        else:
            item = self._parse_slices(item)

            if self.efficient_store:
                _bs, _ts = item
                assert (((_ts.start, _ts.stop, _ts.step) == (0, self.max_seq_length, 1) or
                         _ts == slice(None)),
                        "Efficient store only supports getting entire trajectories")
                tr_item = _bs
            else:
                tr_item = item

            new_data = self._new_data_sn()
            for k, v in self.data.transition_data.items():
                new_data.transition_data[k] = v[tr_item]
            for k, v in self.data.episode_data.items():
                new_data.episode_data[k] = v[item[0]]

            ret_bs = self._get_num_items(item[0], self.batch_size)
            ret_max_t = self._get_num_items(item[1], self.max_seq_length)

            ret = EpisodeBatch(self.scheme, self.groups, ret_bs, ret_max_t, data=new_data, device=self.device)
            return ret

    def _get_num_items(self, indexing_item, max_size):
        if (isinstance(indexing_item, list)
                or isinstance(indexing_item, np.ndarray)
                or isinstance(indexing_item, th.LongTensor)
                or isinstance(indexing_item, th.cuda.LongTensor)):
            return len(indexing_item)
        elif isinstance(indexing_item, slice):
            _range = indexing_item.indices(max_size)
            return 1 + (_range[1] - _range[0] - 1)//_range[2]

    def _new_data_sn(self):
        new_data = SN()
        new_data.transition_data = {}
        new_data.episode_data = {}
        return new_data

    def _parse_slices(self, items):
        parsed = []
        # Only batch slice given, add full time slice
        if (isinstance(items, slice)  # slice a:b
            or isinstance(items, int)  # int i
            or (isinstance(items, (list, np.ndarray, th.LongTensor, th.cuda.LongTensor)))  # [a,b,c]
            ):
            items = (items, slice(None))

        # Need the time indexing to be contiguous
        if isinstance(items[1], list):
            raise IndexError("Indexing across Time must be contiguous")

        for item in items:
            #TODO: stronger checks to ensure only supported options get through
            if isinstance(item, int):
                # Convert single indices to slices
                parsed.append(slice(item, item+1))
            else:
                # Leave slices and lists as is
                parsed.append(item)
        return parsed

    def max_t_filled(self):
        return th.sum(self.data.transition_data["filled"][:], 1).max(0)[0]

    def __repr__(self):
        return "EpisodeBatch. Batch Size:{} Max_seq_len:{} Keys:{} Groups:{}".format(self.batch_size,
                                                                                     self.max_seq_length,
                                                                                     self.scheme.keys(),
                                                                                     self.groups.keys())

class ReplayBuffer(EpisodeBatch):
    def __init__(self, scheme, groups, buffer_size, max_seq_length, preprocess=None, device="cpu", efficient_store=False, max_traj_len=-1):
        assert (max_traj_len <= max_seq_length), "Sampled trajectories must be shorter than max episode length"
        super(ReplayBuffer, self).__init__(scheme, groups, buffer_size, max_seq_length, preprocess=preprocess, device=device, efficient_store=efficient_store)
        self.buffer_size = buffer_size  # same as self.batch_size but more explicit
        self.buffer_index = 0
        self.episodes_in_buffer = 0
        self.max_traj_len = max_traj_len

    def insert_episode_batch(self, ep_batch):
        if self.buffer_index + ep_batch.batch_size <= self.buffer_size:
            self.update(ep_batch.data.transition_data,
                        slice(self.buffer_index, self.buffer_index + ep_batch.batch_size),
                        slice(0, ep_batch.max_seq_length),
                        mark_filled=False)
            self.update(ep_batch.data.episode_data,
                        slice(self.buffer_index, self.buffer_index + ep_batch.batch_size))
            self.buffer_index = (self.buffer_index + ep_batch.batch_size)
            self.episodes_in_buffer = max(self.episodes_in_buffer, self.buffer_index)
            self.buffer_index = self.buffer_index % self.buffer_size
            assert self.buffer_index < self.buffer_size
        else:
            buffer_left = self.buffer_size - self.buffer_index
            self.insert_episode_batch(ep_batch[0:buffer_left, :])
            self.insert_episode_batch(ep_batch[buffer_left:, :])

    def can_sample(self, batch_size, filters={}):
        if len(filters) == 0:
            return self.episodes_in_buffer >= batch_size
        filtered_ep_mask = th.zeros(self.buffer_size, dtype=th.uint8, device=self.device)
        filtered_ep_mask[:self.episodes_in_buffer] = 1
        for item_name, condition in filters.items():
            filtered_ep_mask *= condition(self.data.episode_data[item_name].flatten())
        return filtered_ep_mask.sum().cpu().item() >= batch_size

    def _stratified_randint(self, start, end, num_samps):
        """
        Sample integers unformly while ensuring you sample evenly
        """
        bin_inds = np.linspace(start, end, num_samps + 1).round().astype(int)
        bin_starts = bin_inds[:-1]
        bin_ends = bin_inds[1:]
        return np.random.randint(bin_starts, bin_ends)

    def sample(self, batch_size, filters={}):
        # assert self.can_sample(batch_size, filters=filters)
        filtered_ep_mask = th.zeros(self.buffer_size, dtype=th.uint8, device=self.device)
        filtered_ep_mask[:self.episodes_in_buffer] = 1
        for item_name, condition in filters.items():
            filtered_ep_mask *= condition(self.data.episode_data[item_name].flatten())
        valid_inds = th.arange(self.buffer_size, device=self.device)[filtered_ep_mask]
        if len(valid_inds) == batch_size:
            batch = self[valid_inds]
        else:
            # Uniform sampling only atm
            ep_ids = np.random.choice(valid_inds.cpu().numpy(), batch_size, replace=False)
            batch = self[ep_ids]
        if self.max_traj_len != -1:
            all_lens = batch["filled"].sum(dim=1).flatten()
            # don't need to truncate if all trajs already shorter than max
            if th.any(all_lens > self.max_traj_len):
                # current method selects random contiguous trajectories from all concatenated episodes in batch
                flat_filled = batch["filled"].flatten()
                # have to use np.where bc pytorch 1.1.0 doesn't support this usage
                data_inds = th.tensor(np.where(flat_filled.cpu().numpy())[0],
                                      device=self.device)
                samp_traj_starts = th.tensor(
                    self._stratified_randint(
                        0, len(data_inds) - self.max_traj_len + 1, batch_size),
                    device=self.device
                )
                samp_traj_ends = samp_traj_starts + self.max_traj_len
                traj_inds = th.cat([data_inds[s:e] for s, e in zip(samp_traj_starts, samp_traj_ends)], dim=0)
                for k, v in batch.data.transition_data.items():
                    bs, ts = v.shape[:2]
                    v_shape = v.shape[2:]
                    batch.data.transition_data[k] = v.reshape(bs * ts, *v_shape)[traj_inds].reshape(bs, self.max_traj_len, *v_shape)
                # Alternative method that doesn't overlap trajectories (will have issues w/ sampling ts near start and end less frequently)
                # traj_ends = all_lens.clamp_min(self.max_traj_len).cpu().numpy()
                # traj_start_max = traj_ends - self.max_traj_len
                # samp_traj_starts = np.random.randint(0, traj_start_max + 1)
                # samp_traj_ends = samp_traj_starts + self.max_traj_len
                # traj_bounds = th.tensor(list(list(range(s, e)) for s, e in zip(samp_traj_starts, samp_traj_ends)), device=self.device)
                # for k, v in batch.data.transition_data.items():
                #     n_extra_dims = len(v.shape) - 2
                #     curr_traj_bounds = traj_bounds.reshape(*traj_bounds.shape, *[1 for _ in range(n_extra_dims)]).repeat(1, 1, *v.shape[2:])
                #     batch.data.transition_data[k] = v.gather(1, curr_traj_bounds)

                batch.max_seq_length = self.max_traj_len
        return batch

    def __repr__(self):
        return "ReplayBuffer. {}/{} episodes. Keys:{} Groups:{}".format(self.episodes_in_buffer,
                                                                        self.buffer_size,
                                                                        self.scheme.keys(),
                                                                        self.groups.keys())

    # def _check_slice(self, slice, max_size):
    #     if slice.step is not None:
    #         return slice.step > 0  # pytorch doesn't support negative steps so neither do we
    #     if slice.start is None and slice.stop is None:
    #         return True
    #     elif slice.start is None:
    #         return 0 < slice.stop <= max_size
    #     elif slice.stop is None:
    #         return 0 <= slice.start < max_size
    #     else:
    #         return (0 < slice.stop <= max_size) and (0 <= slice.start < max_size)

if __name__ == "__main__":
    bs = 4
    n_agents = 2
    groups = {"agents": n_agents}

    # "input": {"vshape": (shape), "episode_const": bool, "group": (name), "dtype": dtype}
    scheme = {
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "obs": {"vshape": (3,), "group": "agents"},
        "state": {"vshape": (3,3)},
        "epsilon": {"vshape": (1,), "episode_const": True}
    }
    from transforms import OneHot
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=5)])
    }

    ep_batch = EpisodeBatch(scheme, groups, bs, 3, preprocess=preprocess)

    env_data = {
        "actions": th.ones(n_agents, 1).long(),
        "obs": th.ones(2, 3),
        "state": th.eye(3)
    }
    batch_data = {
        "actions": th.ones(bs, n_agents, 1).long(),
        "obs": th.ones(2, 3).unsqueeze(0).repeat(bs,1,1),
        "state": th.eye(3).unsqueeze(0).repeat(bs,1,1),
    }
    # bs=4 x t=3 x v=3*3

    ep_batch.update(env_data, 0, 0)

    ep_batch.update({"epsilon": th.ones(bs)*.05})

    ep_batch[:, 1].update(batch_data)
    ep_batch.update(batch_data, ts=1)

    ep_batch.update(env_data, 0, 1)

    env_data = {
        "obs": th.ones(2, 3),
        "state": th.eye(3)*2
    }
    ep_batch.update(env_data, 3, 0)

    b2 = ep_batch[0, 1]
    b2.update(env_data, 0, 0)

    replay_buffer = ReplayBuffer(scheme, groups, 5, 3, preprocess=preprocess)

    replay_buffer.insert_episode_batch(ep_batch)

    replay_buffer.insert_episode_batch(ep_batch)

    sampled = replay_buffer.sample(3)

    print(sampled["actions_onehot"])