from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np


class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)

        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info(self.args)

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self, **kwargs):
        self.batch = self.new_batch()
        self.env.reset(**kwargs)
        self.t = 0

    def _get_pre_transition_data(self, env_info, terminal=False):
        if self.args.entity_scheme:
            masks = self.env.get_masks()
            pre_transition_data = {
                "entities": [self.env.get_entities()],
                "avail_actions": [self.env.get_avail_actions()],
            }
            for name, mask in masks.items():
                pre_transition_data[name] = [mask]
            if self.args.hier_agent['task_allocation'] is not None or self.args.hier_agent["copa"]:
                if self.t == 0:
                    pre_transition_data['hier_decision'] = [(1,)]
                else:
                    tasks_changed = env_info.pop("tasks_changed")
                    if (self.hier_timer >= self.args.hier_agent['action_length'] or tasks_changed) and not terminal:
                        self.hier_timer = 0
                        pre_transition_data['hier_decision'] = [(1,)]
                    else:
                        pre_transition_data['hier_decision'] = [(0,)]
        else:
            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }
        return pre_transition_data

    def run(self, test_mode=False, test_scen=None, index=None, n_tasks=None, vid_writer=None):
        """
        test_mode: whether to use greedy action selection or sample actions
        test_scen: whether to run on test scenarios. defaults to matching test_mode.
        vid_writer: imageio video writer object
        """
        if test_scen is None:
            test_scen = test_mode
        self.reset(test=test_scen, index=index, n_tasks=n_tasks)
        terminated = False
        episode_return = 0
        self.hier_timer = 0
        env_info = {}
        self.mac.init_hidden(batch_size=self.batch_size)
        # make sure things like dropout are disabled
        self.mac.eval()
        final_subtask_infos = []

        while not terminated:
            pre_transition_data = self._get_pre_transition_data(env_info)

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
            if vid_writer is not None:
                allocs = None
                if hasattr(self.mac, 'task_allocations') and self.mac.task_allocations is not None:
                    allocs = self.mac.task_allocations.argmax(dim=-1).squeeze().cpu().numpy()
                vid_writer.append_data(self.env.render(allocs=allocs))
            reward, terminated, env_info = self.env.step(actions[0].cpu())
            episode_return += reward

            scenario = None
            if "scenario" in env_info:
                scenario = env_info.pop("scenario")
                armies_defeated = env_info.pop("armies_defeated")
                infiltrated_base = env_info.pop("armies_infiltrated")
                spread = env_info.pop("spread")
            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
                "reset": [(terminated,)],
            }
            for key in self.args.post_transition_items:
                if key in env_info:
                    post_transition_data[key] = [(env_info[key],)]
                    env_info.pop(key)

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1
            self.hier_timer += 1

        if scenario is not None:
            final_subtask_infos.append(
                {"scenario": scenario, "armies_defeated": armies_defeated,
                 "infiltrated_base": infiltrated_base,
                 "spread": spread})

        if vid_writer is not None:
            allocs = None
            if hasattr(self.mac, 'task_allocations') and self.mac.task_allocations is not None:
                allocs = self.mac.task_allocations.argmax(dim=-1).squeeze().cpu().numpy()
            vid_writer.append_data(self.env.render(allocs=allocs))

        last_data = self._get_pre_transition_data(env_info, terminal=True)
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.batch.update({"actions": actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif not test_mode and self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch, final_subtask_infos

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()
