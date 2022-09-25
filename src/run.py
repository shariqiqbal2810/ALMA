import datetime
from functools import partial
from math import ceil
import imageio
import wandb
import os
import pprint
import time
import json
import threading
import uuid
import torch as th
from numpy.random import RandomState
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath, basename, join, splitext

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from envs import s_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot


def run(config, console_logger, wandb_run):
    # check args sanity
    config = args_sanity_check(config, console_logger)

    args = SN(**config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
    logger = Logger(console_logger)

    if not (args.evaluate or args.save_replay) and args.use_wandb:
        logger.setup_wandb()

    console_logger.info("Experiment Parameters:")
    experiment_params = pprint.pformat(config,
                                       indent=4,
                                       width=1)
    console_logger.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f"))
    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(dirname(dirname(abspath(__file__))), "results", args.tb_dirname)
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    # Run and train
    run_sequential(args=args, logger=logger)

    if wandb_run is not None:
        wandb_run.finish()

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)


def load_run(args, wb_run, learner, runner, logger, pi_only=False):
    timesteps = []
    timestep_to_load = 0

    files = wb_run.files()
    timesteps = set(int(f.name.split('_')[0]) for f in files if f.name.endswith('.th'))

    if args.load_step == 0:
        # choose the max timestep
        timestep_to_load = max(timesteps)
    else:
        # choose the timestep closest to load_step
        timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

    files_to_load = [f for f in files if f.name.startswith(str(timestep_to_load))]
    if pi_only:
        files_to_load = [f for f in files_to_load if 'agent.th' in f.name]
    unique_id = uuid.uuid4().hex[:6].lower()
    model_dir = os.path.join(f'./eval_models', unique_id)
    os.makedirs(model_dir)
    for file in files_to_load:
        wandb.restore(file.name, run_path='/'.join(wb_run.path),
                      replace=True, root=model_dir)
    model_path = os.path.join(model_dir, f"{timestep_to_load}_")
    logger.console_logger.info(f"Loading model from run: {wb_run.name}")
    learner.load_models(model_path, evaluate=args.evaluate, pi_only=pi_only)
    runner.t_env = timestep_to_load

def get_wandb_runs(checkpoint_name, wandb_api, args):
    if 'sc2' in args.env:
        metric = 'eval_env_metrics/battle_won_mean'
    elif args.env == 'ff':
        metric = 'eval_env_metrics/solved_mean'
    else:
        raise Exception('Need to define best model metric for this environment')
    exp_runs = wandb_api.runs(
        path=f'{args.wb_entity}/task-allocation',
        filters={'config.name': checkpoint_name},
        order=f'-summary_metrics.{metric}')
    assert len(exp_runs) > 0, "No matching runs found"
    return exp_runs

def evaluate_sequential(args, runner, logger):
    vw = None
    if args.video_path is not None:
        os.makedirs(dirname(args.video_path), exist_ok=True)
        vid_basename_split = splitext(basename(args.video_path))
        if vid_basename_split[1] == '.mp4':
            vid_basename = ''.join(vid_basename_split)
        else:
            vid_basename = ''.join(vid_basename_split) + '.mp4'
        vid_filename = join(dirname(args.video_path), vid_basename)
        vw = imageio.get_writer(vid_filename, format='FFMPEG', mode='I',
                                fps=args.fps, codec='h264', quality=10)

    res_dict = {}

    if args.eval_all_scen:
        if 'sc2' in args.env:
            dict_key = 'scenarios'
        else:
            raise Exception("Environment (%s) does not incorporate multiple scenarios")
        n_runs = len(args.env_args['scenario_dict'][dict_key])
    elif args.eval_n_task_range != "":
        min_n_tasks, max_n_tasks = args.eval_n_task_range.split("-")
        n_task_range = list(range(int(min_n_tasks), int(max_n_tasks) + 1))
        n_runs = len(n_task_range)
    else:
        n_runs = 1
    n_test_batches = max(1, args.test_nepisode // runner.batch_size)

    all_subtask_infos = []
    for i in range(n_runs):
        logger.console_logger.info(f"Running evaluation on setting {i + 1}/{n_runs}")
        run_args = {'test_mode': True, 'vid_writer': vw,
                    'test_scen': True}
        if args.eval_all_scen:
            run_args['index'] = i
        elif args.eval_n_task_range != "":
            run_args['n_tasks'] = n_task_range[i]
        for _ in range(n_test_batches):
            _, new_subtask_infos = runner.run(**run_args)
            all_subtask_infos.extend(new_subtask_infos)
        curr_stats = dict((k, v[-1][1]) for k, v in logger.stats.items())
        if args.eval_all_scen:
            curr_scen = args.env_args['scenario_dict'][dict_key][i]
            # assumes that unique set of agents is a unique scenario
            if 'sc2' in args.env:
                scen_str = "-".join("%i%s" % (count, name[:3]) for count, name in sorted(curr_scen[0], key=lambda x: x[1]))
            else:
                scen_str = "".join(curr_scen[0])
            res_dict[scen_str] = curr_stats
        elif args.eval_n_task_range != "":
            res_dict[n_task_range[i]] = curr_stats
        else:
            res_dict.update(curr_stats)

    if vw is not None:
        vw.close()

    if args.save_replay:
        runner.save_replay()
    return res_dict, all_subtask_infos


def run_sequential(args, logger):
    # Init runner so we can get env info
    if 'entity_scheme' in args.env_args:
        args.entity_scheme = args.env_args['entity_scheme']
    else:
        args.entity_scheme = False

    if args.env in ['sc2custom', 'sc2multiarmy', 'ff']:
        rs = RandomState(0)
        args.env_args['scenario_dict'] = s_REGISTRY[args.scenario](rs=rs)

    if args.hier_agent["task_allocation"] == 'heuristic':
        args.env_args['heuristic_alloc'] = True

    if ('sc2custom' == args.env or 'sc2multiarmy' == args.env):
        args.env_args['n_extra_tags'] = args.n_extra_units

    if args.mixer_subtask_cond is None:
        args.mixer_subtask_cond = args.agent["subtask_cond"]
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    if not args.entity_scheme:
        args.state_shape = env_info["state_shape"]
        # Default/Base scheme
        scheme = {
            "state": {"vshape": env_info["state_shape"]},
            "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
            "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
            "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
            "reward": {"vshape": (1,)},
            "terminated": {"vshape": (1,), "dtype": th.uint8},
            "reset": {"vshape": (1,), "dtype": th.uint8},
            "ep_num": {"vshape": (1,), "dtype": th.long, "episode_const": True}
        }
        groups = {
            "agents": args.n_agents
        }
        args.pre_transition_items = ['state', 'obs', 'avail_actions']
        args.post_transition_items = ['reward', 'terminated', 'reset']
        if 'masks' in env_info:
            # masks that identify what part of observation/state spaces correspond to each entity
            args.obs_masks, args.state_masks = env_info['masks']
        if 'unit_dim' in env_info:
            args.unit_dim = env_info['unit_dim']
    else:
        args.entity_shape = env_info["entity_shape"]
        args.n_entities = env_info["n_entities"]
        # Entity scheme
        scheme = {
            "entities": {"vshape": env_info["entity_shape"], "group": "entities"},
            "obs_mask": {"vshape": env_info["n_entities"], "group": "entities", "dtype": th.uint8},
            "entity_mask": {"vshape": env_info["n_entities"], "dtype": th.uint8},
            "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
            "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
            "reward": {"vshape": (1,)},
            "terminated": {"vshape": (1,), "dtype": th.uint8},
            "reset": {"vshape": (1,), "dtype": th.uint8},
            "t_added": {"vshape": (1,), "dtype": th.long, "episode_const": True}
        }
        args.pre_transition_items = ['entities', 'obs_mask', 'entity_mask', 'avail_actions']
        args.post_transition_items = ['reward', 'terminated', 'reset']
        groups = {
            "agents": args.n_agents,
            "entities": args.n_entities
        }
        if args.multi_task:
            args.n_tasks = env_info["n_tasks"]
            scheme["entity2task_mask"] = {"vshape": args.n_tasks, "group": "entities", "dtype": th.uint8}
            scheme["task_mask"] = {"vshape": args.n_tasks, "dtype": th.uint8}
            args.pre_transition_items += ["entity2task_mask", "task_mask"]
            args.post_transition_items += ["task_rewards", "tasks_terminated"] #, "oracle_attribution"]
            scheme["task_rewards"] = {"vshape": args.n_tasks}
            scheme["tasks_terminated"] = {"vshape": args.n_tasks, "dtype": th.uint8}
            # scheme["oracle_attribution"] = {"vshape": args.n_tasks, "group": "agents", "dtype": th.uint8}
            if args.hier_agent["task_allocation"] is not None or args.hier_agent["copa"]:
                scheme["hier_decision"] = {"vshape": (1,), "dtype": th.uint8}
                args.pre_transition_items += ['hier_decision']

    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }

    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device,
                          efficient_store=args.buffer_opt_mem,
                          max_traj_len=args.max_traj_len)

    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if args.use_cuda:
        learner.cuda()

    if args.checkpoint_run_name != "" or args.checkpoint_unique_id != "":
        assert not (args.checkpoint_run_name != ""
                    and args.checkpoint_unique_id != ""), (
                        "Can only specify one of checkpoint_run_name or checkpoint_unique_id"
                    )
        wandb_api = wandb.Api()
        pi_runs = None
        if args.checkpoint_run_name != "":
            exp_runs = get_wandb_runs(args.checkpoint_run_name, wandb_api, args)
            if args.pi_checkpoint_run_name != "":
                if args.pi_checkpoint_run_name == "same":
                    assert len(exp_runs) > 1, "Need multiple seeds to swap policies"
                    pi_runs = list(exp_runs)[1:] + list(exp_runs)[:1]
                else:
                    pi_runs = get_wandb_runs(args.pi_checkpoint_run_name, wandb_api, args)
            if args.eval_all_models:
                runs = exp_runs
            else:
                # take model with the best final performance
                runs = [exp_runs[0]]
        elif args.checkpoint_unique_id != "":
            runs = [wandb_api.run(f'{args.wb_entity}/task-allocation/{args.checkpoint_unique_id}')]

    if args.checkpoint_run_name != "" or args.checkpoint_unique_id != "" or args.env_args.get('heuristic_ai', False):
        if args.evaluate or args.save_replay:
            if args.eval_path is not None:
                os.makedirs(dirname(args.eval_path), exist_ok=True)
                eval_basename_split = splitext(basename(args.eval_path))
                if eval_basename_split[1] == '.json':
                    eval_basename = ''.join(eval_basename_split)
                else:
                    eval_basename = ''.join(eval_basename_split) + '.json'
                eval_filename = join(dirname(args.eval_path), eval_basename)

            if args.checkpoint_run_name != "" or args.checkpoint_unique_id != "":
                results = []
                for i, wb_run in enumerate(runs):
                    logger.console_logger.info(f"Evaluating model {i + 1}/{len(runs)}")
                    load_run(args, wb_run, learner, runner, logger)
                    if pi_runs is not None:
                        pi_run = pi_runs[i % len(pi_runs)]
                        load_run(args, pi_run, learner, runner, logger, pi_only=True)
                    res_dict, all_subtask_infos = evaluate_sequential(args, runner, logger)
                    results.append(res_dict)

                if args.eval_path is not None:
                    if args.eval_sep:
                        write_struct = all_subtask_infos
                    else:
                        write_struct = results
                    with open(eval_filename, 'w') as f:
                        json.dump(write_struct, f)
            else:
                # heuristic
                res_dict, all_subtask_infos = evaluate_sequential(args, runner, logger)
            runner.close_env()
            logger.print_stats_summary()
            return

    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    while runner.t_env <= args.t_max:

        # Run for a whole episode at a time
        episode_batch, _ = runner.run(test_mode=False)
        buffer.insert_episode_batch(episode_batch)

        if buffer.can_sample(args.batch_size):
            for _ in range(args.training_iters):
                episode_sample = buffer.sample(args.batch_size)

                # Truncate batch to only filled timesteps
                max_ep_t = episode_sample.max_t_filled()
                episode_sample = episode_sample[:, :max_ep_t]

                if episode_sample.device != args.device:
                    episode_sample.to(args.device)

                learner.train(episode_sample, runner.t_env, episode)

                if args.hier_agent["task_allocation"] in ["aql"]:
                    filters = {}
                    if args.hier_agent['decay_old'] > 0:
                        cutoff = args.hier_agent['decay_old']
                        filters['t_added'] = lambda t_added: (runner.t_env - t_added) <= cutoff
                    if not buffer.can_sample(args.batch_size, filters=filters):
                        continue

                    alloc_episode_sample = buffer.sample(args.batch_size, filters=filters)

                    # Truncate batch to only filled timesteps
                    max_ep_t = alloc_episode_sample.max_t_filled()
                    alloc_episode_sample = alloc_episode_sample[:, :max_ep_t]

                    if alloc_episode_sample.device != args.device:
                        alloc_episode_sample.to(args.device)

                    elif args.hier_agent["task_allocation"] == "aql":
                        learner.alloc_train_aql(alloc_episode_sample, runner.t_env, episode)

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

            logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
            logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
            last_time = time.time()

            last_test_T = runner.t_env
            for _ in range(n_test_runs):
                runner.run(test_mode=True)

        if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or
                                model_save_time == 0 or
                                runner.t_env > args.t_max):
            model_save_time = runner.t_env
            if args.use_wandb:
                save_path_base = os.path.join(wandb.run.dir, "%i_" % (runner.t_env))
            else:
                save_path_base = os.path.join("results/models", args.unique_token, str(runner.t_env), "")
            os.makedirs(os.path.dirname(save_path_base), exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(os.path.dirname(save_path_base)))
            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner.save_models(save_path_base)

        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    runner.close_env()
    logger.console_logger.info("Finished Training")


# TODO: Clean this up
def args_sanity_check(config, console_logger):
    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        console_logger.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"]//config["batch_size_run"]) * config["batch_size_run"]

    if (config["hier_agent"]["task_allocation"] is not None
            or config["agent"]["subtask_cond"] is not None):
        assert (config["agent"]["subtask_cond"] is not None
                and config["hier_agent"]["task_allocation"] is not None), (
            "Subtask-conditioning type and task allocation must be specified together")

    # assert (config["run_mode"] in ["parallel_subproc"] and config["use_replay_buffer"]) or (not config["run_mode"] in ["parallel_subproc"]),  \
    #     "need to use replay buffer if running in parallel mode!"

    # assert not (not config["use_replay_buffer"] and (config["batch_size_run"]!=config["batch_size"]) ) , "if not using replay buffer, require batch_size and batch_size_run to be the same."

    # if config["learner"] == "coma":
    #    assert (config["run_mode"] in ["parallel_subproc"]  and config["batch_size_run"]==config["batch_size"]) or \
    #    (not config["run_mode"] in ["parallel_subproc"]  and not config["use_replay_buffer"]), \
    #        "cannot use replay buffer for coma, unless in parallel mode, when it needs to have exactly have size batch_size."

    return config
