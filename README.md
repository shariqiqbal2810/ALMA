# ALMA
Code for [*ALMA: Hierarchical Learning for Composite Multi-Agent Tasks*](https://openreview.net/forum?id=JUXn1vXcrLA) (Iqbal et al., NeurIPS 2022)

This code is built on the [public code release for REFIL](https://github.com/shariqiqbal2810/REFIL) which is built on the [PyMARL framework](https://github.com/oxwhirl/pymarl)

## Citing our work

If you use this repo in your work, please consider citing the corresponding paper:

```bibtex
@inproceedings{iqbal2022alma,
title={ALMA: Hierarchical Learning for Composite Multi-Agent Tasks},
author={Shariq Iqbal and Robby Costales and Fei Sha},
booktitle={Advances in Neural Information Processing Systems},
year={2022},
url={https://openreview.net/forum?id=JUXn1vXcrLA}
}
```

## Installation instructions

1. Install Docker
2. Install NVIDIA Docker if you want to use GPU (recommended)
3. Build the docker image using 
```bash
cd docker
./build.sh
```
4. Set up StarCraft II. If installed already on your machine just make sure `SC2PATH` is set correctly, otherwise run:
```bash
./install_sc2.sh
```
5. Make sure `SC2PATH` is set to the installation directory (`3rdparty/StarCraftII`)
6. Make sure `WANDB_API_KEY` is set if you want to use weights and biases

## Running experiments

Use the following command to run:
```bash
./run.sh <GPU> python3.7 src/main.py \
    --config=<alg> --env-config=<env> --scenario=<scen>
```
with the bracketed parameters replaced as follows:
* `<GPU>`: The index of the GPU you would like to run this experiment on
* `<alg>`: The low-level learning algorithm (choices are `qmix_atten` or `refil`)
* `<env>`: The environment
  * `ff`: SaveTheCity environment
  * `sc2multiarmy`: StarCraft environment
* `<scen>`: Specifies set of tasks in the environment (for StarCraft)
  * `6-8sz_maxsize4_maxarmies3_symmetric`: Stalkers and Zealots Symmetric
  * `6-8sz_maxsize4_maxarmies3_unitdisadvantage`: Stalkers and Zealots Disadvantage
  * `6-8MMM_maxsize4_maxarmies3_symmetric`: MMM Symmetric
  * `6-8MMM_maxsize4_maxarmies3_unitdisadvantage`: MMM Disadvantage

Method-Specific parameters:
* ALMA: Use `--agent.subtask_cond='mask'` and `--hier_agent.task_allocation='aql'`
* ALMA (No Mask):  `--agent.subtask_cond='full_obs'` and `--hier_agent.task_allocation='aql'`
* Heuristic Allocation: Use `--agent.subtask_cond='mask'` and `--hier_agent.task_allocation='heuristic'`
  * StarCraft (Dynamic):  `--env_args.heuristic_style='attacking-type-unassigned-diff'`
  * StarCraft (Matching): `--env_args.heuristic_style='type-unassigned-diff'`
* COPA: `--hier_agent.copa=True`

Environment-Specific hyperparameters:
* `SaveTheCity`
  * Use `--epsilon_anneal_time=2000000` for all methods
  * Use `--hier_agent.action_length=5` for hierarchical methods (allocation-based and COPA)
  * Use `--config=qmix_atten`
* `StarCraft`
  * Use `--hier_agent.action_length=3` for hierarchical methods (allocation-based and COPA)
  * Use `--config=refil`

Miscellaneous parameters:
* Weights and Biases: To use, make a project named "task-allocation" in weights and biases and include the following parameters in your runs. Make sure `WANDB_API_KEY` is set.
  * `--use-wandb=True`: Enables W&B logging,
  * `--wb-notes`: Notes associated with this experiment,
  * `--wb-tags` Specify list of tags separated by spaces
  * `--wb-entity` Specify W&B user or group name