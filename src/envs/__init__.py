from functools import partial

from .multiagentenv import MultiAgentEnv
from .starcraft2 import StarCraft2Env, StarCraft2CustomEnv, StarCraft2MultiArmyEnv
from .firefighters import FireFightersEnv

from .starcraft2 import custom_scenario_registry as sc_scenarios
from .firefighters import scenarios as ff_scenarios


# TODO: Do we need this?
def env_fn(env, **kwargs) -> MultiAgentEnv: # TODO: this may be a more complex function
    # env_args = kwargs.get("env_args", {})
    return env(**kwargs)


REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
REGISTRY["sc2custom"] = partial(env_fn, env=StarCraft2CustomEnv)
REGISTRY["sc2multiarmy"] = partial(env_fn, env=StarCraft2MultiArmyEnv)
REGISTRY["ff"] = partial(env_fn, env=FireFightersEnv)

s_REGISTRY = {}
s_REGISTRY.update(sc_scenarios)
s_REGISTRY.update(ff_scenarios)
