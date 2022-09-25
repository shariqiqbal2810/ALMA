from .agent import Agent
from .allocation_critics import *
from .allocation_policies import *
from functools import partial

ALLOC_CRITIC_REGISTRY = {}
ALLOC_CRITIC_REGISTRY['standard'] = StandardAllocCritic

ALLOC_POLICY_REGISTRY = {}
ALLOC_POLICY_REGISTRY['autoreg'] = AutoregressiveAllocPolicy
