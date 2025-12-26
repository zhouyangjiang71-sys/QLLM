from functools import partial
import sys
import os

from .multiagentenv import MultiAgentEnv



gfootball = True
from .gfootball import GoogleFootballEnv


def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["gfootball"] = partial(env_fn, env=GoogleFootballEnv)

