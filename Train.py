from Environment import parallel_env
from Algorithm import train, eval


env_fn = parallel_env
env_kwargs = {}
train(env_fn, steps=10000000, seed=0, **env_kwargs)
