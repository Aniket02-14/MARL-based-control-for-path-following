from Environment import parallel_env
from Algorithm import eval


env_fn = parallel_env
env_kwargs = {}

eval(env_fn, num_games=10, render_mode=None, **env_kwargs)