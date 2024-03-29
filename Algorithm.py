from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from pettingzoo.utils import parallel_to_aec
import supersuit as ss
import os
import glob

def train(env_fn, steps: int = 10_000, seed: int | None = 0, **env_kwargs):
    # Train a single model to play as each agent in an AEC environment
    #env = env_fn.parallel_env(**env_kwargs)
    env = env_fn(**env_kwargs)

    # Add black death wrapper so the number of agents stays constant
    # MarkovVectorEnv does not support environments with varying numbers of active agents unless black_death is set to True
    #env = ss.black_death_v3(env)
    env.reset(seed=seed)

    print(f"Starting training on {str(env.metadata['name'])}.")

    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 8, num_cpus=1, base_class="stable_baselines3")

    # Use a CNN policy if the observation space is visual
    model = PPO(
        MlpPolicy,
        env,
        verbose=3,
        batch_size=256,
    )

    model.learn(total_timesteps=steps,log_interval=10)

    model.save(f"{env.unwrapped.metadata.get('name')}")

    print("Model has been saved.")

    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.")

    env.close()

def eval(env_fn, num_games: int = 100, render_mode: str | None = None, **env_kwargs):
    # Evaluate a trained agent vs a random agent

    #env = env_fn.env(render_mode=render_mode, **env_kwargs)
    env = env_fn(render_mode=render_mode, **env_kwargs)
    env = parallel_to_aec(env)


    print(
        f"\nStarting evaluation on {str(env.metadata['name'])} (num_games={num_games}, render_mode={render_mode})"
    )

    try:
        latest_policy = max(
            glob.glob(f"{env.metadata['name']}*.zip"), key=os.path.getctime
        )
    except ValueError:
        print("Policy not found.")
        exit(0)

    model = PPO.load(latest_policy)

    rewards = {agent: 0 for agent in env.possible_agents}

    # Note: we evaluate here using an AEC environments, to allow for easy A/B testing against random policies
    # For example, we can see here that using a random agent for archer_0 results in less points than the trained agent
    for i in range(num_games):
        env.reset(seed=i)
        env.action_space(env.possible_agents[0]).seed(i)

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()

            for agent in env.agents:
                rewards[agent] += env.rewards[agent]

            if termination or truncation:
                env.render()
                break
            else:
                if agent == env.possible_agents[0]:
                    act = env.action_space(agent).sample()
                else:
                    act = model.predict(obs, deterministic=True)[0]
            env.step(act)
    env.close()

    avg_reward = sum(rewards.values()) / len(rewards.values())
    avg_reward_per_agent = {
        agent: rewards[agent] / num_games for agent in env.possible_agents
    }
    print(f"Avg reward: {avg_reward}")
    print("Avg reward per agent, per game: ", avg_reward_per_agent)
    print("Full rewards: ", rewards)
    return avg_reward