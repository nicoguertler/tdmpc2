from copy import deepcopy
import warnings

# import gym
import gymnasium
from gymnasium.wrappers import FlattenObservation
import numpy as np

# import fetch_construction
from trifinger_mujoco_env import MoveCubeEnv

from envs.wrappers.multitask import MultitaskWrapper
from envs.wrappers.pixels import PixelWrapper
from envs.wrappers.tensor import TensorWrapper

def missing_dependencies(task):
	raise ValueError(f'Missing dependencies for task {task}; install dependencies to use this environment.')

try:
	from envs.dmcontrol import make_env as make_dm_control_env
except:
	make_dm_control_env = missing_dependencies
try:
	from envs.maniskill import make_env as make_maniskill_env
except:
	make_maniskill_env = missing_dependencies
try:
	from envs.metaworld import make_env as make_metaworld_env
except:
	make_metaworld_env = missing_dependencies
try:
	from envs.myosuite import make_env as make_myosuite_env
except:
	make_myosuite_env = missing_dependencies


warnings.filterwarnings('ignore', category=DeprecationWarning)


# Wrapper for using gymnasium environments
class GymWrapper(gymnasium.Wrapper):
    def __init__(self, env, log_transform_rew=False, max_env_steps=None):
        super().__init__(env)
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.log_transform_rew = log_transform_rew
        if max_env_steps is None:
            self.max_episode_steps = env.spec.max_episode_steps
        else:
            self.max_episode_steps = max_env_steps

    def reset(self, **kwargs):
        obs, _ = self.env.reset(**kwargs)
        return obs

    def render(self, mode='human', **kwargs):
        return self.env.render()

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        if self.log_transform_rew:
            reward = np.log(reward)
        return observation, reward, done, info


def make_gymnasium_env(cfg):
	"""
	Make a gymnasium environment for TD-MPC2 experiments.
	"""
	if cfg.task == "ant-maze-medium":
		env = gymnasium.make("AntMaze_Medium_Diverse_GRDense-v4", terminate_when_unhealthy=False, render_mode='rgb_array')
		env = FlattenObservation(env)
		# in original environment, negative distance is put through an exponential function
		env = GymWrapper(env, log_transform_rew=True)
	elif cfg.task == "fetch":
		env = gymnasium.make("FetchPickAndPlaceTable1Dense-v1", render_mode='rgb_array')
		env = FlattenObservation(env)
		env = GymWrapper(env)
	elif cfg.task == "trifinger-mujoco-lift":
		env = MoveCubeEnv(
			task="lift",
			start_viewer=True,
			# To avoid error about EGL context
			create_camera_renderer=False,
		)
		env = FlattenObservation(env)
		env = GymWrapper(env, max_env_steps=env.unwrapped.episode_length)
	return env


def make_multitask_env(cfg):
	"""
	Make a multi-task environment for TD-MPC2 experiments.
	"""
	print('Creating multi-task environment with tasks:', cfg.tasks)
	envs = []
	for task in cfg.tasks:
		_cfg = deepcopy(cfg)
		_cfg.task = task
		_cfg.multitask = False
		env = make_env(_cfg)
		if env is None:
			raise ValueError('Unknown task:', task)
		envs.append(env)
	env = MultitaskWrapper(cfg, envs)
	cfg.obs_shapes = env._obs_dims
	cfg.action_dims = env._action_dims
	cfg.episode_lengths = env._episode_lengths
	return env
	

def make_env(cfg):
	"""
	Make an environment for TD-MPC2 experiments.
	"""
	# gym.logger.set_level(40)
	if cfg.multitask:
		env = make_multitask_env(cfg)

	else:
		env = None
		# for fn in [make_gymnasium_env, make_dm_control_env, make_maniskill_env, make_metaworld_env, make_myosuite_env]:
		# 	try:
		# 		env = fn(cfg)
		# 	except ValueError:
		# 		pass
		env = make_gymnasium_env(cfg)
		if env is None:
			raise ValueError(f'Failed to make environment "{cfg.task}": please verify that dependencies are installed and that the task exists.')
		env = TensorWrapper(env)
	if cfg.get('obs', 'state') == 'rgb':
		env = PixelWrapper(cfg, env)
	try: # Dict
		cfg.obs_shape = {k: v.shape for k, v in env.observation_space.spaces.items()}
	except: # Box
		cfg.obs_shape = {cfg.get('obs', 'state'): env.observation_space.shape}
	cfg.action_dim = env.action_space.shape[0]
	cfg.episode_length = env.max_episode_steps
	cfg.seed_steps = max(1000, 5*cfg.episode_length)
	return env
