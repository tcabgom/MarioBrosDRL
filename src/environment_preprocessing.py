from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gymnasium.wrappers import GrayScaleObservation, ResizeObservation
from nes_py.wrappers import JoypadSpace
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack


def reduce_action_space(env):
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    return env


def reduce_observation_space(env):
    # https://gymnasium.farama.org/api/wrappers/
    env = GrayScaleObservation(env, keep_dim=True) # Convert the image to black and white
    env = ResizeObservation(env, shape=(84, 84))   # Reduce the image size

    return env


def enhance_observation_space(env):
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=4)  # Stacking frames to let the model recognize speed
    return env
