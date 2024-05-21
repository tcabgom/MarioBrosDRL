from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gymnasium.wrappers import GrayScaleObservation, ResizeObservation
from nes_py.wrappers import JoypadSpace
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.atari_wrappers import AtariWrapper


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


def atari_wrapper(env):
    # https://stable-baselines3.readthedocs.io/en/master/common/atari_wrappers.html

    # Noop reset: obtain initial state by taking random number of no-ops on reset
    # Frame skipping: 4 by default
    # Max-pooling: most recent two observations
    # Termination signal when a life is lost.
    # Resize to a square image: 84x84 by default
    # Grayscale observation
    # Clip reward to {-1, 0, 1}
    # Sticky actions: disabled by default

    return AtariWrapper(env)
