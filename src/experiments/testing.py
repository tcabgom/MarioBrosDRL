import gym_super_mario_bros
import gymnasium
from stable_baselines3.common.monitor import Monitor

from src.experiments import environment_preprocessing, experiment_utils


def test_super_mario_bros(model_path):
    env = gym_super_mario_bros.make('SuperMarioBros-v0', render_mode="human")
    experiment_utils.print_environment_data(env)
    env = Monitor(env, experiment_utils.LOG_DIR)
    env = environment_preprocessing.reduce_action_space(env)
    env = environment_preprocessing.reduce_observation_space(env)
    env = environment_preprocessing.enhance_observation_space(env)
    experiment_utils.print_environment_data(env)
    experiment_utils.load_and_test_model(env, model_path)


def test_space_invaders(model_path):
    env = gymnasium.make("ALE/SpaceInvaders-v5", render_mode='human')
    experiment_utils.print_environment_data(env)
    env = Monitor(env, experiment_utils.LOG_DIR)
    env = environment_preprocessing.reduce_observation_space(env)
    env = environment_preprocessing.enhance_observation_space(env)
    experiment_utils.print_environment_data(env)
    experiment_utils.load_and_test_model(env, model_path)
