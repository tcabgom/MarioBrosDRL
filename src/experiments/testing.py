import gym_super_mario_bros
import gymnasium
from stable_baselines3.common.monitor import Monitor

from src.experiments import environment_preprocessing, experiment_utils


def test_super_mario_bros(model_path, algorithm):
    env = gym_super_mario_bros.make('SuperMarioBros-v0', render_mode="human")
    experiment_utils.print_environment_data(env)
    env = Monitor(env, experiment_utils.LOG_DIR)
    env = environment_preprocessing.reduce_action_space(env)
    env = environment_preprocessing.reduce_observation_space(env)
    env = environment_preprocessing.enhance_observation_space(env)
    experiment_utils.print_environment_data(env)
    experiment_utils.load_and_test_model(env, model_path, algorithm)
    print("Test completed")


def test_space_invaders(model_path, algorithm):
    env = gymnasium.make("ALE/SpaceInvaders-v5", render_mode='human')
    experiment_utils.print_environment_data(env)
    env = Monitor(env, experiment_utils.LOG_DIR)
    env = environment_preprocessing.reduce_observation_space(env)
    env = environment_preprocessing.enhance_observation_space(env)
    experiment_utils.print_environment_data(env)
    experiment_utils.load_and_test_model(env, model_path, algorithm)
    print("Test completed")


def test_random_actions_super_mario_bros(print_observation):
    env = gym_super_mario_bros.make('SuperMarioBros-v0', render_mode="human")
    experiment_utils.print_environment_data(env)
    env = Monitor(env, experiment_utils.LOG_DIR)
    env = environment_preprocessing.reduce_action_space(env)
    env = environment_preprocessing.reduce_observation_space(env)
    env = environment_preprocessing.enhance_observation_space(env)
    experiment_utils.print_environment_data(env)
    experiment_utils.test_random_actions_tutorial(env, print_observation)
    env.close()


def test_random_actions_space_invaders(print_observation):
    env = gymnasium.make("ALE/SpaceInvaders-v5", render_mode='human')
    experiment_utils.print_environment_data(env)
    env = Monitor(env, experiment_utils.LOG_DIR)
    env = environment_preprocessing.reduce_observation_space(env)
    env = environment_preprocessing.enhance_observation_space(env)
    experiment_utils.print_environment_data(env)
    experiment_utils.test_random_actions_tutorial(env, print_observation)
    env.close()


if __name__ == '__main__':
    #test_random_actions_super_mario_bros()
    #test_random_actions_space_invaders(False)
    test_space_invaders("../../train/train_PERIODIC_1000000_195-05_579-27_8085-42.zip", "A2C")
