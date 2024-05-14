import gym_super_mario_bros
import gymnasium
from stable_baselines3.common.monitor import Monitor

import environment_preprocessing
import experiment_utils


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
    env = gymnasium.make("SpaceInvadersNoFrameskip-v4", render_mode='human')
    #env = gymnasium.make("SpaceInvaders-ramNoFrameskip-v4", render_mode='human')
    experiment_utils.print_environment_data(env)
    env = Monitor(env, experiment_utils.LOG_DIR)
    env = environment_preprocessing.reduce_observation_space(env)
    #env = environment_preprocessing.atari_wrapper(env)
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
    env = gymnasium.make("SpaceInvadersNoFrameskip-v4", render_mode='human')
    experiment_utils.print_environment_data(env)
    env = Monitor(env, experiment_utils.LOG_DIR)
    #env = environment_preprocessing.reduce_observation_space(env)
    env = environment_preprocessing.atari_wrapper(env)
    #env = environment_preprocessing.enhance_observation_space(env)
    experiment_utils.print_environment_data(env)
    experiment_utils.test_random_actions_tutorial(env, print_observation)
    env.close()


if __name__ == '__main__':
    #test_random_actions_super_mario_bros(False)
    #test_random_actions_space_invaders(True)
    #test_super_mario_bros("../results/SuperMarioBros/DQN/01_BestModelWithManualExperiments/ppo_12_BEST_3350000_2222-27_1358-38_45956-72", "PPO")
    #test_space_invaders("../train/best_model_3000000", "DQN")
    test_space_invaders(
        "../results/SpaceInvaders/DQN/01_BestModelWithManualExperiments/DQN_02/best_model_7800000",
        "DQN")
    #test_space_invaders("../results/SpaceInvaders/A2C/RAM/01_BestModelWithManualExperiments/train_PERIODIC_6000000_270-00_2151-00_5153-64", "A2C")