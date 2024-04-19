import gym_super_mario_bros
import gymnasium
import optuna_training
from stable_baselines3.common.monitor import Monitor

import environment_preprocessing
import experiment_utils



def train_super_mario_bros(check_freq, save_freq_best, total_timesteps, algorithm):
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    experiment_utils.print_environment_data(env)
    env = Monitor(env, experiment_utils.LOG_DIR)
    env = environment_preprocessing.reduce_action_space(env)
    env = environment_preprocessing.reduce_observation_space(env)
    env = environment_preprocessing.enhance_observation_space(env)
    experiment_utils.print_environment_data(env)
    model = experiment_utils.create_model(env, algorithm)
    experiment_utils.train_agent(model, check_freq, save_freq_best, total_timesteps)
    env.close()
    print('Model trained')


def train_space_invaders(check_freq, save_freq_best, total_timesteps, algorithm):
    env = gymnasium.make("SpaceInvadersNoFrameskip-v4", render_mode='rgb_array')
    experiment_utils.print_environment_data(env)
    env = Monitor(env, experiment_utils.LOG_DIR)
    #env = environment_preprocessing.reduce_observation_space(env)
    env = environment_preprocessing.atari_wrapper(env)
    env = environment_preprocessing.enhance_observation_space(env)
    experiment_utils.print_environment_data(env)
    model = experiment_utils.create_model(env, algorithm)
    experiment_utils.train_agent(model, check_freq, save_freq_best, total_timesteps)
    env.close()
    print('Model trained')


if __name__ == '__main__':
    train_space_invaders(2000000, 1000000, 14000000, "PPO")
    #optuna_training.search_hyperparameters_optuna(500000, 500000, 1000000, 15, "A2C")
