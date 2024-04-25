import gym_super_mario_bros
import gymnasium
from stable_baselines3.common.vec_env import VecFrameStack, VecNormalize, SubprocVecEnv

import optuna_training
from stable_baselines3.common.monitor import Monitor

import environment_preprocessing
import experiment_utils
from src import A2C_model


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

def train_space_invaders_stacking_environments(check_freq, save_freq_best, total_timesteps, algorithm):
    def make_env():
        e = gymnasium.make("SpaceInvadersNoFrameskip-v4", render_mode='rgb_array')
        e = environment_preprocessing.atari_wrapper(e)
        e = environment_preprocessing.enhance_observation_space(e)
        return e

    env_list = [make_env() for _ in range(8)]
    env = SubprocVecEnv(env_list)
    model = experiment_utils.create_model(env, algorithm)
    experiment_utils.print_environment_data(env)
    experiment_utils.train_agent(model, check_freq, save_freq_best, total_timesteps)
    env.close()
    print('Model trained')

def train_space_invaders_ram(check_freq, save_freq_best, total_timesteps):
    env = gymnasium.make("SpaceInvaders-ramNoFrameskip-v4", render_mode='rgb_array')
    experiment_utils.print_environment_data(env)
    model = A2C_model.create_A2C_mlp_model(env)
    experiment_utils.train_agent(model, check_freq, save_freq_best, total_timesteps)
    env.close()
    print('Model trained')


if __name__ == '__main__':
    train_space_invaders_ram(10000, 10000, 10000)
    #train_space_invaders_stacking_environments(10000, 10000, 10000, "PPO")
    #train_space_invaders(2000000, 1000000, 14000000, "DQN")
    #optuna_training.search_hyperparameters_optuna(500000, 500000, 1000000, 15, "A2C")
