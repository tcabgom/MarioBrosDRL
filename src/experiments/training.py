import gym_super_mario_bros
import gymnasium
from stable_baselines3.common.monitor import Monitor

from src.experiments import experiment_utils, environment_preprocessing
from src.models import DQN_model, A2C_model, PPO_model


def train_super_mario_bros(check_freq, total_timesteps, algorithm):
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    experiment_utils.print_environment_data(env)
    env = Monitor(env, experiment_utils.LOG_DIR)
    env = environment_preprocessing.reduce_action_space(env)
    env = environment_preprocessing.reduce_observation_space(env)
    env = environment_preprocessing.enhance_observation_space(env)
    experiment_utils.print_environment_data(env)
    model = experiment_utils.create_model(env, algorithm)
    experiment_utils.train_agent(model, check_freq, total_timesteps)
    env.close()
    print('Model trained')


def train_space_invaders(check_freq, total_timesteps, algorithm):
    env = gymnasium.make("ALE/SpaceInvaders-v5", render_mode='rgb_array')
    experiment_utils.print_environment_data(env)
    env = Monitor(env, experiment_utils.LOG_DIR)
    env = environment_preprocessing.reduce_observation_space(env)
    env = environment_preprocessing.enhance_observation_space(env)
    experiment_utils.print_environment_data(env)
    model = experiment_utils.create_model(env, algorithm)
    experiment_utils.train_agent(model, check_freq, total_timesteps)
    env.close()
    print('Model trained')

if __name__ == '__main__':
    train_space_invaders(25000, 100000, 'A2C')