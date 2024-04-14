import gc
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
import gymnasium
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from stable_baselines3 import DQN, A2C
import os
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium.wrappers import GrayScaleObservation, ResizeObservation
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
import pandas as pd
import numpy as np
from . import optuna_training
import torch
import json


CHECKPOINT_DIR = "../../train/"
LOG_DIR = "../../logs/"


def test_random_actions_tutorial(env):
    """
    Renders the given environment performing random actions.

    Args:
        env: The environment that will be used

    Returns:
        None
    """
    terminated = True
    truncated = False
    for step in range(5000):
        if terminated or truncated:
            observation = env.reset()
            print("resetting environment")
        observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
        env.render()

    env.close()


class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, save_path, model_name, check_freq=5000, save_freq_best=100000, save_freq_force=200000,
                 verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_freq_best = save_freq_best
        self.save_freq_force = save_freq_force
        self.save_path = save_path
        self.model_name = model_name
        self.current_best_info_mean = {'r': -np.inf, 'l': 0, 't': 0}
        self.current_best_model = None
        self.current_best_n_calls = None
        self.current_best_changed = False

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            if self.model.ep_info_buffer:
                df = pd.DataFrame(self.model.ep_info_buffer)
                info_mean = df.mean()
                if info_mean['r'] > self.current_best_info_mean['r']:
                    model_path = os.path.join(self.save_path, "best_model_tmp")
                    self.model.save(model_path)
                    self.current_best_info_mean = info_mean
                    self.current_best_changed = True
                    self.current_best_n_calls = self.n_calls
                    # print('new best model {}'.format(self.current_best_info_mean))

        if self.n_calls % self.save_freq_best == 0 and self.current_best_changed:
            model_path = os.path.join(self.save_path, '{}_BEST_{}_{:.2f}_{:.2f}_{:.2f}'
                                      .format(self.model_name, self.n_calls, self.current_best_info_mean['r'],
                                              self.current_best_info_mean['l'], self.current_best_info_mean['t'])
                                      .replace('.', '-'))
            print('Saving new BEST model to {}'.format(model_path))
            self.model.save(model_path)
            self.current_best_changed = False

        if self.n_calls % self.save_freq_force == 0:
            if self.model.ep_info_buffer:
                df = pd.DataFrame(self.model.ep_info_buffer)
                info_mean = df.mean()
                model_path = os.path.join(self.save_path, '{}_PERIODIC_{}_{:.2f}_{:.2f}_{:.2f}'
                                          .format(self.model_name, self.n_calls, info_mean['r'],
                                                  info_mean['l'], info_mean['t'])
                                          .replace('.', '-'))
                print('Saving PERIODIC model to {}'.format(model_path))
                self.model.save(model_path)
        return True



def train_agent(model, check_freq, total_timesteps):
    """
    Trains the given environment with the given model

    Args:
        env: The environment that will be used to train
        model: The model that will be trained
        check_freq: The number of iterations that will run before saving a copy of the model
        total_timesteps: The number of iterations of the training

    Returns:
        model: The model after all iterations of the training
    """
    callback = TrainAndLoggingCallback(check_freq=check_freq, save_path=CHECKPOINT_DIR, model_name="train")
    model.learn(total_timesteps=total_timesteps, callback=callback)

    return model


def load_and_test_model(env, model_path):
    """
    Loads a trained model of the environment to test its performance
    """
    terminated = True
    truncated = False
    model = DQN.load(model_path, env=env)
    vec_env = model.get_env()
    observation = vec_env.reset()
    for step in range(15000):
        action, _state = model.predict(observation)
        observation, reward, done, info = vec_env.step(action)
        env.render()

    env.close()


def print_environment_data(env):
    print("\n################################################################")
    print("Número de acciones: " + str(env.action_space))
    print("Espacio observable: " + str(env.observation_space))
    print("################################################################\n")

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


def train_super_mario_bros(check_freq, total_timesteps):
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    print_environment_data(env)
    env = Monitor(env, LOG_DIR)
    env = reduce_action_space(env)
    env = reduce_observation_space(env)
    env = enhance_observation_space(env)
    print_environment_data(env)
    model = create_DQN_model(env)
    train_agent(model, check_freq, total_timesteps)
    print('Model trained')


def train_super_mario_bros2(check_freq, total_timesteps):
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    #env = CustomRewardWrapper(env)
    print_environment_data(env)
    env = Monitor(env, LOG_DIR)
    env = reduce_action_space(env)
    env = reduce_observation_space(env)
    env = enhance_observation_space(env)
    print_environment_data(env)
    model = create_custom_DQN_model()
    train_agent(model, check_freq, total_timesteps)
    print('Model trained')


def test_super_mario_bros(model_path):
    env = gym_super_mario_bros.make('SuperMarioBros-v0', render_mode="human")
    print_environment_data(env)
    env = Monitor(env, LOG_DIR)
    env = reduce_action_space(env)
    env = reduce_observation_space(env)
    env = enhance_observation_space(env)
    print_environment_data(env)
    load_and_test_model(env, model_path)


def train_space_invaders(check_freq, total_timesteps):
    env = gymnasium.make("ALE/SpaceInvaders-v5", render_mode='rgb_array')
    print_environment_data(env)
    env = Monitor(env, LOG_DIR)
    env = reduce_observation_space(env)
    env = enhance_observation_space(env)
    print_environment_data(env)
    model = create_A2C_model(env)
    #model = create_DQN_model(env)
    train_agent(model, check_freq, total_timesteps)
    print('Model trained')


def test_space_invaders(model_path):
    env = gymnasium.make("ALE/SpaceInvaders-v5", render_mode='human')
    print_environment_data(env)
    env = Monitor(env, LOG_DIR)
    env = reduce_observation_space(env)
    env = enhance_observation_space(env)
    print_environment_data(env)
    load_and_test_model(env, model_path)


def objective_aux(trial):
    # https://optuna.org/

    env = gymnasium.make("ALE/SpaceInvaders-v5", render_mode='rgb_array')
    print_environment_data(env)
    env = Monitor(env, LOG_DIR)
    env = reduce_observation_space(env)
    env = enhance_observation_space(env)
    print_environment_data(env)
    exploration_final_eps = trial.suggest_float('exploration_final_eps', 0.005, 0.01)
    learning_rate = trial.suggest_float('learning_rate', 0.0001, 0.01)
    train_frequency = trial.suggest_int('train_frequency', 2, 6)
    buffer_size = trial.suggest_int('buffer_size', 600000, 1000000)
    # gamma = trial.suggest_float('gamma', 0.95, 0.95)

    hyperparams = {
        'exploration_final_eps': exploration_final_eps,
        'learning_rate': learning_rate,
        'train_frequency': train_frequency,
        'buffer_size': buffer_size,
        'gamma': 0.95                                       # Parametro optimo
    }
    with open(f"{CHECKPOINT_DIR}/hyperparams_trial_{trial.number}.json", 'w') as f:
        json.dump(hyperparams, f)

    model = create_custom_DQN_model(env, exploration_final_eps, learning_rate, train_frequency, buffer_size, 0.95)
    print(exploration_final_eps, learning_rate, train_frequency, buffer_size, 0.95)
    callback = TrainAndLoggingCallback(save_path=CHECKPOINT_DIR, model_name='dqn_optuna_3', check_freq=1000,
                                       save_freq_best=750000, save_freq_force=750000)
    model.learn(total_timesteps=1500000, callback=callback)

    ret = float(callback.current_best_info_mean['r'])
    torch.cuda.empty_cache()
    del callback
    del env
    del model
    return ret


def objective(trial):
    ret = objective_aux(trial)
    gc.collect()
    return ret


def create_A2C_model(env):
    # https://stable-baselines.readthedocs.io/en/master/modules/a2c.html
    return A2C("CnnPolicy", env,
               verbose=1,                               # Controls the verbosity level (0: no output, 1: training information)
               tensorboard_log=LOG_DIR,                 # Directory for storing Tensorboard logs
               learning_rate=0.0005,                    # The learning rate for the optimizer
               gamma=0.99,                              # Discount factor for future rewards
               ent_coef=0.01,                           # Entropy coefficient for the loss calculation
               vf_coef=0.5,                             # Value function coefficient for the loss calculation
               max_grad_norm=0.5,                       # Clipping of gradients during optimization
               gae_lambda=0.95,                         # Lambda for the Generalized Advantage Estimator
               n_steps=5,                               # Number of steps to run for each environment per update
               policy_kwargs=dict(net_arch=[64, 64])    # Additional network architecture
               # model_name= "A2C",        # Model name
               )


def create_DQN_model(env):
    return DQN("CnnPolicy", env,
                verbose=1,                     # Controls the verbosity level (0: no output, 1: training information)
                tensorboard_log=LOG_DIR,       # Directory for storing Tensorboard logs
                learning_rate = 0.01,          # The learning rate for the optimizer
                buffer_size=100000,            # Size of the replay buffer
                learning_starts=20000,         # Number of steps before starting to update the model
                train_freq=2,                  # Number of steps between updates of the model
                gradient_steps=1,              # Number of gradient steps to take per update
                target_update_interval=10000,  # Update target network every `target_update_interval` steps
                exploration_fraction=0.05,     # Fraction of total timesteps during which exploration rate is decreased
                exploration_final_eps=0.01,    # Final value of the exploration rate
                max_grad_norm=10,              # Clipping of gradients during optimization
                gamma=0.99                     # Discount factor for future rewards
                #device = "cuda:0"
                )

def create_custom_DQN_model(env,
                     exploration_final_eps,
                     learning_rate,
                     train_frequency,
                     buffer_size,
                     gamma):
    return DQN("CnnPolicy", env,
                verbose=1,                                      # Controls the verbosity level (0: no output, 1: training information)
                tensorboard_log=LOG_DIR,                        # Directory for storing Tensorboard logs
                learning_rate=learning_rate,                    # The learning rate for the optimizer
                buffer_size=buffer_size,                        # Size of the replay buffer
                learning_starts=20000,                          # Number of steps before starting to update the model
                train_freq=train_frequency,                     # Number of steps between updates of the model
                gradient_steps=1,                               # Number of gradient steps to take per update
                target_update_interval=10000,                   # Update target network every `target_update_interval` steps
                exploration_fraction=0.05,                      # Fraction of total timesteps during which exploration rate is decreased
                exploration_final_eps=exploration_final_eps,    # Final value of the exploration rate
                max_grad_norm=10,                               # Clipping of gradients during optimization
                gamma=gamma                                     # Discount factor for future rewards
                #device = "cuda:0"
                )


def search_hyperparameters_optuna():
    study = optuna_training.create_study(direction='maximize')
    results = []
    study.optimize(objective, n_trials=50)
    print('### TRIALS COMPLETE ###')
    trial = study.best_trial
    print('Best Value: ', trial.value)
    print('Best Params: ')
    for key, value in trial.params.items():
        print(f'    {key}: {value}')


if __name__ == '__main__':
    #train_super_mario_bros2(200000,8000000)
    #train_super_mario_bros(200000,8000000)
    #train_space_invaders(200000, 1000000)
    test_space_invaders("../../train/train_BEST_200000_251-75_710-67_1622-01.zip")
    #search_hyperparameters_optuna()
