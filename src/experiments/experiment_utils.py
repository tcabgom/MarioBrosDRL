import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import A2C, DQN, PPO

from src.models import A2C_model, PPO_model, DQN_model

CHECKPOINT_DIR = "../../train/"
LOG_DIR = "../../logs/"


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


def print_environment_data(env):
    print("\n################################################################")
    print("Número de acciones: " + str(env.action_space))
    print("Espacio observable: " + str(env.observation_space))
    print("################################################################\n")


def test_random_actions_tutorial(env, print_observation):
    """
    Renders the given environment performing random actions.

    Args:
        env: The environment that will be used

    Returns:
        None
    """
    terminated = False
    truncated = False
    observation = env.reset()
    for step in range(5000):
        if terminated or truncated:
            observation = env.reset()
            print("resetting environment")
        observation, reward, done, info = env.step([env.action_space.sample()])
        if print_observation:
            plt.imshow(np.squeeze(observation))
        plt.show()
        env.render()

    env.close()
    print("Environment closed")


def create_model(env, algorithm):
    if algorithm == "DQN":
        model = DQN_model.create_DQN_model(env)
    elif algorithm == "PPO":
        model = PPO_model.create_PPO_model(env)
    elif algorithm == "A2C":
        model = A2C_model.create_A2C_model(env)
    else:
        print("Invalid Algorithm")
        model = None
    return model


def train_agent(model, check_freq, save_freq_best, total_timesteps):
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
    callback = TrainAndLoggingCallback(check_freq=check_freq,
                                       save_path=CHECKPOINT_DIR,
                                       save_freq_best=save_freq_best,
                                       model_name="train")
    model.learn(total_timesteps=total_timesteps, callback=callback)

    return model


def load_and_test_model(env, model_path, algorithm):
    """
    Loads a trained model of the environment to test its performance
    """
    terminated = True
    truncated = False
    if algorithm == "DQN":
        model = DQN.load(model_path, env=env)
    elif algorithm == "PPO":
        model = PPO.load(model_path, env=env)
    elif algorithm == "A2C":
        model = A2C.load(model_path, env=env)
    else:
        print("Invalid Algorithm")
        return
    vec_env = model.get_env()
    observation = vec_env.reset()
    for step in range(15000):
        action, _state = model.predict(observation)
        print(action)
        observation, reward, done, info = vec_env.step(action)
        env.render()

    env.close()
