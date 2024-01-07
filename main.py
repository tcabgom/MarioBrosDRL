from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from stable_baselines3 import PPO
import os
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium.wrappers import GrayScaleObservation, ResizeObservation
#from stable_baselines3.common.envs import VecFrameStack
#from stable_baselines3.common.envs import DummyVecEnv


CHECKPOINT_DIR = "./train/"
LOG_DIR = "./logs/"


def test_random_actions_tutorial():

    env = gym_super_mario_bros.make('SuperMarioBros-v3', render_mode="human")
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    print_environment_data(env)

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
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, "best_model_{}".format(self.n_calls))
            self.model.save(model_path)
        return True


def train_agent():
    env = gym_super_mario_bros.make('SuperMarioBros-v0', render_mode=None)
    print_environment_data(env)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = GrayScaleObservation(env, keep_dim=True) # Convertimos la imagen en blanco y negro
    env = ResizeObservation(env, shape=(60, 64))   # Reducimos el tamaño de la imagen al 25%
    #env = DummyVecEnv([lambda: env])
    #env = VecFrameStack(env, n_stack=2)
    print_environment_data(env)
    callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)
    model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=0.000001, n_steps=512)
    model.learn(total_timesteps=4000000, callback=callback)
    return model


def print_environment_data(env):
    print("Número de acciones: " + str(env.action_space))
    print("Espacio observable: " + str(env.observation_space))


if __name__ == '__main__':
    test_random_actions_tutorial()
    #m = train_agent()
    print('model trained')
