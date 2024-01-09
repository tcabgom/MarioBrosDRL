from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
import gymnasium
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from stable_baselines3 import PPO
import os
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium.wrappers import GrayScaleObservation, ResizeObservation, FrameStack, FlattenObservation, AtariPreprocessing


CHECKPOINT_DIR = "./train/"
LOG_DIR = "./logs/"


def test_random_actions_tutorial():

    env = gym_super_mario_bros.make('SuperMarioBros-v3', render_mode="human")

    env = reduce_action_space(env)
    env = reduce_observation_space(env)
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

    env = reduce_action_space(env)
    env = reduce_observation_space(env)
    env = enhance_observation_space(env)
    #env = AtariPreprocessing(env,frame_skip=1, screen_size=84, grayscale_obs=True, scale_obs=False)

    print_environment_data(env)

    callback = TrainAndLoggingCallback(check_freq=100000, save_path=CHECKPOINT_DIR)

    model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=0.000001, n_steps=512)
    model.learn(total_timesteps=4000000, callback=callback)

    return model


class FrameSkipWrapper(gymnasium.Wrapper):
    def __init__(self, env, skip):
        super(FrameSkipWrapper, self).__init__(env)
        self.skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for i in range(self.skip):
            observation, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            done = done or terminated  # Update the 'done' flag
            if done:
                break
        return observation, total_reward, done, truncated, info




def load_and_test_model():

    env = gym_super_mario_bros.make('SuperMarioBros-v3', render_mode="human")
    env = reduce_action_space(env)
    env = reduce_observation_space(env)

    terminated = True
    truncated = False
    model = PPO.load('./train/best_model_5400000', env=env)
    vec_env = model.get_env()
    observation = vec_env.reset()
    for step in range(5000):
        action, _state = model.predict(observation)
        observation, reward, done, info = vec_env.step(action)
        env.render()

    env.close()


def print_environment_data(env):
    print("Número de acciones: " + str(env.action_space))
    print("Espacio observable: " + str(env.observation_space))

def reduce_action_space(env):
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    return env

def reduce_observation_space(env):
    # https://gymnasium.farama.org/api/wrappers/
    env = GrayScaleObservation(env, keep_dim=True) # Convertimos la imagen en blanco y negro
    env = ResizeObservation(env, shape=(60, 64))   # Reducimos el tamaño de la imagen al 25%

    #env = FrameSkipWrapper(env,4)                 # Hace que vaya más lento, probablemente está mal
    #env = FlattenObservation(env)                 # Parece que no funciona, tampoco estoy seguro de si era útil

    return env

def enhance_observation_space(env):
    env = FrameStack(env, num_stack=3)
    return env

if __name__ == '__main__':
    load_and_test_model()
    #test_random_actions_tutorial()
    #m = train_agent()
    print('model trained')
