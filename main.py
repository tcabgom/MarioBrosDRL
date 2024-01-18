from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
import gymnasium
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from stable_baselines3 import PPO, DQN
import os
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium.wrappers import GrayScaleObservation, ResizeObservation
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack


CHECKPOINT_DIR = "./train/"
LOG_DIR = "./logs/"


def test_random_actions_tutorial(env):
    """
    Renders the given environment performing random actions.

    Args:
        env: The environment that will be used

    Returns:
        None
    """
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


def train_agent(env, model, check_freq, total_timesteps):
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
    print_environment_data(env)

    #env = reduce_action_space(env)
    env = reduce_observation_space(env)
    env = enhance_observation_space(env)

    print_environment_data(env)

    callback = TrainAndLoggingCallback(check_freq=check_freq, save_path=CHECKPOINT_DIR)
    model.learn(total_timesteps=total_timesteps, callback=callback)

    return model


def load_and_test_model(env, model_path):
    """
    Loads a trained model of the environment to test its performance
    """
    print_environment_data(env)
    #env = reduce_action_space(env)
    env = reduce_observation_space(env)
    env = enhance_observation_space(env)
    print_environment_data(env)

    terminated = True
    truncated = False
    model = PPO.load(model_path, env=env)
    print("############################"+str(model.observation_space))
    vec_env = model.get_env()
    observation = vec_env.reset()
    for step in range(5000):
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
    env = ResizeObservation(env, shape=(60, 64))   # Reduce the image size

    return env


def enhance_observation_space(env):
    env = DummyVecEnv([lambda: env])     #
    env = VecFrameStack(env, n_stack=4)  # Stacking frames to let the model recognize speed
    return env


if __name__ == '__main__':
    #env = gymnasium.make("ALE/SpaceInvaders-v5", render_mode='rgb_array')
    #env = gymnasium.make('ALE/SpaceInvaders-v5', render_mode="human")
    env = gym_super_mario_bros.make('SuperMarioBros-v0', render_mode="human")
    model = DQN("CnnPolicy", env,
                verbose=1,                    # Controls the verbosity level (0: no output, 1: training information)
                tensorboard_log=LOG_DIR,      # Directory for storing Tensorboard logs
                learning_rate=0.00001,        # The learning rate for the optimizer
                buffer_size=50000,            # Size of the replay buffer
                learning_starts=1000,         # Number of steps before starting to update the model
                train_freq=4,                 # Number of steps between updates of the model
                gradient_steps=1,             # Number of gradient steps to take per update
                target_update_interval=10000, # Update target network every `target_update_interval` steps
                exploration_fraction=0.05,    # Fraction of total timesteps during which exploration rate is decreased
                exploration_final_eps=0.01,   # Final value of the exploration rate
                max_grad_norm=10,             # Clipping of gradients during optimization
                gamma=0.99                    # Discount factor for future rewards
                )
    #test_random_actions_tutorial(env)
    load_and_test_model(env, "./train/best_model_1000")
    #m = train_agent(env, model, 500, 1000)
    print('Model trained')
