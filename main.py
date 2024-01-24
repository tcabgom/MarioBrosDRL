from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
import gymnasium
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from stable_baselines3 import PPO, DQN
import os
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium.wrappers import GrayScaleObservation, ResizeObservation
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor


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
    callback = TrainAndLoggingCallback(check_freq=check_freq, save_path=CHECKPOINT_DIR)
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
    env = ResizeObservation(env, shape=(84, 84))   # Reduce the image size

    return env


def enhance_observation_space(env):
    env = DummyVecEnv([lambda: env])     #
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
    train_agent(env, model, check_freq, total_timesteps)
    print('Model trained')


def test_super_mario_bros(model_path):
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
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
    model = create_DQN_model(env)
    train_agent(env, model, check_freq, total_timesteps)
    print('Model trained')


def test_space_invaders(model_path):
    env = gymnasium.make("ALE/SpaceInvaders-v5", render_mode='human')
    print_environment_data(env)
    env = Monitor(env, LOG_DIR)
    env = reduce_observation_space(env)
    env = enhance_observation_space(env)
    print_environment_data(env)
    load_and_test_model(env, model_path)


def create_DQN_model(env):
    return  DQN("CnnPolicy", env,
                verbose=1,                    # Controls the verbosity level (0: no output, 1: training information)
                tensorboard_log=LOG_DIR,      # Directory for storing Tensorboard logs
                learning_rate=0.0001,         # The learning rate for the optimizer
                buffer_size=300000,           # Size of the replay buffer
                learning_starts=20000,        # Number of steps before starting to update the model
                train_freq=2,                 # Number of steps between updates of the model
                gradient_steps=1,             # Number of gradient steps to take per update
                target_update_interval=10000, # Update target network every `target_update_interval` steps
                exploration_fraction=0.05,    # Fraction of total timesteps during which exploration rate is decreased
                exploration_final_eps=0.01,   # Final value of the exploration rate
                max_grad_norm=10,             # Clipping of gradients during optimization
                gamma=0.999                  # Discount factor for future rewards
                #device = "cuda:0"
                )


if __name__ == '__main__':
    train_space_invaders(200000, 8000000)
    #test_space_invaders("./train/best_model_7800000")

