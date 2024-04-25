from stable_baselines3 import A2C
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike

import experiment_utils


def create_A2C_model(env):
    # https://stable-baselines.readthedocs.io/en/master/modules/a2c.html
    return A2C("CnnPolicy", env,
               verbose=1,                                   # Controls the verbosity level (0: no output, 1: training information)
               tensorboard_log=experiment_utils.LOG_DIR,    # Directory for storing Tensorboard logs
               learning_rate=0.0005,                        # The learning rate for the optimizer
               gamma=0.99,                                  # Discount factor for future rewards
               ent_coef=0.01,                               # Entropy coefficient for the loss calculation
               vf_coef=0.25,                                 # Value function coefficient for the loss calculation
               max_grad_norm=0.5,                           # Clipping of gradients during optimization
               gae_lambda=0.95,                             # Lambda for the Generalized Advantage Estimator
               n_steps=16,                                  # Number of steps to run for each environment per update
               policy_kwargs=dict(optimizer_class=RMSpropTFLike, optimizer_kwargs=dict(eps=1e-5)),  # Additional network architecture
               normalize_advantage=False,                    # Whether to normalize the advantage
               #device = "cuda:0"
               )


def create_A2C_mlp_model(env):
    # Cambiamos la política a MlpPolicy para entrenar sobre RAM
    return A2C("MlpPolicy", env,
               verbose=1,
               tensorboard_log=experiment_utils.LOG_DIR,
               learning_rate=0.0005,
               gamma=0.99,
               ent_coef=0.01,
               vf_coef=0.25,
               max_grad_norm=0.5,
               gae_lambda=0.95,
               n_steps=16,
               policy_kwargs=dict(net_arch=[256, 256]),
               normalize_advantage=False,
               #device = "cuda:0"
               )


def create_custom_A2C_model(env, learning_rate, n_steps, gamma, ent_coef, vf_coef):
    return A2C("CnnPolicy", env,
               verbose=1,
               tensorboard_log=experiment_utils.LOG_DIR,
               learning_rate=learning_rate,
               gamma=gamma,
               ent_coef=ent_coef,
               vf_coef=vf_coef,
               max_grad_norm=0.5,
               gae_lambda=0.95,
               n_steps=n_steps,
               policy_kwargs=dict(net_arch=[64, 64]),
               #device = "cuda:0"
               )
