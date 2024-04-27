from stable_baselines3 import PPO
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike

import experiment_utils


def create_PPO_model(env) -> PPO:
    return PPO("CnnPolicy", env,
                verbose=1,                                  # Controls the verbosity level (0: no output, 1: training information)
                tensorboard_log=experiment_utils.LOG_DIR,   # Directory for storing Tensorboard logs
                learning_rate=0.00025,                     # The learning rate for the optimizer
                #gamma=0.99,                                 # Discount factor for future rewards
                #gae_lambda=0.95,                            # Lambda for the Generalized Advantage Estimator
                #max_grad_norm=0.5,                          # Clipping of gradients during optimization
                n_steps=2048,                                # Number of steps to run for each environment per update
                #n_epochs=4,                                 # Number of epochs when optimizing the surrogate loss
                #batch_size=256,                             # Minibatch size for each environment
                #clip_range=0.1,                             # Clipping parameter, mainly used for PPO
                #ent_coef=0.01,                              # Entropy coefficient for the loss calculation
                #vf_coef=0.5,                                # Value function coefficient for the loss calculation
                #policy_kwargs=dict(optimizer_class=RMSpropTFLike, optimizer_kwargs=dict(eps=1e-5))     # Additional network architecture
                # device = "cuda:0"
                )


def create_custom_PPO_model(env, n_steps, gamma, learning_rate, ent_coef, vf_coef, clip_range, clip_range_vf) -> PPO:
    return PPO("CnnPolicy", env,
                verbose=1,
                tensorboard_log=experiment_utils.LOG_DIR,
                learning_rate=learning_rate,
                n_steps=n_steps,
                batch_size=64,
                n_epochs=4,
                gamma=gamma,
                gae_lambda=0.95,
                clip_range=clip_range,
                ent_coef=ent_coef,
                vf_coef=vf_coef,
                max_grad_norm=0.5,
                policy_kwargs=dict(net_arch=[64, 64])
                # device = "cuda:0"
                )
