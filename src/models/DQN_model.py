from stable_baselines3 import DQN

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
                verbose=1,
                tensorboard_log=LOG_DIR,
                learning_rate=learning_rate,
                buffer_size=buffer_size,
                learning_starts=20000,
                train_freq=train_frequency,
                gradient_steps=1,
                target_update_interval=10000,
                exploration_fraction=0.05,
                exploration_final_eps=exploration_final_eps,
                max_grad_norm=10,
                gamma=gamma
                #device = "cuda:0"
                )