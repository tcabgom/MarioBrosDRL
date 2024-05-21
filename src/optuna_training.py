import gc
import json
import torch
import gymnasium
from stable_baselines3.common.monitor import Monitor

import A2C_model, DQN_model, PPO_model
import optuna
import environment_preprocessing
import experiment_utils



def objective_aux(trial):
    # https://optuna.org/

    #env = gymnasium.make("ALE/SpaceInvaders-v5", render_mode='rgb_array')
    env = gymnasium.make("SpaceInvaders-ramNoFrameskip-v4", render_mode='rgb_array')
    experiment_utils.print_environment_data(env)
    env = Monitor(env, experiment_utils.LOG_DIR)
    env = environment_preprocessing.reduce_observation_space(env)
    env = environment_preprocessing.enhance_observation_space(env)
    experiment_utils.print_environment_data(env)

    #model = suggest_hyperparameters_DQN(trial, env)
    model = suggest_hyperparameters_A2C(trial, env)
    #model = suggest_hyperparameters_PPO(trial, env)

    callback = experiment_utils.TrainAndLoggingCallback(save_path=experiment_utils.CHECKPOINT_DIR, model_name='a2c_optuna_ram', check_freq=1000,
                                       save_freq_best=500000, save_freq_force=1000000)
    model.learn(total_timesteps=2000000, callback=callback)

    ret = float(callback.current_best_info_mean['r'])
    torch.cuda.empty_cache()
    del callback
    del env
    del model
    return ret


def suggest_hyperparameters_DQN(trial, env):
    # For DQN experiments
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
        'gamma': 0.95
    }

    with open(f"{experiment_utils.CHECKPOINT_DIR}/hyperparams_trial_{trial.number}.json", 'w') as f:
        json.dump(hyperparams, f)

    model = DQN_model.create_custom_DQN_model(env, exploration_final_eps, learning_rate, train_frequency, buffer_size, 0.95) #TODO
    return model


def suggest_hyperparameters_A2C(trial, env):
    # For A2C experiments
    learning_rate = trial.suggest_float('learning_rate', 0.0001, 0.01)
    n_steps = trial.suggest_int('n_steps', 8, 32)
    gamma = trial.suggest_float('gamma', 0.95, 0.999)
    ent_coef = trial.suggest_float('ent_coef', 0.001, 0.033)
    vf_coef = trial.suggest_float('vf_coef', 0.25, 0.75)

    hyperparams = {
        'learning_rate': learning_rate,
        'n_steps': n_steps,
        'gamma': gamma,
        'ent_coef': ent_coef,
        'vf_coef': vf_coef
    }

    with open(f"{experiment_utils.CHECKPOINT_DIR}/hyperparams_trial_{trial.number}.json", 'w') as f:
        json.dump(hyperparams, f)

    model = A2C_model.create_custom_A2C_mlp_model(env, learning_rate, n_steps, gamma, ent_coef, vf_coef)
    return model


def suggest_hyperparameters_PPO(trial, env):
    # For PPO experiments
    n_steps = trial.suggest_int('n_steps', 5, 15)
    gamma = trial.suggest_float('gamma', 0.95, 0.99)
    learning_rate = trial.suggest_float('learning_rate', 0.0001, 0.01)
    ent_coef = trial.suggest_float('ent_coef', 0.0, 0.1)
    vf_coef = trial.suggest_float('vf_coef', 0.25, 0.75)
    clip_range = trial.suggest_float('clip_range', 0.1, 0.4)
    clip_range_vf = trial.suggest_float('clip_range_vf', 0.1, 0.4)

    hyperparams = {
        'n_steps': n_steps,
        'gamma': gamma,
        'learning_rate': learning_rate,
        'ent_coef': ent_coef,
        'vf_coef': vf_coef,
        'clip_range': clip_range,
        'clip_range_vf': clip_range_vf
    }

    with open(f"{experiment_utils.CHECKPOINT_DIR}/hyperparams_trial_{trial.number}.json", 'w') as f:
        json.dump(hyperparams, f)

    model = PPO_model.create_custom_PPO_model(env, n_steps, gamma, learning_rate, ent_coef, vf_coef, clip_range, clip_range_vf)
    return model


def objective(trial):
    ret = objective_aux(trial)
    gc.collect()
    return ret


def search_hyperparameters_optuna(n_trials):
    study = optuna.create_study(direction='maximize')
    results = []
    study.optimize(objective, n_trials=n_trials)
    print('### TRIALS COMPLETE ###')
    trial = study.best_trial
    print('Best Value: ', trial.value)
    print('Best Params: ')
    for key, value in trial.params.items():
        print(f'    {key}: {value}')
