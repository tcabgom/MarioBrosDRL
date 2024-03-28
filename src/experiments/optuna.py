import json
import torch
import gymnasium
from stable_baselines3.common.monitor import Monitor

import optuna
from src.experiments import environment_preprocessing, experiment_utils


def objective_aux(trial):
    # https://optuna.org/

    env = gymnasium.make("ALE/SpaceInvaders-v5", render_mode='rgb_array')
    experiment_utils.print_environment_data(env)
    env = Monitor(env, experiment_utils.LOG_DIR)
    env = environment_preprocessing.reduce_observation_space(env)
    env = environment_preprocessing.enhance_observation_space(env)
    experiment_utils.print_environment_data(env)
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
    with open(f"{experiment_utils.CHECKPOINT_DIR}/hyperparams_trial_{trial.number}.json", 'w') as f:
        json.dump(hyperparams, f)

    model = create_custom_DQN_model(env, exploration_final_eps, learning_rate, train_frequency, buffer_size, 0.95)
    print(exploration_final_eps, learning_rate, train_frequency, buffer_size, 0.95)
    callback = experiment_utils.TrainAndLoggingCallback(save_path=experiment_utils.CHECKPOINT_DIR, model_name='dqn_optuna_3', check_freq=1000,
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


def search_hyperparameters_optuna():
    study = optuna.create_study(direction='maximize')
    results = []
    study.optimize(objective, n_trials=50)
    print('### TRIALS COMPLETE ###')
    trial = study.best_trial
    print('Best Value: ', trial.value)
    print('Best Params: ')
    for key, value in trial.params.items():
        print(f'    {key}: {value}')
