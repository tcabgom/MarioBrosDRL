from keras.backend import abs, less_equal, square, switch
from keras.layers import Conv2D, Dense, Flatten, Input, Lambda, Multiply
from keras.models import Model
from keras.optimizers import RMSprop
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

CHECKPOINT_DIR = "train/"
LOG_DIR = "logs/"


def huber_loss(truth, prediction, delta: float=1.0):
    residual = abs(prediction-truth)
    condition = less_equal(residual, prediction)
    success_return = 0.5 * square(residual)
    fail_return = delta * residual - 0.5 * square(delta)
    return switch(condition, success_return, fail_return)


def create_custom_neural_network_dqn_model(
        image: tuple = (84, 84),
        action_space: int = 7,
        stacked_frames: int = 4,
        loss=huber_loss,
        optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01),
):

    # https://keras.io/api/layers/
    # NEURAL NETWORK ARCHITECTURE:
    # Input layer (Number of units determined by the image size and stacked frames)
    # Hidden 2D convolutional layer (Filters: 32, Filter size: 8x8, Stride: 4x4)
    # Hidden 2D convolutional layer (Filters: 64, Filter size: 4x4, Stride: 2x2)
    # Hidden 2D convolutional layer (Filters: 64, Filter size: 3x3, Stride: 1x1)
    # Hidden dense layer (Units 512)
    # Output layer (Number of units determined by action space)

    convolutional_network_input = Input((*image, stacked_frames), name='cnn')
    cnn = Lambda(lambda x: x / 255.0)(convolutional_network_input)
    cnn = Conv2D(32, (8, 8), strides=(4, 4), activation='relu')(cnn)
    cnn = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(cnn)
    cnn = Conv2D(64, (3, 3), activation='relu')(cnn)
    cnn = Flatten()(cnn)
    cnn = Dense(512, activation='relu')(cnn)
    cnn = Dense(action_space)(cnn)

    mask_input = Input((action_space,), name='mask')
    output = Multiply()([cnn, mask_input])

    model = Model(inputs=[convolutional_network_input, mask_input], outputs=output)
    model.compile(loss=loss, optimizer=optimizer)

    model = DQN('CnnPolicy', DummyVecEnv([lambda: None]),
                policy_kwargs={'cnn_extractor': model},
                verbose=1,
                tensorboard_log=LOG_DIR,
                learning_rate=0.01,
                buffer_size=100000,
                learning_starts=20000,
                train_freq=2,
                gradient_steps=1,
                target_update_interval=10000,
                exploration_fraction=0.05,
                exploration_final_eps=0.01,
                max_grad_norm=10,
                gamma=0.99,
                device='cuda:0'
                )

    return model

