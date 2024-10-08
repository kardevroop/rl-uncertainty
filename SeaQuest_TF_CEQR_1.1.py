# CEQR-DQN - original
# conda deactivate
# conda activate TF_Torch_RL
# cd /home/sr8685/
# /home/sr8685/anaconda3/envs/TF_Torch_RL/bin/python /home/sr8685/RL/Atari/SeaQuest_TF_CEQR_1.1.py

# pip install tensorflow==2.12.0
# pip install gym[atari]
# pip install gymnasium==0.29.1
# pip install MinAtar

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import random
import numpy as np
import time
import gymnasium as gym
from collections import deque
import seaborn as sns
import tensorflow as tf
from datetime import datetime
import logging
tf.get_logger().setLevel(logging.ERROR)
from tensorflow import keras
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.optimizers import adam_v2
from keras import layers, Sequential, Model
from keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.optimizers import Adam
from renu_utils import run_time, save_dir, plot_graphs

# Before disabling GPUs
print("\nBefore disabling GPUs")
print("# GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("Available GPUs: ", tf.config.list_physical_devices('GPU'))

run_time_calc = run_time()

parent_dir = '/home/sr8685/RL/Atari/results_TF/CEQR-DQN/'
file_name = '-TF-SeaQuest-CEQR-Orig'
file_dir = save_dir(run_time_calc, parent_dir, file_name)

# After disabling GPUs
try:
    # Disable all GPUS
    print("\nTry block")
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'
except:
    print("\nExcept block")
    # Invalid device or cannot modify virtual devices once initialized.
    pass

print("\nAfter disabling GPUs")
print("# GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("Available GPUs: ", tf.config.list_physical_devices('GPU'), "\n")

def QRegLoss(pred, y):
    # n_quantiles = 20
    batch_size, num_quant = pred.shape
    
    pred = tf.tile(tf.expand_dims(pred, axis=2), [1, 1, num_quant])    
    y = tf.transpose(tf.tile(tf.expand_dims(y, axis=2), [1, 1, num_quant]), perm=[0, 2, 1])
    tau = np.linspace(0, 1-1/n_quantiles, n_quantiles) + 0.5/n_quantiles
    tau = tf.tile(tf.expand_dims(tf.expand_dims(tau, 0), 2), [batch_size, 1, num_quant])
    tau = tf.cast(tau, tf.float32)
    diff = y - pred
    if kappa == 0:
        huber_loss = tf.abs(diff)
    else:
        huber_loss = tf.abs(diff)
        huber_loss = tf.clip_by_value(huber_loss, clip_value_min=0.0, clip_value_max=kappa)
        huber_loss = tf.pow(huber_loss, 2)

    quantile_loss = huber_loss * tf.math.abs(tau - tf.cast(diff < 0, tf.float32))

    return tf.reduce_sum(tf.reduce_mean(tf.reduce_mean(quantile_loss, axis=2), axis=0))

def batch_cali_loss(y, x, lamda=0.5):
    batch_size, num_q = y.shape
    # Create quantile levels tensor
    q_list = tf.linspace(0.0, 1.0, num=num_q+2)[1:-1]
    # q_list = tf.expand_dims(q_list, axis=0)
    q_list = tf.tile(tf.expand_dims(q_list, 0), [batch_size, 1])  # Expand and tile along the first dimension
    # q_list = q_list.expand(batch_size, -1)  # shape (batch_size, num_q)

    # Calculate coverage and indicator matrices
    idx_under = (y <= x)
    idx_over = ~idx_under
    coverage = tf.reduce_mean(tf.cast(idx_under, tf.float32), axis=1)  # shape (batch_size,)

    # Calculate mean differences where predictions are under or over the targets
    mean_diff_under = tf.reduce_mean((y - x) * tf.cast(idx_under, tf.float32), axis=1)
    # print(f"mean under tf.expand_dims(mean_diff_under, axis=1): {tf.expand_dims(mean_diff_under, axis=1)}")
    mean_diff_over = tf.reduce_mean((y - x) * tf.cast(idx_over, tf.float32), axis=1)
    # print(f"mean over tf.expand_dims(mean_diff_over, axis=1): {tf.expand_dims(mean_diff_over, axis=1).shape}")

    # Determine whether each prediction falls under or over the corresponding quantile
    # cov_under = coverage.unsqueeze(1) < q_list
    cov_under = tf.expand_dims(coverage, axis=1) < q_list
    # print(f"cov under: {cov_under.shape}")
    cov_over = ~cov_under
    # print(f"cov over: {cov_over}")
    
    # print(f"cov_under * mean_diff_over.unsqueeze(1): {(tf.cast(cov_under, dtype=tf.float32) * tf.expand_dims(mean_diff_over, axis=1))}")
    # print(f"cov_over * mean_diff_under.unsqueeze(1): {(tf.cast(cov_over, dtype=tf.float32) * tf.expand_dims(mean_diff_under, axis=1))}")
    loss_list = (tf.cast(cov_under, dtype=tf.float32) * tf.expand_dims(mean_diff_over, axis=1)) + (tf.cast(cov_over, dtype=tf.float32) * tf.expand_dims(mean_diff_under, axis=1))

    # handle scaling
    with tf.GradientTape() as tape:
        cov_diff = tf.abs(tf.expand_dims(coverage, axis=1) - q_list)
        
    loss_list = cov_diff * loss_list
    loss = tf.reduce_mean(loss_list)

    # handle sharpness penalty
    # Assume x already contains the predicted quantiles
    # Hence, for opposite quantiles, we simply use 1 - x
    opp_pred_y = 1.0 - x

    # Check if each quantile is below or above the median (0.5)
    with tf.GradientTape() as tape:
        below_med = q_list <= 0.5
        above_med = ~below_med

    # Calculate sharpness penalty based on whether the quantile is below or above the median
    sharp_penalty = tf.cast(below_med, dtype=tf.float32) * (opp_pred_y - x) + tf.cast(above_med, dtype=tf.float32) * (x - opp_pred_y)
    # print(f"sharp penalty: {sharp_penalty.shape}")    
    with tf.GradientTape() as tape:
        width_positive = sharp_penalty > 0.0
        # print(f"width positive: {width_positive.shape}")

    # Penalize sharpness only if centered interval observation proportions are too high
    # Calculate expected and observed interval proportions
    with tf.GradientTape() as tape:
        exp_interval_props = tf.abs((2 * q_list) - 1)
        interval_lower = tf.minimum(x, opp_pred_y)
        interval_upper = tf.maximum(x, opp_pred_y)

        # Check if y falls within the predicted interval
        within_interval = (interval_lower <= y) & (y <= interval_upper)
        obs_interval_props = tf.reduce_mean(tf.cast(within_interval, tf.float32), axis=1)
        # obs_interval_props = obs_interval_props.unsqueeze(1).expand(-1, num_q)
        obs_interval_props = tf.tile(tf.expand_dims(obs_interval_props, axis=1), [1, num_q])

        obs_over_exp = obs_interval_props > exp_interval_props
        # print(f"obs over exp: {obs_over_exp.shape}")

        # Reshape tensors to match the required dimensions for the penalty calculation
        #obs_over_exp = obs_over_exp.unsqueeze(1).expand(-1, num_q)
        # width_positive = width_positive.expand(-1, num_q)
        # width_positive = tf.tile(width_positive, [1, num_q])
        # print(f"width positive before: {width_positive.shape}")
        # width_positive = tf.repeat(width_positive, repeats=num_q, axis=1)
        # print(f"width positive: {width_positive.shape}")

    # Apply sharpness penalty based on whether observed interval proportions are too high
    sharp_penalty = tf.cast(obs_over_exp, dtype=tf.float32) * tf.cast(width_positive, dtype=tf.float32) * sharp_penalty

    loss = ((1 - lamda) * loss) + (
        (lamda) * tf.reduce_mean(sharp_penalty)
    )

    return loss

def loss_fn(x,y):
    quant_loss = QRegLoss(x,y) # for distributional RL quantile loss
    cal_loss = batch_cali_loss(y, x)    
    return quant_loss + cal_loss

def intervalscore(label, upper, lower):

    s = (upper - lower)
    
    label_lt_lower = tf.cast(label < lower, dtype=tf.float32)
    label_gt_upper = tf.cast(label > upper, dtype=tf.float32)

    s += label_lt_lower * (2 / 0.95) * (lower - label) + label_gt_upper * (2 / 0.95) * (label - upper)
    s += label_lt_lower * (2 / 0.05) * (lower - label) + label_gt_upper * (2 / 0.05) * (label - upper)

    s_mean = tf.reduce_mean(tf.reduce_mean(s, axis=0))

    return s_mean

def combinedcalibrationloss(label, upper, lower, lamda):

    marginalcov = tf.reduce_mean(tf.cast((lower <= label) & (label <= upper), tf.float32), axis=0)
    
    marginalcov_lt_09 = marginalcov < 0.9
    marginalcov_gt_09 = marginalcov > 0.9

    calibration_upper = tf.where(marginalcov_lt_09, tf.clip_by_value(label - upper, clip_value_min=0, clip_value_max=float('inf')), tf.clip_by_value(upper - label, clip_value_min=0, clip_value_max=float('inf')))
    calibration_lower = tf.where(marginalcov_lt_09, tf.clip_by_value(label - lower, clip_value_min=0, clip_value_max=float('inf')), tf.clip_by_value(lower - label, clip_value_min=0, clip_value_max=float('inf')))

    calibration_upper = tf.reduce_mean(tf.reduce_mean(calibration_upper, axis=0))
    calibration_lower = tf.reduce_mean(tf.reduce_mean(calibration_lower, axis=0))
    calibration = calibration_upper + calibration_lower

    sharpness = upper - lower
    sharpness = tf.where(tf.expand_dims(marginalcov_gt_09, axis=0), sharpness, tf.zeros_like(sharpness))
    sharpness = tf.reduce_mean(tf.reduce_mean(sharpness, axis=0))

    combinedcalloss = (1 - lamda) * calibration + lamda * sharpness

    return combinedcalloss

def gamma_cal_loss(mu, y, lamda):
    gamma_lowq = mu[:,0]
    gamma_upq = mu[:,1]
    if tf.rank(gamma_lowq) == 1:
        gamma_lowq = tf.expand_dims(gamma_lowq, axis=-1)
        gamma_upq = tf.expand_dims(gamma_upq, axis=-1)
    interval_loss = intervalscore(y,gamma_upq,gamma_lowq) 
    calibrationloss = combinedcalibrationloss(y,gamma_upq,gamma_lowq,lamda) # for evidentical quantile calibration loss
    evi_cal_loss = interval_loss + calibrationloss

    return evi_cal_loss

def NIG_NLL(y, mu, v, alpha, beta, reduce=True):
    twoBlambda = 2*beta*(1+v)

    nll = 0.5*tf.math.log(np.pi/v)  \
        - alpha*tf.math.log(twoBlambda)  \
        + (alpha+0.5) * tf.math.log(v*(y-mu)**2 + twoBlambda)  \
        + tf.math.lgamma(alpha)  \
        - tf.math.lgamma(alpha+0.5)

    return tf.reduce_mean(nll) if reduce else nll

def KL_NIG(mu1, v1, a1, b1, mu2, v2, a2, b2):
    KL = 0.5 * (a1 - 1) / b1 * (v2 * tf.square(mu2 - mu1))  \
        + 0.5 * v2 / v1  \
        - 0.5 * tf.math.log(tf.abs(v2) / tf.abs(v1))  \
        - 0.5 + a2 * tf.math.log(b1 / b2)  \
        - (tf.math.lgamma(a1) - tf.math.lgamma(a2))  \
        + (a1 - a2) * tf.math.digamma(a1)  \
        - (b1 - b2) * a1/b1
    return KL

def NIG_Reg(y, mu, v, alpha, beta, q, omega=0.01, reduce=True, kl=False):
    # error = tf.stop_gradient(tf.abs(y-mu))
    # error = tf.abs(y-mu)
    e = y - mu
    error = tf.maximum(q * e, (q - 1) * e)

    if kl:
        kl = KL_NIG(mu, v, alpha, beta, mu, omega, 1+omega, beta)
        reg = error*kl
    else:
        evi = 2 * v + alpha + 1/beta
        reg = error*evi

    return tf.reduce_mean(reg) if reduce else reg

def EDLLoss(pred, mu, v, alpha, beta, evi_coeff=1):
    loss = 0.0

    for i, q in enumerate([0.05, 0.95]):
        mu_i = mu[:, i]
        v_i = v[:, i]
        alpha_i = alpha[:, i]
        beta_i = beta[:, i]

        loss_nll = NIG_NLL(pred, mu_i, v_i, alpha_i, beta_i)
        loss_reg = NIG_Reg(pred, mu_i, v_i, alpha_i, beta_i, q)

        loss += loss_nll + evi_coeff * loss_reg

    return loss

def init_weights(layer, gain):
    if isinstance(layer, Dense) or isinstance(layer, Conv2D):
        initializer = tf.keras.initializers.Orthogonal(gain=1) # added gain=1; original: gain=gain
        layer.kernel_initializer = initializer
        # if layer.bias is not None:
        if hasattr(layer, 'bias') and layer.use_bias:
            if layer.bias is not None:
                layer.bias_initializer = tf.keras.initializers.Zeros()

class CNN_EDL(Model):
    def __init__(self, n_outputs, weight_scale=np.sqrt(2)):
        super(CNN_EDL, self).__init__();
        self.weight_scale = weight_scale
        print(f"\nquants: {n_quantiles}, n_outs: {n_outputs}, n_actions: {action_size}, chs: {state_size}\n")
        
        # inputs = tf.keras.Input(shape=(1, 10, 10, 10))
        # self.conv = tf.keras.layers.Conv2D(16, kernel_size=3, strides=1, padding="same", input_shape=(10, 10, 10))
        # self.conv = tf.keras.layers.Conv2D(16, kernel_size=3, strides=1, padding="same", activation='prelu', input_shape=(10, 10, 10))
        self.conv = tf.keras.Sequential([
            tf.keras.layers.Conv2D(16, (3, 3), strides=1, padding='same',
                                   kernel_initializer=tf.keras.initializers.HeNormal()),
            tf.keras.layers.PReLU(alpha_initializer=tf.constant_initializer(0.25))
        ])
        # self.fc_hidden = layers.Dense(1600)
        self.fc_hidden = tf.keras.Sequential([
            tf.keras.layers.Dense(1600, activation='relu', 
                                  kernel_initializer=tf.keras.initializers.HeNormal())
        ])
        # self.output_layer = layers.Dense(n_outputs)
        self.output_layer = tf.keras.Sequential([
            tf.keras.layers.Dense(n_outputs, kernel_initializer=tf.keras.initializers.HeNormal())
        ])
        # self.edl = layers.Dense(4 * action_size * 2 * n_quantiles) # 4 EDL outputs (mu, nu, alpha, beta)
        self.edl = tf.keras.Sequential([
            tf.keras.layers.Dense(4 * action_size * 2 * n_quantiles, 
                                  kernel_initializer=tf.keras.initializers.HeNormal())
        ])
        
        # Apply custom weight initialization
        init_weights(self.conv, self.weight_scale)
        init_weights(self.fc_hidden, self.weight_scale)
        init_weights(self.output_layer, self.weight_scale)
        init_weights(self.edl, self.weight_scale)

    def call(self, x):
        if len(x.shape) != 4:
            x = tf.expand_dims(x, axis=0)
        # print(f"input: {x}")
        x = tf.transpose(x, perm=[0, 3, 1, 2])  # Permute to [batch, channels, height, width]
        # print(f"input transposed: {x}")
        x = self.conv(x)
        # x = tf.keras.layers.PReLU(alpha_initializer=tf.constant_initializer(0.25))(x)
        # print(f"conv: {x}")
        # x = tf.transpose(x, perm=[0, 3, 1, 2])
        # print(f"conv transposed: {x.shape}")
        x = tf.reshape(x, shape=(x.shape[0], -1))  # Flatten
        # print(f"flattened: {x}")
        # x = tf.nn.relu(self.fc_hidden(x))
        x = self.fc_hidden(x)
        # print(f"hidden: {x}")
        out = self.output_layer(x)
        # print(f"out: {out}")
        evidence = self.edl(x)
        # print(f"evidence: {x.shape}")

        # split_size = action_size * 2 * n_quantiles
        mu, logv, logalpha, logbeta = tf.split(evidence, 4, axis=-1)
        # v = tf.nn.softplus(logv)
        v = tf.keras.activations.softplus(logv)
        # alpha = tf.nn.softplus(logalpha) + 1
        alpha = tf.keras.activations.softplus(logalpha) + 1
        # beta = tf.nn.softplus(logbeta)
        beta = tf.keras.activations.softplus(logbeta)
        
        return out, tf.concat([mu, v, alpha, beta], axis=-1)

def multivariate_normal_sample(mean, covariance_matrix, num_samples=1):
    chol_cov = tf.linalg.cholesky(covariance_matrix)
    std_normal_samples = tf.random.normal((num_samples, tf.shape(mean)[0]))
    samples = tf.matmul(std_normal_samples, chol_cov, transpose_b=True) + mean
    
    return samples

def mvn_act(state):
    al_factor = 0 # 0.5
    ep_factor = 0.005 # 0.2
    
    net, evidence = model(state)
    net = tf.reshape(net, (action_size, n_quantiles))
    action_mean = tf.reduce_mean(net, axis=1).numpy()
    mu, v, alpha, beta = tf.split(evidence, 4, axis=-1)
    
    mu = tf.reshape(mu, (action_size, 2, n_quantiles))
    v = tf.reshape(v, (action_size, 2, n_quantiles))
    alpha = tf.reshape(alpha, (action_size, 2, n_quantiles))
    beta = tf.reshape(beta, (action_size, 2, n_quantiles))
    # print(f"mu: {mu}")
    # print(f"v: {v}")
    # print(f"alpha: {alpha}")
    # print(f"beta: {beta}")
    
    variance = tf.sqrt((beta /(v*(alpha - 1))))
    # print(f"variance: {variance}")
    # print(f"\nmu diff shape: {tf.abs(mu[:,1,:] - mu[:,0,:]).shape}")
    # print(f"mu diff: {tf.abs(mu[:,1,:] - mu[:,0,:])}")
    u_al = tf.reduce_mean(tf.abs(mu[:,1,:] - mu[:,0,:]),axis=1)
    u_ep = tf.reduce_mean(0.5*(tf.abs((mu[:,0,:]+variance[:,0,:])-(mu[:,0,:]-variance[:,0,:])) + tf.abs((mu[:,1,:]+variance[:,1,:])-(mu[:,1,:]-variance[:,1,:]))),axis=1) + 1e-8

    print(f"u_al: {u_al}")
    # adjusted_action_mean = action_mean - (al_factor * u_al)
    adjusted_action_mean = action_mean - (ep_factor * u_ep)
    print(f"u_ep: {u_ep}")
    diag_uncertainties = ep_factor * np.diagflat(u_ep)

    Q_hat = multivariate_normal_sample(action_mean, diag_uncertainties)
    action = np.argmax(Q_hat)
    print(f"action: {action}")
    u_al_0.append(u_al[0])
    u_al_1.append(u_al[1])
    u_ep_0.append(tf.sqrt(u_ep)[0])
    u_ep_1.append(tf.sqrt(u_ep)[1])
    mean_0.append(action_mean[0])
    mean_1.append(action_mean[1])

    return action

def replay(memory, batch_size):     
    minibatch = random.sample(memory, batch_size)
    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []
    # for m in minibatch:        
    #     print(f"Minibatch: {m[0].shape}")
    for m in minibatch:
        mod_state = m[0]
        if len(m[0].shape) != 4:
            mod_state = tf.expand_dims(m[0], axis=0)
        states.append(mod_state)
        actions.append(m[1])
        rewards.append(m[2])
        mod_next_state = m[3]
        if len(m[3].shape) != 4:
            mod_next_state = tf.expand_dims(m[3], axis=0)
        next_states.append(mod_next_state)
        dones.append(m[4])
    return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

game = "MinAtar/Seaquest-v1"
seed=random.randint(0, 1e6) # 190890
notes = "UADQN"
env = gym.make(game) #, random_seed=seed)
print(f"# Actions: {env.action_space.n}")
print(f"Observations' shape: {env.observation_space.shape}")
# env = MinAtarWrapper(env)

height = env.observation_space.shape[0]
print(f"Height: {height}")
width = env.observation_space.shape[1]
print(f"Width: {width}")
state_size = env.observation_space.shape[2]
print(f"State Size: {state_size}")
action_size = env.action_space.n
print(f"Action Size: {action_size}")

n_seeds = 1
network = CNN_EDL
n_quantiles=50
weight_scale=3
noise_scale=1
epistemic_factor=0.005
aleatoric_factor=0
evi_coeff = 0.5
kappa=1 # 0
replay_start_size=5000
timesteps=2 # 1000000 # 5000000
replay_buffer_size=100000
gamma=0.99
update_target_frequency=1000
minibatch_size=1 # 32
learning_rate=1e-4
adam_epsilon=1e-8
log_folder_details=game+'-CEQR-DQN'
update_frequency=1
save_period=1e7
seed=seed

u_al_0 = []; tu_al_0 = [];
u_al_1 = []; tu_al_1 = [];
u_ep_0 = []; tu_ep_0 = [];
u_ep_1 = []; tu_ep_1 = [];
mean_0 = []; tmean_0 = [];
mean_1 = []; tmean_1 = [];
ep_score = []; ep_score_t = [];

# set seeds
tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)
env.seed(seed)

lossfn = loss_fn
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, epsilon=adam_epsilon)
memory = deque(maxlen=replay_buffer_size)

n_outputs = 6 * n_quantiles

model = CNN_EDL(n_outputs)
model.build((10, 10, 10))
model.summary();

target_net = CNN_EDL(n_outputs)
target_net.build((10, 10, 10))
target_net.summary()
target_net.set_weights(model.get_weights())

params = list(model.trainable_variables)
optimizer = Adam(learning_rate=learning_rate)
optimizer.build(params)

not_greedy = 0
curr_step = 0

losses = []; losses_t = [];

state, _ = env.reset()
score = 0
t1 = time.time()

for t in range(timesteps):

    is_training_ready = t >= replay_start_size

    print(f"\nTimestep: {t}; is_training_ready: {is_training_ready}")
    # print(f"state: {tf.cast(state, dtype=tf.float32)}")
    action = mvn_act(tf.cast(state, dtype=tf.float32));
    
    tu_al_0.append(curr_step)
    tu_al_1.append(curr_step)
    tu_ep_0.append(curr_step)
    tu_ep_1.append(curr_step)
    tmean_0.append(curr_step)
    tmean_1.append(curr_step)
                
    next_state, reward, terminated, truncated, info = env.step(action);
    done = terminated or truncated
    next_state = tf.reshape(next_state, (1,  height, width, state_size));
    memory.append((state, action, reward, next_state, done));

    score += reward;
    ep_score_t.append(t)
    ep_score.append(score)

    if done:
        print("Timestep: {}, score: {}, Time: {} s".format(t, score, round(time.time() - t1, 3)))
        # print(f"Timestep: {t}; Episode score: {score}")
        state, _ = env.reset()
        score = 0
        t1 = time.time()
        not_greedy = 0

    else:
        state = next_state

    if is_training_ready:
        if t % update_frequency == 0:
            replay_buffer = replay(memory, minibatch_size)
            states, actions, rewards, next_states, dones = replay_buffer

            with tf.GradientTape(persistent=True) as tape:
                tape.watch([])
                next_states = tf.reshape(next_states, (minibatch_size, height, width, state_size))
                target, _ = target_net(tf.cast(next_states, dtype=tf.float32))
                target = tf.reshape(target, (minibatch_size, action_size, n_quantiles))
                
                best_action_val = tf.math.reduce_max(tf.reduce_mean(target, axis = 2).numpy(), axis=1, keepdims=True)
                best_action_id = tf.math.argmax(tf.reduce_mean(target, axis = 2).numpy(), axis=1, output_type=tf.int32)
                best_action = tf.expand_dims(best_action_id, axis=1)
                float_terminateds = [float(elem) for elem in dones]
                float_dones = np.array(float_terminateds).reshape(-1, 1)
                rewards_expanded = tf.tile(tf.expand_dims(tf.expand_dims(rewards, axis=1), axis=2), [1, 1, n_quantiles])
                dones_expanded = tf.tile(tf.expand_dims(float_dones, axis=2), [1, 1, n_quantiles])
                TD_target = tf.cast(rewards_expanded, tf.float32) + tf.cast(1 - dones_expanded, tf.float32) * tf.cast(gamma, tf.float32) * tf.cast(tf.expand_dims(best_action_val, axis=2), tf.float32) # * tf.cast(best_action_val, tf.float32)
                
                states = tf.reshape(states, (minibatch_size, height, width, state_size))
                out, evidence = model(tf.cast(states, dtype=tf.float32))
                out = tf.reshape(out, (minibatch_size, action_size, n_quantiles))

                mu, v, alpha, beta = tf.split(evidence, 4, axis=-1)
                mu = tf.reshape(mu, (minibatch_size, action_size, 2, n_quantiles))
                v = tf.reshape(v, (minibatch_size, action_size, 2, n_quantiles))
                alpha = tf.reshape(alpha, (minibatch_size, action_size, 2, n_quantiles))
                beta = tf.reshape(beta, (minibatch_size, action_size, 2, n_quantiles))

                actions_expanded = tf.tile(tf.expand_dims(tf.expand_dims(actions, axis=1), axis=2), [1, 1, n_quantiles]) # (32, 1, 50)

                a_one_hot = tf.one_hot(actions_expanded, depth=action_size, dtype=out.dtype)  # Shape (32, 1, 50, 18)
                result = tf.reduce_sum(tf.expand_dims(out, axis=1) * tf.transpose(a_one_hot, perm=[0, 1, 3, 2]), axis=2)  # Shape (32, 1, 50)

                loss = lossfn(tf.squeeze(result), tf.squeeze(TD_target))
                # print(f"loss: {loss}")
                loss_edl = EDLLoss(TD_target, mu, v, alpha, beta, evi_coeff)
                # print(f"loss_evidence: {loss_edl}")        
                loss_mu_calib = gamma_cal_loss(mu, TD_target, 0.5)
                # print(f"loss_gamma: {loss_mu_calib}")

                total_loss = loss + loss_edl + loss_mu_calib

                model_gradients = tape.gradient(total_loss, model.trainable_variables)
                
            optimizer.apply_gradients(zip(model_gradients, model.trainable_variables))
            del tape

            print(f"Timestep: {t}; Loss: {loss};")
            if done:
                pass
            
            losses_t.append(t)
            losses.append(loss)
        
        if t % update_target_frequency == 0:
            target_net.set_weights(model.get_weights())
    curr_step += 1

plotter = plot_graphs(file_dir.graph_dir, window_size=100)
for boole in (True, False):
    plotter.plot('score', np.array(ep_score_t), np.array(ep_score), mov_avg=boole)
    plotter.plot('loss', np.array(losses_t), np.array(losses), mov_avg=boole)
    plotter.plot('mean0', np.array(tmean_0), np.array(mean_0), mov_avg=boole)
    plotter.plot('mean1', np.array(tmean_1), np.array(mean_1), mov_avg=boole)

    plotter.plot('u_al0', np.array(tu_al_0), np.array(u_al_0), mov_avg=boole)
    plotter.plot('u_al1', np.array(tu_al_1), np.array(u_al_1), mov_avg=boole)
    plotter.plot('u_ep0', np.array(tu_ep_0), np.array(u_ep_0), mov_avg=boole)
    plotter.plot('u_ep1', np.array(tu_ep_1), np.array(u_ep_1), mov_avg=boole)

n = file_dir.print_dir()
run_time_calc.print_time(n)
print(f"\nExecuted file: {os.path.basename(__file__)}\n")