# Keras 2.1.6, tensorflow 1.12.0
import numpy as np
import new_vissim_env as dv


def init_rl():
    # Init a DDPG
    BUFFER_SIZE = 1000000
    BATCH_SIZE = 32
    GAMMA = 0.99
    TAU = 0.001  # Target Network HyperParameters
    LRA = 0.0001  # Learning rate for Actor
    LRC = 0.001  # Lerning rate for Critic

    action_dim = 2  # acce/lc
    state_dim = 26  # of sensors input

    np.random.seed(5274)

    EXPLORE = 1000000
    done = 0
    epsilon = 1

    total_loss = 0
    total_reward = 0
    # Whether train 1 or run only 0
    train_indicator = 0
    s_t = 0
    a_t = 0
    r_t = 0
    s_t1 = 0
    speed_limit = 22
    sensor_dis = 100
    ddpg = dv.DDPG(BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU, LRA, LRC, action_dim, state_dim, EXPLORE, epsilon,
                   total_loss, total_reward, train_indicator, s_t, a_t, r_t, s_t1,
                   done, speed_limit, sensor_dis)
    print("Now we load the weight")
    try:
        ddpg.actor.model.load_weights("actormodel.h5")
        ddpg.critic.model.load_weights("criticmodel.h5")
        ddpg.actor.target_model.load_weights("actor_target_model.h5")
        ddpg.critic.target_model.load_weights("critic_target_model.h5")
        print("Weight load successfully")
    except:
        print("Cannot find the weight")
    return ddpg


def rl_control(ddpg, info):
    # info is a dict, corresponding to the info of func 'make_observaton' in new_vissim_env
    acce, lc_flag = ddpg.actor_output_LC(info)
    return acce, lc_flag


# single step usage
ddpg = init_rl()
# input info around the subject vehicle, gap (m), vel(m/s)
input_info = {"gap_lead": 50, "vel_lead": 15, "vel": 12, "gap_leftlead": 20, "gap_leftlag": 30, "gap_rightlead": 40,
              "gap_rightlag": 50, "gap_lag": 20, "vel_rightlead": 12, "vel_rightlag": 10, "vel_leftlead": 11,
              "vel_leftlag": 13, "vel_lag": 16}
acce, lc_flag = rl_control(ddpg, input_info)
print(acce, lc_flag)
