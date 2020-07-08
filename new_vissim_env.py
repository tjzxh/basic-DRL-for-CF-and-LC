import numpy as np
import random
import argparse
from keras.models import model_from_json, Model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
import tensorflow as tf
# from keras.engine.training import collect_trainable_weights
import json
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork


class DDPG(object):
    def __init__(self, BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU, LRA, LRC, action_dim, state_dim, EXPLORE, epsilon,
                 total_loss, total_reward, train_indicator, s_t, a_t, r_t, s_t1, done, speed_lmit, sensor_dis):
        self.BUFFER_SIZE = BUFFER_SIZE
        self.BATCH_SIZE = BATCH_SIZE
        self.GAMMA = GAMMA
        self.TAU = TAU
        self.LRA = LRA
        self.LRC = LRC
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.EXPLORE = EXPLORE
        self.epsilon = epsilon
        self.total_loss = total_loss
        self.total_reward = total_reward
        self.train_indicator = train_indicator
        self.s_t = s_t
        self.a_t = a_t
        self.r_t = r_t
        self.s_t1 = s_t1
        self.done = done
        self.speed_limit = speed_lmit
        self.sensor_dis = sensor_dis
        # Tensorflow GPU optimization
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        from keras import backend as K
        K.set_session(sess)
        self.actor = ActorNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRA)
        self.critic = CriticNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRC)

    def make_observaton(self, input_info):
        gap_lead = input_info["gap_lead"]
        vel_lead = input_info["vel_lead"]
        vel = input_info["vel"]
        gap_leftlead = input_info["gap_leftlead"]
        gap_leftlag = input_info["gap_leftlag"]
        gap_rightlead = input_info["gap_rightlead"]
        gap_rightlag = input_info["gap_rightlag"]
        gap_lag = input_info["gap_lag"]
        vel_rightlead = input_info["vel_rightlead"]
        vel_rightlag = input_info["vel_rightlag"]
        vel_leftlead = input_info["vel_leftlead"]
        vel_leftlag = input_info["vel_leftlag"]
        vel_lag = input_info["vel_lag"]
        Observation = np.array(
            [vel / 22, 0 / 2, (vel - vel_lead) / 11, gap_lead / 100, 0 / 2, 1.5 / 4,
             (vel - vel_leftlead) / 11, gap_leftlead / 100, 0 / 2, 1.5 / 4, (vel - vel_rightlead) / 11,
             gap_rightlead / 100,
             0 / 2, 1.5 / 4, (vel - vel_leftlag) / 11, gap_leftlag / 100, 0 / 2, 1.5 / 4,
             (vel - vel_rightlag) / 11, gap_rightlag / 100, 0 / 2, 1.5 / 4, (vel - vel_lag) / 11, gap_lag / 100,
             0 / 2, 1.5 / 4])
        return Observation

    def actor_output(self, input_info):
        s_t = self.make_observaton(input_info)
        action_original = self.actor.model.predict(s_t.reshape(1, s_t.shape[0]))[0][0]
        if action_original > 0:
            acce = action_original * 1.92
        else:
            acce = action_original * 1.97
        a_idm = acce

        # dynamic constraints
        if a_idm < -3:
            a_idm = -3
        if a_idm > 3:
            a_idm = 3
        return a_idm

    def actor_output_LC(self, input_info):
        s_t = self.make_observaton(input_info)
        action_original = self.actor.model.predict(s_t.reshape(1, s_t.shape[0]))
        acce_original = action_original[0][1]
        lc_original = action_original[0][0]
        if acce_original > 0:
            acce = acce_original * 3.5
        else:
            acce = acce_original * 8
        if 0 <= lc_original <= 0.1739523314093953:
            LaneChanging = -1
        elif 0.1739523314093953 <= lc_original <= 1 - 0.1739523314093953:
            LaneChanging = 0
        else:
            LaneChanging = 1
        acce_output = acce

        # # dynamic constraints
        # if acce < -3:
        #     acce_output = -3
        # if acce > 3:
        #     acce_output = 3
        return acce_output, LaneChanging
