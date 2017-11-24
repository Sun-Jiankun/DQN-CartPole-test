# -*- coding:utf-8 -*-
import gym
import tensorflow as tf
import numpy as np
import random
from collections import deque

#Hyper Parameters for main
ENV_NAME = 'CartPole-v0'
EPISODE = 10000
STEP = 300
TEST = 10

#Hyper Parameters for DQN
GAMMA = 0.9
INITIAL_EPSILON = 0.5
FINAL_EPSILIN = 0.01
REPLAY_SIZE = 10000
BATCH_SIZE = 32

class DQN():
    #DQN Agent
    def __init__(self,env):
        self.replay_buffer = deque()   # be used to save quintuple (state,one_hot_action,reward,next_state,done)
        self.time_step = 0
        self.epsilon = INITIAL_EPSILON
        self.state_dim = env.observation_space.shape[0]  # the shape of observation ,which in CartPole-v0 is 4
        self.action_dim = env.action_space.n  # the shape of action,which is 2 in CartPole-v0

        # when construct the DQN, construct the tensorflow graph directly
        self.create_Q_network() # create an ANN network graph
        self.create_training_method()

        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())


    # Q = W2*g(W1*x+b1) + b2
    def create_Q_network(self):
        self.hidden_num = 20
        W1 = self.weight_variable([self.state_dim,self.hidden_num])
        b1 = self.bias_variable([self.hidden_num])
        W2 = self.weight_variable([self.hidden_num,self.action_dim])
        b2 = self.bias_variable([self.action_dim])
        # input layer
        self.state_input = tf.placeholder("float",[None,self.state_dim])
        # hidden layers
        h_layer = tf.nn.relu(tf.matmul(self.state_input,W1) + b1)
        # Q Value layer
        self.Q_value = tf.matmul(h_layer,W2) + b2  # None*action_dim

    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape)
        return tf.Variable(initial)

    def bias_variable(self,shape):
        initial = tf.constant(0.01,shape = shape)
        return tf.Variable(initial)

    # what is this method used for?
    def create_training_method(self):
        self.action_input = tf.placeholder("float",[None,self.action_dim])
        self.y_input = tf.placeholder("float",[None])

        #  dot multiply, multiply the corresponding elements between Q_value and action_input
        Q_action = tf.reduce_sum(tf.multiply(self.Q_value,self.action_input),reduction_indices = 1)
        self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
        self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost)

    def perceive(self,state,action,reward,next_state,done):
        one_hot_action = np.zeros(self.action_dim) # save action with one hot way
        one_hot_action[action] = 1
        self.replay_buffer.append((state,one_hot_action,reward,next_state,done))
        if len(self.replay_buffer) > REPLAY_SIZE: # pop the headmost element when the buffer is full
            self.replay_buffer.popleft()

        # question: is it that after the first time it reaches the replayer_buffer,it is a always time to train the Q_network when it moves each step?
        if len(self.replay_buffer) > BATCH_SIZE: # train the Q_network when the batch_size is reached
            self.train_Q_network()

    def train_Q_network(self):
        self.time_step += 1
        minibatch = random.sample(self.replay_buffer,BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]

        y_batch = []
        Q_value_batch = self.Q_value.eval(feed_dict = {self.state_input:next_state_batch})
        for i in range(0,BATCH_SIZE):
            done = minibatch[i][4]
            if done:
                y_batch.append(reward_batch[i])
            else :
                y_batch.append(reward_batch[i] + GAMMA*np.max(Q_value_batch[i]))
        self.optimizer.run(feed_dict={
            self.y_input:y_batch,
            self.action_input:action_batch,
            self.state_input:state_batch
            })

    # output random action depend on different circumstances
    def egreedy_action(self,state):
        Q_value = self.Q_value.eval(feed_dict = {
            self.state_input:[state]
            })[0]
        if random.random() <= self.epsilon:
            return random.randint(0,self.action_dim - 1)
        else:
            return np.argmax(Q_value)
        self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON)/10000

    # output depends on the neural network
    def action(self,state):
        return np.argmax(self.Q_value.eval(feed_dict = {
            self.state_input:[state]
            })[0])


def main():
    # initialize OpenAI Gym env and dqn agent
    env = gym.make(ENV_NAME)   # output : <TimeLimit<CartPoleEnv<CartPole-v0>>>
    agent = DQN(env)  # an agent instance of DQN which has a series of actions

    # train EPOSIDE = 10000 iteratively
    for episode in range(EPISODE):
        # initialize task
        state = env.reset()  # output : an array whose dimension is 1*4
        # Train            # in each eposide train 300 steps
        for step in range(STEP):
            action = agent.egreedy_action(state) # e-greedy action for train
            next_state,reward,done,_ = env.step(action)
            # Define reward for agent
            reward_agent = -1 if done else 0.1
            agent.perceive(state,action,reward,next_state,done)
            state = next_state
            if done:
                break

    # Test every 100 episodes
        if episode % 100 == 0:
            total_reward = 0
            for i in range(TEST):
                state = env.reset()
                for j in range(STEP):
                    env.render()
                    action = agent.action(state) # direct action for test
                    state,reward,done,_ = env.step(action)
                    total_reward += reward
                    if done:
                        break
            ave_reward = total_reward/TEST
            print ('episode:',episode,'Evaluation Average Reward:',ave_reward)
            if ave_reward >= 200:
                break

if __name__ =='__main__':
    main()
