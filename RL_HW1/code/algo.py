# import numpy as np
#
# from abc import abstractmethod
#
# class QAgent:
# 	def __init__(self,):
# 		pass
#
# 	@abstractmethod
# 	def select_action(self, ob):
# 		pass

import numpy as np
from abc import abstractmethod


class QAgent:
    def __init__(self, state_shape, action_shape, learning_rate=0.1, discount_factor=0.99, epsilon=1.0, epsilon_decay=0.99, min_epsilon=0.1):
        self.state_shape = state_shape
        self.action_shape = action_shape
        # 初始化Q表，初始值为0
        self.q_table = np.zeros(state_shape + (action_shape,))  # 将状态和动作形状组合
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            # 随机选择动作（探索）
            return np.random.randint(self.action_shape)
        else:
            # 根据Q值选择动作（利用）
            return np.argmax(self.q_table[tuple(state)])

    def update(self, state, action, reward, next_state, done):
        old_value = self.q_table[tuple(state) + (action,)]
        next_max = np.max(self.q_table[tuple(next_state)])

        # Q-Learning 更新规则
        new_value = old_value + self.learning_rate * (reward + self.discount_factor * next_max * (not done) - old_value)
        self.q_table[tuple(state) + (action,)] = new_value

        if done:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)  # 减少探索