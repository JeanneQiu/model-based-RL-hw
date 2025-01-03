import numpy as np
import random

from abc import abstractmethod
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()

class QAgent:
    def __init__(self, epsilon=0.2, lr=0.2, discount=0.9, action_shape = 4):
        self.action_shape = action_shape
        self.epsilon = epsilon  # 探索率
        self.lr = lr  # 学习率
        self.discount = discount  # 折扣因子
        self.q_table = {}

    def select_action(self, state):
        state = str(state)
        if state not in self.q_table.keys():
            actions = np.zeros(self.action_shape)
            self.q_table[state] = actions
            return np.random.randint(self.action_shape)

        if np.random.rand() < self.epsilon:
            # 随机选择动作（探索）
            return np.random.randint(self.action_shape)

        return np.argmax(self.q_table[state])


    def update_q_value(self, s, a, r, s_, done):
        # 计算下一个状态的最大 Q 值
        s_ = str(s_)
        if s_ not in self.q_table.keys():
            actions = np.zeros(self.action_shape)
            self.q_table[s_] = actions
        max_q_next = np.max(self.q_table[s_])  # 获取下一个状态下的最大 Q 值

        # 使用 Q-learning 更新公式更新当前状态-动作对的 Q 值
        target = r if done else r + self.discount * max_q_next
        self.q_table[str(s)][a] += self.lr * (target - self.q_table[str(s)][a])
        self.q_table[str(s)][a] = np.clip(self.q_table[str(s)][a], -100, 100)

class Model:
    def __init__(self, width, height, policy):
        self.width = width
        self.height = height
        self.policy = policy
        pass

    @abstractmethod
    def store_transition(self, s, a, r, s_):
        pass

    @abstractmethod
    def sample_state(self):
        pass

    @abstractmethod
    def sample_action(self, s):
        pass

    @abstractmethod
    def predict(self, s, a):
        pass


class DynaModel(Model):
    def __init__(self):
        """
        初始化 Dyna-Q 模型。
        :param width: 环境的宽度（状态空间的第一个维度）。
        :param height: 环境的高度（状态空间的第二个维度）。
        :param policy: 强化学习策略对象（如 Q-learning）。
        """
        # self.width = width
        # self.height = height
        # self.policy = policy
        self.transitions = {}  # 用于存储 (s, a) -> (s', r) 的模型

    def store_transition(self, s, a, r, s_):
        """
        存储状态转移信息 (s, a) -> (s', r)。
        :param s: 当前状态。
        :param a: 当前动作。
        :param r: 当前奖励。
        :param s_: 转移后的状态。
        """
        s_key = str(s)  # 状态: 字典键
        if s_key not in self.transitions.keys():
            self.transitions[s_key] = {}
        self.transitions[s_key][a] = (r, s_)  # 存储奖励和下一个状态

    def sample_state(self):
        """
        从已观察到的状态中随机采样一个状态。
        :return: 随机采样的状态。
        """
        result = random.sample(self.transitions.keys(), 1)[0].strip('[').strip(']').split()
        result = np.array(list(map(int, result)))
        return result, None

    def sample_action(self, s):
        result = random.sample(self.transitions[str(s)].keys(), 1)[0]
        return result


    def predict(self, s, a):
        """
        基于模型预测 (s, a) 的转移结果 (r, s')。
        :param s: 当前状态。
        :param a: 当前动作。
        :return: 奖励 r 和转移后的状态 s'。
        """
        s_key = str(s)
        if s_key in self.transitions and a in self.transitions[s_key]:
            return self.transitions[s_key][a]  # 返回模型中存储的 (r, s')
        else:
            return 0, s  # 默认返回零奖励和当前状态

    def train_transition(self, m):

        pass


class NetworkModel(Model):
    def __init__(self, width, height, policy):
        Model.__init__(self, width, height, policy)
        self.x_ph = tf.placeholder(dtype=tf.float32, shape=[None, 3], name='x')
        self.x_next_ph = tf.placeholder(dtype=tf.float32, shape=[None, 3], name='x_next')
        self.a_ph = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='a')
        self.r_ph = tf.placeholder(dtype=tf.float32, shape=[None], name='r')
        # 使用Xavier初始化
        initializer = tf.initializers.glorot_uniform()
        h1 = tf.layers.dense(tf.concat([self.x_ph, self.a_ph], axis=-1), units=256, activation=tf.nn.relu,
                             kernel_initializer=initializer)
        # h1 = tf.layers.dense(tf.concat([self.x_ph, self.a_ph], axis=-1), units=256, activation=tf.nn.relu)
        h1_bn = tf.layers.batch_normalization(h1)  # 添加批量归一化
        h2 = tf.layers.dense(h1_bn, units=256, activation=tf.nn.relu)
        self.next_x = tf.layers.dense(h2, units=3, activation=tf.nn.tanh) * 1.3 + self.x_ph
        self.x_mse = tf.reduce_mean(tf.square(self.next_x - self.x_next_ph))

        # 定义优化器
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-5)
        # 梯度裁剪
        self.grads_and_vars = self.optimizer.compute_gradients(self.x_mse)
        # self.clipped_grads_and_vars = [(tf.clip_by_norm(grad, 30.0), var) for grad, var in self.grads_and_vars]
        self.clipped_grads_and_vars = [
            (tf.clip_by_norm(grad, 1.0) if grad is not None else grad, var)
            for grad, var in self.grads_and_vars
        ]

        self.opt_x = self.optimizer.apply_gradients(self.clipped_grads_and_vars)

        # self.opt_x = tf.train.RMSPropOptimizer(learning_rate=1e-5).minimize(self.x_mse)
        gpu_options = tf.GPUOptions(allow_growth=True)
        tf_config = tf.ConfigProto(gpu_options=gpu_options)
        self.sess = tf.Session(config=tf_config)
        self.sess.run(tf.variables_initializer(tf.global_variables()))
        # 经验池
        self.buffer = []
        self.sensitive_index = []

    def norm_s(self, s):
        return s

    def de_norm_s(self, s):
        s = np.clip(np.round(s), 0, self.width - 1).astype(np.int32)
        s[2] = np.clip(s[2], 0, 1).astype(np.int32)
        return s

    def store_transition(self, s, a, r, s_):
        s = self.norm_s(s)
        s_ = self.norm_s(s_)
        self.buffer.append([s, a, r, s_])
        if s[-1] - s_[-1] != 0:
            self.sensitive_index.append(len(self.buffer) - 1)

    def train_transition(self, batch_size):
        s_list = []
        a_list = []
        r_list = []
        s_next_list = []
        for _ in range(batch_size):
            idx = np.random.randint(0, len(self.buffer))
            s, a, r, s_ = self.buffer[idx]
            s_list.append(s)
            a_list.append([a])
            r_list.append(r)
            s_next_list.append(s_)

        x_mse = self.sess.run([self.x_mse, self.opt_x], feed_dict={
            self.x_ph: s_list, self.a_ph: a_list, self.x_next_ph: s_next_list
        })[:1]

        return x_mse

    def sample_state(self):
        idx = np.random.randint(0, len(self.buffer))
        s, a, r, s_ = self.buffer[idx]
        return self.de_norm_s(s), idx

    def sample_action(self, s):
        return self.policy.select_action(s)

    def predict(self, s, a):
        s_ = self.sess.run(self.next_x, feed_dict={self.x_ph: [s], self.a_ph: [[a]]})
        return None, self.de_norm_s(s_[0])

class OptNetwork(NetworkModel):
    def store_transition(self, s, a, r, s_):
        s = self.norm_s(s)
        s_ = self.norm_s(s_)
        self.buffer.append([s, a, r, s_])
        if s[-1] - s_[-1] != 0:
            self.sensitive_index.append(len(self.buffer) - 1)

    def train_transition(self, batch_size):
        s_list = []
        a_list = []
        r_list = []
        s_next_list = []
        for _ in range(batch_size):
            idx = np.random.randint(0, len(self.buffer))
            s, a, r, s_ = self.buffer[idx]
            s_list.append(s)
            a_list.append([a])
            r_list.append(r)
            s_next_list.append(s_)

        if len(self.sensitive_index) > 0:
            for _ in range(batch_size):
                idx = np.random.randint(0, len(self.sensitive_index))
                idx = self.sensitive_index[idx]
                s, a, r, s_ = self.buffer[idx]
                s_list.append(s)
                a_list.append([a])
                r_list.append(r)
                s_next_list.append(s_)

        x_mse = self.sess.run([self.x_mse, self.opt_x], feed_dict={
            self.x_ph: s_list, self.a_ph: a_list, self.x_next_ph: s_next_list
        })[:1]
        return x_mse

