from arguments import get_args
from algo import *
import numpy as np
import time
import gym
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from env import Make_Env
# from gym_minigrid.wrappers import *
import scipy.misc

t = str(time.time())

def plot(record, info, t, suffix):
    plt.figure()
    fig, ax = plt.subplots()
    ax.plot(record['steps'], record['mean'],
            color='blue', label='reward')
    ax.fill_between(record['steps'], record['min'], record['max'],
                    color='blue', alpha=0.2)
    ax.set_xlabel('number of steps')
    ax.set_ylabel('Average score per episode')
    import os
    os.makedirs('-{}'.format(info) + suffix, exist_ok=True)
    fig.savefig('-{}'.format(info) + suffix + '/performance.png')
    plt.close()

    window = 30
    if len(record['mean']) < window:
        return False

    recent_means = record['mean'][-window:]
    max_diff = max(recent_means) - min(recent_means)
    current_mean = np.mean(recent_means)  # 当前窗口内的平均值

    if max_diff < 5 and current_mean > 85:
        # 如果满足收敛条件，返回 True，样本数和时间
        convergence_steps = record['steps'][-1]  # 最近的样本数
        convergence_time = t
        with open('convergence_network_n.txt', 'a') as f:
            # 每次写入一行，包括suffix、样本数和收敛时间
            f.write(f'{suffix} {convergence_steps} {convergence_time}\n')

        print(f'{suffix}: Convergence achieved at step {convergence_steps}, time {convergence_time}')
        return True

    return False

def main(args):

    # load hyperparameters
    args = args
    num_updates = int(args.num_frames // args.num_steps)
    start = time.time()
    record = {'steps': [0],
              'max': [0],
              'mean': [0],
              'min': [0]}

    # environment initial
    envs = Make_Env(env_mode=2)
    action_shape = envs.action_shape
    observation_shape = envs.state_shape
    print(action_shape, observation_shape)
    epsilon = 0.2
    alpha = 0.2
    gamma = 0.99
    # dynamics_model = NetworkModel(8, 8, policy=agent)
    agent = QAgent(epsilon=epsilon, lr=alpha, discount=gamma, action_shape=action_shape)
    dynamics_model = DynaModel()
    n = args.n # 每次模拟随机采样状态次数
    m = args.m # 调用train_transition
    start_planning = args.sp # 从第100次更新开始进行模拟
    h = args.h # 单个状态下模拟步数

    model_type = args.model_type

    if model_type == 'network':
        dynamics_model = NetworkModel(8, 8, policy=agent)
    elif model_type == 'optnetwork':
        dynamics_model = OptNetwork(8, 8, policy=agent)
    elif model_type == 'dyna':
        dynamics_model = DynaModel()
    else:
        print('Unable to identify model type: {}'.format(str(model_type)))
    # start to train your agent
    for i in range(num_updates * 2):
        # 环境交互
        obs = envs.reset()
        obs = obs.astype(int)
        # print("obs: ", obs)
        for step in range(args.num_steps):

            # interact with the environment
            action = agent.select_action(obs)
            obs_next, reward, done, info = envs.step(action)
            # print("obs_next: ", obs_next, "reward: ", reward , "done: ", done)
            obs_next = obs_next.astype(int)
            # add your Q-learning algorithm
            agent.update_q_value(obs, action, reward, obs_next, done)
            dynamics_model.store_transition(obs, action, reward, obs_next)
            obs = obs_next

            if done:
                obs = envs.reset()

        if i > start_planning:
            for _ in range(n): # 采样 n 个状态
                s, idx = dynamics_model.sample_state()
                # buf_tuple = dynamics_model.buffer[idx]
                for _ in range(h): # 从该状态模拟 h 步
                    # print("s: ", s)
                    a = agent.select_action(s)
                    _, s_ = dynamics_model.predict(s, a)
                    r = envs.R(s, a, s_)
                    done = envs.D(s, a, s_)
                    # add your Q-learning algorithm
                    agent.update_q_value(s, a, r, s_, done)

                    s = s_
                    if done:
                        break

        for _ in range(m):
            dynamics_model.train_transition(32)

        recent_mean_rewards = []  # 用于记录最近10次的奖励均值
        if (i + 1) % (args.log_interval) == 0:

            total_num_steps = (i + 1) * args.num_steps
            obs = envs.reset()
            obs = obs.astype(int)
            reward_episode_set = []
            reward_episode = 0.
            for step in range(args.test_steps):
                action = agent.select_action(obs)
                obs_next, reward, done, info = envs.step(action)
                reward_episode += reward
                obs = obs_next
                if done:
                    reward_episode_set.append(reward_episode)
                    reward_episode = 0.
                    obs = envs.reset()

            end = time.time()
            print("TIME {} Updates {}, num timesteps {}, FPS {} \n avrage/min/max reward {:.1f}/{:.1f}/{:.1f}".format(
                    time.strftime("%Hh %Mm %Ss", time.gmtime(end - start)),
                    i, total_num_steps, int(total_num_steps / (end - start)),
                    np.mean(reward_episode_set),
                    np.min(reward_episode_set),
                    np.max(reward_episode_set)))
            record['steps'].append(total_num_steps)
            record['mean'].append(np.mean(reward_episode_set))
            record['max'].append(np.max(reward_episode_set))
            record['min'].append(np.min(reward_episode_set))
            params = {
                '-n=': n,
                '-m=': m,
                '-h=': h,
                '-sp=': start_planning
            }
            suffix = ''
            for k, v in params.items():
                suffix += k + str(v)
            converged = plot(record, args.info ,
                             time.strftime("%Hh %Mm %Ss", time.gmtime(end - start)), suffix)

            if converged:
                break  # 收敛后退出训练

if __name__ == "__main__":
    args = get_args()

    ms = [100]
    ns = [200]
    hs = [2]
    sps = [90]



    for n in ns:
        for m in ms:
            for h in hs:
                for sp in sps:
                    args.n = n
                    args.m = m
                    args.h = h
                    args.sp = sp
                    main(args)