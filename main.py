"""Run Atari Environment with DQN."""
import argparse
import os
import random
import numpy as np
import tensorflow # TODO: remove this
# always import env (import cv2) first, to avoid opencv magic
from env import Environment
import torch
from dqn import DQNAgent, PredDQNAgent
from policy import GreedyEpsilonPolicy, LinearDecayGreedyEpsilonPolicy
from model import PredDQNetwork, DQNetwork, DuelingQNetwork, PredDuelingQNetwork
from core import ReplayMemory
from logger import Logger


def get_output_folder(parent_dir, env_name):
    """Return save folder.

    Assumes folders in the parent_dir have suffix -run{run
    number}. Finds the highest run number and sets the output folder
    to that number + 1. This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results
    on the same plots with different names.

    Parameters
    ----------
    parent_dir: str
      Path of the directory containing all experiment runs.

    Returns
    -------
    parent_dir/run_dir
      Path to this run's save directory.
    """
    os.makedirs(parent_dir, exist_ok=True)
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1

    child_dir = os.path.join(parent_dir, env_name)
    child_dir = child_dir + '-run{}'.format(experiment_id)
    return child_dir


def main():
    parser = argparse.ArgumentParser(description='Run DQN on Atari')
    parser.add_argument('--env', default='SpaceInvaders-v0', help='Atari env name')
    parser.add_argument('--seed', default=6666999, type=int, help='Random seed')
    parser.add_argument('--lr', default=0.00025, type=float, help='learning rate')
    parser.add_argument('--alpha', default=0.95, type=float,
                        help='squared gradient momentum for RMSprop')
    parser.add_argument('--momentum', default=0.95, type=float,
                        help='gradient momentum for RMSprop')
    parser.add_argument('--rms_eps', default=0.01, type=float,
                        help='min squared gradient for RMS prop')
    parser.add_argument('--q_net', default='', type=str, help='load pretrained q net')
    parser.add_argument('--gamma', default=0.99, type=float, help='discount factor')
    parser.add_argument('--num_iters', default=50000000, type=int)
    parser.add_argument('--replay_buffer_size', default=1000000, type=int)
    parser.add_argument('--num_frames', default=4, type=int, help='nframe, QNet input')
    parser.add_argument('--frame_size', default=84, type=int)
    parser.add_argument('--target_q_sync_interval', default=10000, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--update_freq', default=4, type=int)
    parser.add_argument('--train_start_eps', default=1.0, type=float)
    parser.add_argument('--train_final_eps', default=0.1, type=float)
    parser.add_argument('--train_eps_num_steps', default=1000000, type=int)
    parser.add_argument('--eval_eps', default=0.05, type=float)
    parser.add_argument('--num_burn_in', default=50000, type=int)
    parser.add_argument('--negative_dead_reward', action='store_true',
                        help='whether die in SpaceInvaders-v0 gives a negative reward')
    parser.add_argument('--use_double_dqn', action='store_true')
    parser.add_argument('--output', default='experiments/test')
    parser.add_argument('--algorithm', default='dqn', type=str)

    args = parser.parse_args()
    args.output = '%s_%s' % (args.output, args.algorithm)
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    with open(os.path.join(args.output, 'configs.txt'), 'w') as f:
        print >>f, args

    random.seed(666666)
    np.random.seed(99999)
    torch.manual_seed(77777)
    torch.cuda.manual_seed(555774)
    return args


if __name__ == '__main__':
    args = main()
    torch.backends.cudnn.benckmark = True

    env = Environment(args.env,
                      args.num_frames,
                      args.frame_size,
                      negative_dead_reward=args.negative_dead_reward)
    eval_env = Environment(args.env,
                           args.num_frames,
                           args.frame_size,
                           record=True,
                           video_callable=lambda x:True,
                           mnt_path=os.path.join(args.output, 'monitor'))
    env.seed(888888)
    eval_env.seed(555555)

    replay_memory = ReplayMemory(args.replay_buffer_size)
    train_policy = LinearDecayGreedyEpsilonPolicy(
        args.train_start_eps, args.train_final_eps, args.train_eps_num_steps)
    eval_policy = GreedyEpsilonPolicy(args.eval_eps)
    optim_args = {
        'lr': args.lr,
        'alpha': args.alpha,
        'momentum': args.momentum,
        'eps': args.rms_eps,
        'centered': True
    }

    if 'dueling' == args.algorithm:
        QNClass = DuelingQNetwork
        AgentClass = DQNAgent
    elif 'pdueling' == args.algorithm:
        QNClass = PredDuelingQNetwork
        AgentClass = PredDQNAgent
    elif 'pdqn' == args.algorithm:
        QNClass = PredDQNetwork
        AgentClass = PredDQNAgent
    elif 'dqn' == args.algorithm:
        QNClass = DQNetwork
        AgentClass = DQNAgent
    else:
        assert False, '%s is not implemented yet' % args.algorithm

    q_net = QNClass(args.num_frames,
                    args.frame_size,
                    env.num_actions,
                    optim_args,
                    args.q_net)
    agent = AgentClass(q_net,
                       replay_memory,
                       args.gamma,
                       args.target_q_sync_interval,
                       args.use_double_dqn)
    eval_args = {
        'eval_env': eval_env,
        'eval_per_iter': 100000,
        'eval_policy': eval_policy,
        'num_episodes': 20,
    }
    logger = Logger(os.path.join(args.output, 'train_log.txt'))

    agent.burn_in(env, args.num_burn_in)
    agent.train(
        env, train_policy, args.batch_size, args.num_iters, args.update_freq,
        eval_args, logger, args.output)

    # fianl eval
    eval_log = agent.eval(eval_env, eval_policy, 100)
    # eval_env.reset() # finish the recording for the very last episode (?)
    print logger.log(eval_log)
