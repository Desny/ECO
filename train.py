import os
import sys
import time
import argparse
import ast
import logging
import numpy as np
import paddle.fluid as fluid

from models import TSN
from reader import UCFReader
from config import parse_config, merge_configs, print_configs

logging.root.handlers = []
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(filename='logger.log', level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

# 命令行参数解析工具
def parse_args():
    parser = argparse.ArgumentParser("Paddle Video train script")
    parser.add_argument(
        '--model_name',
        type=str,
        default='tsn',
        help='name of model to train.')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='training batch size. None to use config file setting.')
    parser.add_argument(
        '--sample_frames',
        type=int,
        default=32,
        help='num of sample frames'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=None,
        help='learning rate use for training. None to use config file setting.')
    parser.add_argument(
        '--pretrain',
        type=str,
        default=None,
        help='path to pretrain weights. None to use default weights path in  ~/.paddle/weights.'
    )
    parser.add_argument(
        '--use_gpu',
        type=ast.literal_eval,
        default=False,
        help='default use gpu.')
    parser.add_argument(
        '--epoch',
        type=int,
        default=100,
        help='epoch number, 0 for read from config file')
    parser.add_argument(
        '--save_dir',
        type=str,
        default='checkpoints_models',
        help='directory name to save train snapshoot')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/tsn.txt',
        help='path to config file of model')
    args = parser.parse_args()
    return args


def train(args):
    # parse config
    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    with fluid.dygraph.guard(place):
        config = parse_config(args.config)
        train_config = merge_configs(config, 'train', vars(args))
        #根据自己定义的网络，声明train_model
        train_model = TSN(args.batch_size, 32)
        train_model.train()
        opt = fluid.optimizer.Momentum(0.001, 0.9, parameter_list=train_model.parameters())

        if args.pretrain:
            # 加载上一次训练的模型，继续训练
            model, _ = fluid.dygraph.load_dygraph(args.save_dir + '/tsn_model')
            train_model.load_dict(model)

        # build model
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        # 迭代器，每次输送一个批次的数据，每个数据的形式(imgs, label)
        train_reader = UCFReader('train', train_config).create_reader()

        epochs = args.epoch
        for i in range(epochs):
            for batch_id, data in enumerate(train_reader()):
                dy_x_data = np.array([x[0] for x in data]).astype('float32')    # dy_x_data=imgs  shape:(batch_size, seg_num*seglen, 3, target_size, target_size)
                y_data = np.array([x[1] for x in data]).astype('int64')    # y_data=label  shape:(batch_size,)

                img = fluid.dygraph.to_variable(dy_x_data)
                label = fluid.dygraph.to_variable(y_data)
                label = fluid.layers.reshape(label, [args.batch_size, 1])
                label.stop_gradient = True  # 反向求导时，只有目标函数中的label部分对label求导（而不是label与img融合求导）
                
                out, acc = train_model(img, label)
                
                loss = fluid.layers.cross_entropy(out, label)
                avg_loss = fluid.layers.mean(loss)

                avg_loss.backward()

                opt.minimize(avg_loss)
                train_model.clear_gradients()
                
                
                if batch_id % 100 == 0:
                    print("Loss at epoch {} step {}: {}, acc: {}".format(i, batch_id, avg_loss.numpy(), acc.numpy()))
                    fluid.dygraph.save_dygraph(train_model.state_dict(), args.save_dir + '/tsn_model')
        print("Final loss: {}".format(avg_loss.numpy()))
                


if __name__ == "__main__":
    args = parse_args()

    train(args)
