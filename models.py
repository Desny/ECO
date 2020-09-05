import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Conv2D, Conv3D, Pool2D, BatchNorm, Linear, Dropout
from paddle.fluid.layers import pool3d
import numpy as np


class Conv2DBNLayer(fluid.dygraph.Layer):
    def __init__(self, num_channels, num_filters, filter_size, stride=1, padding=0, act=None):
        super(Conv2DBNLayer, self).__init__()
        self._conv = Conv2D(num_channels=num_channels,
                            num_filters=num_filters,
                            filter_size=filter_size,
                            stride=stride,
                            padding=padding,
                            act=act)

        self._batch_norm = BatchNorm(num_filters, act=act)

    def forward(self, inputs):
        y = self._conv(inputs)
        y = self._batch_norm(y)
        return y


class Conv3DBNLayer(fluid.dygraph.Layer):
    def __init__(self, num_channels, num_filters, filter_size, stride=1, padding=0, act=None):
        super(Conv3DBNLayer, self).__init__()
        self._conv = Conv3D(num_channels=num_channels,
                            num_filters=num_filters,
                            filter_size=filter_size,
                            stride=stride,
                            padding=padding,
                            act=act)

        self._batch_norm = BatchNorm(num_filters, act=act)

    def forward(self, inputs):
        y = self._conv(inputs)
        y = self._batch_norm(y)
        return y


# in:192  out:256  size:28*28(不变)
class Inception3aLayer(fluid.dygraph.Layer):
    def __init__(self):
        super(Inception3aLayer, self).__init__()
        self.inception_3a_1x1 = Conv2DBNLayer(num_channels=192, num_filters=64, filter_size=1, act='relu')

        self.inception_3a_3x3_reduce = Conv2DBNLayer(num_channels=192, num_filters=64, filter_size=1, act='relu')
        self.inception_3a_3x3 = Conv2DBNLayer(num_channels=64, num_filters=64, filter_size=3, padding=1, act='relu')

        self.inception_3a_double_3x3_reduce = Conv2DBNLayer(num_channels=192, num_filters=64, filter_size=1, act='relu')
        self.inception_3a_double_3x3_1 = Conv2DBNLayer(num_channels=64, num_filters=96, filter_size=3,
                                                     padding=1, act='relu')
        self.inception_3a_double_3x3_2 = Conv2DBNLayer(num_channels=96, num_filters=96, filter_size=3,
                                                     padding=1, act='relu')

        self.inception_3a_pool = Pool2D(pool_size=3, pool_type='avg', pool_padding=1)
        self.inception_3a_pool_proj = Conv2DBNLayer(num_channels=192, num_filters=32, filter_size=1, act='relu')


    def forward(self, inputs):
        inception_3a_1x1_bn = self.inception_3a_1x1(inputs)
        inception_3a_1x1_bn = inception_3a_1x1_bn.numpy()

        y = self.inception_3a_3x3_reduce(inputs)
        inception_3a_3x3_bn = self.inception_3a_3x3(y)
        inception_3a_3x3_bn = inception_3a_3x3_bn.numpy()

        y = self.inception_3a_double_3x3_reduce(inputs)
        y = self.inception_3a_double_3x3_1(y)
        inception_3a_double_3x3_2_bn = self.inception_3a_double_3x3_2(y)
        inception_3a_double_3x3_2_bn = inception_3a_double_3x3_2_bn.numpy()

        y = self.inception_3a_pool(inputs)
        inception_3a_pool_proj_bn = self.inception_3a_pool_proj(y)
        inception_3a_pool_proj_bn = inception_3a_pool_proj_bn.numpy()


        # inception_3a_output  <=Concat<=
        # inception_3a_1x1_bn,inception_3a_3x3_bn,inception_3a_double_3x3_2_bn,inception_3a_pool_proj_bn
        inception_3a_output = np.concatenate((inception_3a_1x1_bn, inception_3a_3x3_bn,
                                                   inception_3a_double_3x3_2_bn, inception_3a_pool_proj_bn), axis=1)

        return inception_3a_output


class Inception3bLayer(fluid.dygraph.Layer):
    def __init__(self):
        super(Inception3bLayer, self).__init__()
        self.inception_3b_1x1 = Conv2DBNLayer(num_channels=256, num_filters=64, filter_size=1, act='relu')

        self.inception_3b_3x3_reduce = Conv2DBNLayer(num_channels=256, num_filters=64, filter_size=1, act='relu')
        self.inception_3b_3x3 = Conv2DBNLayer(num_channels=64, num_filters=96, filter_size=3, padding=1, act='relu')

        self.inception_3b_double_3x3_reduce = Conv2DBNLayer(num_channels=256, num_filters=64, filter_size=1, act='relu')
        self.inception_3b_double_3x3_1 = Conv2DBNLayer(num_channels=64, num_filters=96, filter_size=3,
                                                     padding=1, act='relu')
        self.inception_3b_double_3x3_2 = Conv2DBNLayer(num_channels=96, num_filters=96, filter_size=3,
                                                     padding=1, act='relu')

        self.inception_3b_pool = Pool2D(pool_size=3, pool_type='avg', pool_padding=1)
        self.inception_3b_pool_proj = Conv2DBNLayer(num_channels=256, num_filters=64, filter_size=1, act='relu')


    def forward(self, inputs):
        inputs = fluid.dygraph.to_variable(inputs)
        inception_3b_1x1_bn = self.inception_3b_1x1(inputs)
        inception_3b_1x1_bn = inception_3b_1x1_bn.numpy()

        y = self.inception_3b_3x3_reduce(inputs)
        inception_3b_3x3_bn = self.inception_3b_3x3(y)
        inception_3b_3x3_bn = inception_3b_3x3_bn.numpy()

        y = self.inception_3b_double_3x3_reduce(inputs)
        y = self.inception_3b_double_3x3_1(y)
        inception_3b_double_3x3_2_bn = self.inception_3b_double_3x3_2(y)
        inception_3b_double_3x3_2_bn = inception_3b_double_3x3_2_bn.numpy()

        y = self.inception_3b_pool(inputs)
        inception_3b_pool_proj_bn = self.inception_3b_pool_proj(y)
        inception_3b_pool_proj_bn = inception_3b_pool_proj_bn.numpy()

        # inception_3b_output  <=Concat<=
        # inception_3b_1x1_bn,inception_3b_3x3_bn,inception_3b_double_3x3_2_bn,inception_3b_pool_proj_bn
        inception_3b_output = np.concatenate((inception_3b_1x1_bn, inception_3b_3x3_bn,
                                              inception_3b_double_3x3_2_bn, inception_3b_pool_proj_bn), axis=1)

        return inception_3b_output


class Inception3cForC3D(fluid.dygraph.Layer):
    def __init__(self):
        super(Inception3cForC3D, self).__init__()
        self.inception_3c_double_3x3_reduce = Conv2DBNLayer(num_channels=320, num_filters=64, filter_size=1, act='relu')
        self.inception_3c_double_3x3_1 = Conv2DBNLayer(num_channels=64, num_filters=96, filter_size=3,
                                                     padding=1, act='relu')

    def forward(self, inputs):
        y = self.inception_3c_double_3x3_reduce(inputs)
        inception_3c_double_3x3_1 = self.inception_3c_double_3x3_1(y)

        return inception_3c_double_3x3_1


class C3D(fluid.dygraph.Layer):
    def __init__(self):
        super(C3D, self).__init__()
        self.res3a_2 = Conv3DBNLayer(num_channels=96, num_filters=128, padding=1, filter_size=3, act='relu')
        self.res3b_1 = Conv3DBNLayer(num_channels=128, num_filters=128, padding=1, filter_size=3, act='relu')
        self.res3b_2 = Conv3DBNLayer(num_channels=128, num_filters=128, padding=1, filter_size=3, act='relu')
        # res3b<=Eltwise<=res3b_2,res3a_2
        self.res3b_bn = BatchNorm(num_channels=128, act='relu')

        self.res4a_1 = Conv3DBNLayer(num_channels=128, num_filters=256, padding=1, filter_size=3, stride=2, act='relu')
        self.res4a_2 = Conv3D(num_channels=256, num_filters=256, padding=1, filter_size=3)
        # res4a_down<=Conv3d<=res3b_bn
        self.res4a_down = Conv3D(num_channels=128, num_filters=256, padding=1, filter_size=3, stride=2)
        # res4a<=Eltwise<=res4a_2,res4a_down
        self.res4a = BatchNorm(num_channels=256, act='relu')

        self.res4b_1 = Conv3DBNLayer(num_channels=256, num_filters=256, padding=1, filter_size=3, act='relu')
        self.res4b_2 = Conv3DBNLayer(num_channels=256, num_filters=256, padding=1, filter_size=3, act='relu')
        # res4b<=Eltwise<=res4b_2,res4a
        self.res4b_bn = BatchNorm(num_channels=256, act='relu')

        self.res5a_1 = Conv3DBNLayer(num_channels=256, num_filters=512, padding=1, stride=2, filter_size=3, act='relu')
        self.res5a_2 = Conv3D(num_channels=512, num_filters=512, padding=1, filter_size=3, act='relu')
        # res5a_down<=Conv3d<=res4b_bn
        self.res5a_down = Conv3D(num_channels=256, num_filters=512, padding=1, stride=2, filter_size=3)
        # res5a<=Eltwise<=res5a_2,res5a_down
        self.res5a = BatchNorm(num_channels=512, act='relu')

        self.res5b_1 = Conv3DBNLayer(num_channels=512, num_filters=512, padding=1, filter_size=3, act='relu')
        self.res5b_2 = Conv3D(num_channels=512, num_filters=512, padding=1, filter_size=3)
        # res5b<=Eltwise<=res5b_2,res5a
        self.res5b = BatchNorm(num_channels=512, act='relu')


    def forward(self, inputs):
        res3a_2 = self.res3a_2(inputs)
        y = self.res3b_1(res3a_2)
        res3b_2 = self.res3b_2(y)
        # res3b<=Eltwise<=res3b_2,res3a_2
        res3b = fluid.layers.elementwise_add(x=res3a_2, y=res3b_2)
        res3b_bn = self.res3b_bn(res3b)

        y = self.res4a_1(res3b_bn)
        res4a_2 = self.res4a_2(y)
        res4a_down = self.res4a_down(res3b_bn)
        # res4a<=Eltwise<=res4a_2,res4a_down
        res4a = fluid.layers.elementwise_add(x=res4a_down, y=res4a_2)
        res4a = self.res4a(res4a)

        y = self.res4b_1(res4a)
        res4b_2 = self.res4b_2(y)
        # res4b<=Eltwise<=res4b_2,res4a
        res4b = fluid.layers.elementwise_add(x=res4a, y=res4b_2)
        res4b_bn = self.res4b_bn(res4b)

        y = self.res5a_1(res4b_bn)
        res5a_2 = self.res5a_2(y)
        res5a_down = self.res5a_down(res4b_bn)
        # res5a<=Eltwise<=res5a_2,res5a_down
        res5a = fluid.layers.elementwise_add(x=res5a_down, y=res5a_2)
        res5a = self.res5a(res5a)

        y = self.res5b_1(res5a)
        res5b_2 = self.res5b_2(y)
        # res5b<=Eltwise<=res5b_2,res5a
        res5b = fluid.layers.elementwise_add(x=res5a, y=res5b_2)
        res5b = self.res5b(res5b)

        return res5b


class Conv2DNets(fluid.dygraph.Layer):
    def __init__(self, batch_size, sample):
        super(Conv2DNets, self).__init__()

        self.batch_size = batch_size
        self.sample = sample

        self.inception_4a_1x1 = Conv2DBNLayer(num_channels=576, num_filters=224, filter_size=1, act='relu')

        self.inception_4a_3x3_reduce = Conv2DBNLayer(num_channels=576, num_filters=64, filter_size=1, act='relu')
        self.inception_4a_3x3 = Conv2DBNLayer(num_channels=64, num_filters=96, padding=1, filter_size=3, act='relu')

        self.inception_4a_double_3x3_reduce = Conv2DBNLayer(num_channels=576, num_filters=96, filter_size=1, act='relu')
        self.inception_4a_double_3x3_1 = Conv2DBNLayer(num_channels=96, num_filters=128, filter_size=3,
                                                       padding=1, act='relu')
        self.inception_4a_double_3x3_2 = Conv2DBNLayer(num_channels=128, num_filters=128, filter_size=3, padding=1,
                                                       act='relu')

        self.inception_4a_pool = Pool2D(pool_size=3, pool_type='avg', pool_padding=1)
        self.inception_4a_pool_proj = Conv2DBNLayer(num_channels=576, num_filters=128, filter_size=1, act='relu')
        # inception_4a_output  <=Concat<=
        # inception_4a_1x1_bn(14*14 224),
        # inception_4a_3x3_bn(14*14 96),
        # inception_4a_double_3x3_2_bn(14*14 128),
        # inception_4a_pool_proj_bn(14*14 128)

        self.inception_4b_1x1 = Conv2DBNLayer(num_channels=576, num_filters=192, filter_size=1, act='relu')

        self.inception_4b_3x3_reduce = Conv2DBNLayer(num_channels=576, num_filters=96, filter_size=1, act='relu')
        self.inception_4b_3x3 = Conv2DBNLayer(num_channels=96, num_filters=128, filter_size=3,
                                              padding=1, act='relu')

        self.inception_4b_double_3x3_reduce = Conv2DBNLayer(num_channels=576, num_filters=96, filter_size=1, act='relu')
        self.inception_4b_double_3x3_1 = Conv2DBNLayer(num_channels=96, num_filters=128, filter_size=3,
                                                       padding=1, act='relu')
        self.inception_4b_double_3x3_2 = Conv2DBNLayer(num_channels=128, num_filters=128, filter_size=3,
                                                       padding=1, act='relu')

        self.inception_4b_pool = Pool2D(pool_size=3, pool_type='avg', pool_padding=1)
        self.inception_4b_pool_proj = Conv2DBNLayer(num_channels=576, num_filters=128, filter_size=1, act='relu')
        # inception_4b_output  <=Concat<=
        # inception_4b_1x1_bn(14*14 192),
        # inception_4b_3x3_bn(14*14 128),
        # inception_4b_double_3x3_2_bn(14*14 128),
        # inception_4b_pool_proj_bn(14*14 128)

        self.inception_4c_1x1 = Conv2DBNLayer(num_channels=576, num_filters=160, filter_size=1, act='relu')

        self.inception_4c_3x3_reduce = Conv2DBNLayer(num_channels=576, num_filters=128, filter_size=1, act='relu')
        self.inception_4c_3x3 = Conv2DBNLayer(num_channels=128, num_filters=160, filter_size=3, padding=1, act='relu')

        self.inception_4c_double_3x3_reduce = Conv2DBNLayer(num_channels=576, num_filters=128,
                                                            filter_size=1, act='relu')
        self.inception_4c_double_3x3_1 = Conv2DBNLayer(num_channels=128, num_filters=160, filter_size=3,
                                                       padding=1, act='relu')
        self.inception_4c_double_3x3_2 = Conv2DBNLayer(num_channels=160, num_filters=160, filter_size=3,
                                                       padding=1, act='relu')

        self.inception_4c_pool = Pool2D(pool_size=3, pool_type='avg', pool_padding=1)
        self.inception_4c_pool_proj = Conv2DBNLayer(num_channels=576, num_filters=96, filter_size=1, act='relu')
        #inception_4c_output  <=Concat<=
        # inception_4c_1x1_bn(14*14 160),
        # inception_4c_3x3_bn(14*14 160),
        # inception_4c_double_3x3_2_bn(14*14 160),
        # inception_4c_pool_proj_bn(14*14 96)

        self.inception_4d_1x1 = Conv2DBNLayer(num_channels=576, num_filters=96, filter_size=1, act='relu')

        self.inception_4d_3x3_reduce = Conv2DBNLayer(num_channels=576, num_filters=128, filter_size=1, act='relu')
        self.inception_4d_3x3 = Conv2DBNLayer(num_channels=128, num_filters=192, filter_size=3, padding=1, act='relu')

        self.inception_4d_double_3x3_reduce = Conv2DBNLayer(num_channels=576, num_filters=160,
                                                            filter_size=1, act='relu')
        self.inception_4d_double_3x3_1 = Conv2DBNLayer(num_channels=160, num_filters=192, filter_size=3,
                                                       padding=1, act='relu')
        self.inception_4d_double_3x3_2 = Conv2DBNLayer(num_channels=192, num_filters=192, filter_size=3,
                                                       padding=1, act='relu')

        self.inception_4d_pool = Pool2D(pool_size=3, pool_type='avg', pool_padding=1)
        self.inception_4d_pool_proj = Conv2DBNLayer(num_channels=576, num_filters=96, filter_size=1, act='relu')
        # inception_4d_output  <=Concat<=
        # inception_4d_1x1_bn(14*14 96),
        # inception_4d_3x3_bn(14*14 192),
        # inception_4d_double_3x3_2_bn(14*14 192),
        # inception_4d_pool_proj_bn(14*14 96)

        self.inception_4e_3x3_reduce = Conv2DBNLayer(num_channels=576, num_filters=128, filter_size=1, act='relu')
        self.inception_4e_3x3 = Conv2DBNLayer(num_channels=128, num_filters=192, filter_size=3,
                                              padding=1, stride=2, act='relu')  # 7*7

        self.inception_4e_double_3x3_reduce = Conv2DBNLayer(num_channels=576, num_filters=128,
                                                            filter_size=1, act='relu')
        self.inception_4e_double_3x3_1 = Conv2DBNLayer(num_channels=128, num_filters=256, filter_size=3,
                                                       padding=1, act='relu')
        self.inception_4e_double_3x3_2 = Conv2DBNLayer(num_channels=256, num_filters=256, filter_size=3,
                                                       padding=1, stride=2, act='relu')  # 7*7

        self.inception_4e_pool = Pool2D(pool_size=3, pool_type='max', pool_stride=2, pool_padding=1)  # 7*7
        # inception_4e_output  <=Concat<=
        # inception_4e_3x3_bn(7*7 192),
        # inception_4e_double_3x3_2_bn(7*7 256),
        # inception_4e_pool(7*7 576)

        self.inception_5a_1x1 = Conv2DBNLayer(num_channels=1024, num_filters=352, filter_size=1, act='relu')

        self.inception_5a_3x3_reduce = Conv2DBNLayer(num_channels=1024, num_filters=192, filter_size=1, act='relu')
        self.inception_5a_3x3 = Conv2DBNLayer(num_channels=192, num_filters=320, filter_size=3, padding=1, act='relu')

        self.inception_5a_double_3x3_reduce = Conv2DBNLayer(num_channels=1024, num_filters=160,
                                                            filter_size=1, act='relu')
        self.inception_5a_double_3x3_1 = Conv2DBNLayer(num_channels=160, num_filters=224, filter_size=3,
                                                       padding=1, act='relu')
        self.inception_5a_double_3x3_2 = Conv2DBNLayer(num_channels=224, num_filters=224, filter_size=3,
                                                       padding=1, act='relu')

        self.inception_5a_pool = Pool2D(pool_size=3, pool_type='avg', pool_padding=1)
        self.inception_5a_pool_proj = Conv2DBNLayer(num_channels=1024, num_filters=128, filter_size=1, act='relu')
        # inception_5a_output  <=Concat<=
        # inception_5a_1x1_bn(7*7 352),
        # inception_5a_3x3_bn(7*7 320),
        # inception_5a_double_3x3_2_bn(7*7 224),
        # inception_5a_pool_proj_bn(7*7 128)

        self.inception_5b_1x1 = Conv2DBNLayer(num_channels=1024, num_filters=352, filter_size=1, act='relu')

        self.inception_5b_3x3_reduce = Conv2DBNLayer(num_channels=1024, num_filters=192, filter_size=1, act='relu')
        self.inception_5b_3x3 = Conv2DBNLayer(num_channels=192, num_filters=320, filter_size=3, padding=1, act='relu')

        self.inception_5b_double_3x3_reduce = Conv2DBNLayer(num_channels=1024, num_filters=192,
                                                            filter_size=1, act='relu')
        self.inception_5b_double_3x3_1 = Conv2DBNLayer(num_channels=192, num_filters=224, filter_size=3,
                                                       padding=1, act='relu')
        self.inception_5b_double_3x3_2 = Conv2DBNLayer(num_channels=224, num_filters=224, filter_size=3,
                                                       padding=1, act='relu')

        self.inception_5b_pool = Pool2D(pool_size=3, pool_type='max', pool_padding=1)
        self.inception_5b_pool_proj = Conv2DBNLayer(num_channels=1024, num_filters=128, filter_size=1, act='relu')
        # inception_5b_output  <=Concat<=
        # inception_5b_1x1_bn(7*7 352),
        # inception_5b_3x3_bn(7*7 320),
        # inception_5b_double_3x3_2_bn(7*7 224),
        # inception_5b_pool_proj_bn(7*7 128)

        self.global_pool2D_pre = Pool2D(pool_size=7, pool_type='avg')  # 1*1 1024
        self.global_pool2D_pre_drop = Dropout(p=0.5)
        # global_pool2D_reshape_consensus<=Pooling3d<=global_pool2D_pre

        # global_pool3D<=Pooling3d<=res5b_bn
        self.global_pool3D_drop = Dropout(p=0.3)
        # global_pool<=Concat<=global_pool2D_reshape_consensus,global_pool3D
        # fc_action<=InnerProduct<=global_pool


    def forward(self, inputs):  # inputs = inception_3c_output(混合连接成的)
        inception_4a_1x1 = self.inception_4a_1x1(inputs)

        y = self.inception_4a_3x3_reduce(inputs)
        inception_4a_3x3 = self.inception_4a_3x3(y)

        y = self.inception_4a_double_3x3_reduce(inputs)
        y = self.inception_4a_double_3x3_1(y)
        inception_4a_double_3x3_2 = self.inception_4a_double_3x3_2(y)

        y = self.inception_4a_pool(inputs)
        inception_4a_pool_proj = self.inception_4a_pool_proj(y)
        # inception_4a_output  <=Concat<=
        # inception_4a_1x1_bn,inception_4a_3x3_bn,inception_4a_double_3x3_2_bn,inception_4a_pool_proj_bn
        inception_4a_1x1 = inception_4a_1x1.numpy()
        inception_4a_3x3 = inception_4a_3x3.numpy()
        inception_4a_double_3x3_2 = inception_4a_double_3x3_2.numpy()
        inception_4a_pool_proj = inception_4a_pool_proj.numpy()
        inception_4a_output = np.concatenate((inception_4a_1x1, inception_4a_3x3,
                                              inception_4a_double_3x3_2, inception_4a_pool_proj), axis=1)
        inception_4a_output = fluid.dygraph.to_variable(inception_4a_output)

        inception_4b_1x1 = self.inception_4b_1x1(inception_4a_output)

        y = self.inception_4b_3x3_reduce(inception_4a_output)
        inception_4b_3x3 = self.inception_4b_3x3(y)

        y = self.inception_4b_double_3x3_reduce(inception_4a_output)
        y = self.inception_4b_double_3x3_1(y)
        inception_4b_double_3x3_2 = self.inception_4b_double_3x3_2(y)

        y = self.inception_4b_pool(inception_4a_output)
        inception_4b_pool_proj = self.inception_4b_pool_proj(y)
        # inception_4b_output  <=Concat<=
        # inception_4b_1x1_bn,inception_4b_3x3_bn,inception_4b_double_3x3_2_bn,inception_4b_pool_proj_bn
        inception_4b_1x1 = inception_4b_1x1.numpy()
        inception_4b_3x3 = inception_4b_3x3.numpy()
        inception_4b_double_3x3_2 = inception_4b_double_3x3_2.numpy()
        inception_4b_pool_proj = inception_4b_pool_proj.numpy()
        inception_4b_output = np.concatenate((inception_4b_1x1, inception_4b_3x3,
                                              inception_4b_double_3x3_2, inception_4b_pool_proj), axis=1)
        inception_4b_output = fluid.dygraph.to_variable(inception_4b_output)

        inception_4c_1x1 = self.inception_4c_1x1(inception_4b_output)

        y = self.inception_4c_3x3_reduce(inception_4b_output)
        inception_4c_3x3 = self.inception_4c_3x3(y)

        y = self.inception_4c_double_3x3_reduce(inception_4b_output)
        y = self.inception_4c_double_3x3_1(y)
        inception_4c_double_3x3_2 = self.inception_4c_double_3x3_2(y)

        y = self.inception_4c_pool(inception_4b_output)
        inception_4c_pool_proj = self.inception_4c_pool_proj(y)
        # inception_4c_output  <=Concat<=
        # inception_4c_1x1_bn,inception_4c_3x3_bn,inception_4c_double_3x3_2_bn,inception_4c_pool_proj_bn
        inception_4c_1x1 = inception_4c_1x1.numpy()
        inception_4c_3x3 = inception_4c_3x3.numpy()
        inception_4c_double_3x3_2 = inception_4c_double_3x3_2.numpy()
        inception_4c_pool_proj = inception_4c_pool_proj.numpy()
        inception_4c_output = np.concatenate((inception_4c_1x1, inception_4c_3x3, inception_4c_double_3x3_2,
                                              inception_4c_pool_proj), axis=1)
        inception_4c_output = fluid.dygraph.to_variable(inception_4c_output)

        inception_4d_1x1 = self.inception_4d_1x1(inception_4c_output)

        y = self.inception_4d_3x3_reduce(inception_4c_output)
        inception_4d_3x3 = self.inception_4d_3x3(y)

        y = self.inception_4d_double_3x3_reduce(inception_4c_output)
        y = self.inception_4d_double_3x3_1(y)
        inception_4d_double_3x3_2 = self.inception_4d_double_3x3_2(y)

        y = self.inception_4d_pool(inception_4c_output)
        inception_4d_pool_proj = self.inception_4d_pool_proj(y)
        # inception_4d_output  <=Concat<=
        # inception_4d_1x1_bn,inception_4d_3x3_bn,inception_4d_double_3x3_2_bn,inception_4d_pool_proj_bn
        inception_4d_1x1 = inception_4d_1x1.numpy()
        inception_4d_3x3 = inception_4d_3x3.numpy()
        inception_4d_double_3x3_2 = inception_4d_double_3x3_2.numpy()
        inception_4d_pool_proj = inception_4d_pool_proj.numpy()
        inception_4d_output = np.concatenate((inception_4d_1x1, inception_4d_3x3,
                                              inception_4d_double_3x3_2, inception_4d_pool_proj), axis=1)
        inception_4d_output = fluid.dygraph.to_variable(inception_4d_output)

        y = self.inception_4e_3x3_reduce(inception_4d_output)
        inception_4e_3x3 = self.inception_4e_3x3(y)

        y = self.inception_4e_double_3x3_reduce(inception_4d_output)
        y = self.inception_4e_double_3x3_1(y)
        inception_4e_double_3x3_2 = self.inception_4e_double_3x3_2(y)

        inception_4e_pool = self.inception_4e_pool(inception_4d_output)
        # inception_4e_output<=Concat<=inception_4e_3x3_bn,inception_4e_double_3x3_2_bn,inception_4e_pool
        inception_4e_3x3 = inception_4e_3x3.numpy()
        inception_4e_double_3x3_2 = inception_4e_double_3x3_2.numpy()
        inception_4e_pool = inception_4e_pool.numpy()
        inception_4e_output = np.concatenate((inception_4e_3x3, inception_4e_double_3x3_2, inception_4e_pool), axis=1)
        inception_4e_output = fluid.dygraph.to_variable(inception_4e_output)

        inception_5a_1x1 = self.inception_5a_1x1(inception_4e_output)

        y = self.inception_5a_3x3_reduce(inception_4e_output)
        inception_5a_3x3 = self.inception_5a_3x3(y)

        y = self.inception_5a_double_3x3_reduce(inception_4e_output)
        y = self.inception_5a_double_3x3_1(y)
        inception_5a_double_3x3_2 = self.inception_5a_double_3x3_2(y)

        y = self.inception_5a_pool(inception_4e_output)
        inception_5a_pool_proj = self.inception_5a_pool_proj(y)
        # inception_5a_output  <=Concat<=
        # inception_5a_1x1_bn,inception_5a_3x3_bn,inception_5a_double_3x3_2_bn,inception_5a_pool_proj_bn
        inception_5a_1x1 = inception_5a_1x1.numpy()
        inception_5a_3x3 = inception_5a_3x3.numpy()
        inception_5a_double_3x3_2 = inception_5a_double_3x3_2.numpy()
        inception_5a_pool_proj = inception_5a_pool_proj.numpy()
        inception_5a_output = np.concatenate((inception_5a_1x1, inception_5a_3x3,
                                              inception_5a_double_3x3_2, inception_5a_pool_proj), axis=1)
        inception_5a_output = fluid.dygraph.to_variable(inception_5a_output)

        inception_5b_1x1 = self.inception_5b_1x1(inception_5a_output)

        y = self.inception_5b_3x3_reduce(inception_5a_output)
        inception_5b_3x3 = self.inception_5b_3x3(y)

        y = self.inception_5b_double_3x3_reduce(inception_5a_output)
        y = self.inception_5b_double_3x3_1(y)
        inception_5b_double_3x3_2 = self.inception_5b_double_3x3_2(y)

        y = self.inception_5b_pool(inception_5a_output)
        inception_5b_pool_proj = self.inception_5b_pool_proj(y)
        # inception_5b_output  <=Concat<=
        # inception_5b_1x1_bn,inception_5b_3x3_bn,inception_5b_double_3x3_2_bn,inception_5b_pool_proj_bn
        inception_5b_1x1 = inception_5b_1x1.numpy()
        inception_5b_3x3 = inception_5b_3x3.numpy()
        inception_5b_double_3x3_2 = inception_5b_double_3x3_2.numpy()
        inception_5b_pool_proj = inception_5b_pool_proj.numpy()
        inception_5b_output = np.concatenate((inception_5b_1x1, inception_5b_3x3, inception_5b_double_3x3_2,
                                              inception_5b_pool_proj), axis=1)
        inception_5b_output = fluid.dygraph.to_variable(inception_5b_output)

        global_pool2D_pre = self.global_pool2D_pre(inception_5b_output)
        global_pool2D_pre_drop = self.global_pool2D_pre_drop(global_pool2D_pre)

        global_pool2D_pre_drop = global_pool2D_pre_drop.numpy()
        global_pool2D_pre_drop = global_pool2D_pre_drop[np.newaxis, :]
        global_pool2D_pre_drop = fluid.dygraph.to_variable(global_pool2D_pre_drop)
        global_pool2D_pre_drop = fluid.layers.reshape(global_pool2D_pre_drop, [self.batch_size, self.sample,
                 global_pool2D_pre_drop.shape[2], global_pool2D_pre_drop.shape[3], global_pool2D_pre_drop.shape[4]])
        global_pool2D_pre_drop = fluid.layers.transpose(global_pool2D_pre_drop, [0, 2, 1, 3, 4])
        
        global_pool2D_reshape_consensus = pool3d(global_pool2D_pre_drop, pool_type='avg', pool_size=(self.sample, 1, 1))

        return global_pool2D_reshape_consensus


class TSN(fluid.dygraph.Layer):
    def __init__(self, batch_size, sample):
        super(TSN, self).__init__()

        self.batch_size = batch_size
        self.sample = sample

        self.conv1 = Conv2DBNLayer(num_channels=3, num_filters=64, filter_size=7, stride=2, padding=3, act='relu')
        self.pool1 = Pool2D(pool_size=3, pool_type='max', pool_stride=2, pool_padding=1)
        self.conv2_reduce = Conv2DBNLayer(num_channels=64, num_filters=64, filter_size=1, act='relu')
        self.conv2 = Conv2DBNLayer(num_channels=64, num_filters=192, filter_size=3, padding=1, act='relu')
        self.pool2 = Pool2D(pool_size=3, pool_type='max', pool_stride=2, pool_padding=1)

        self.inception_3a = Inception3aLayer()
        self.inception_3b = Inception3bLayer()
        self.inception_3c_c3d = Inception3cForC3D()

        self.c3d = C3D()

        self.inception_3c_3x3_reduce = Conv2DBNLayer(num_channels=320, num_filters=128, filter_size=1, act='relu')
        self.inception_3c_3x3 = Conv2DBNLayer(num_channels=128, num_filters=160, filter_size=3, 
                                                padding=1, stride=2, act='relu')

        self.inception_3c_double_3x3_2 = Conv2DBNLayer(num_channels=96, num_filters=96, padding=1, filter_size=3,
                                                       stride=2)
        # inception_3c_pool<=Pooling<=inception_3b_output
        self.inception_3c_pool = Pool2D(pool_size=3, pool_type='max', pool_stride=2, pool_padding=1)
        # inception_3c_output  <=Concat<=
        # inception_3c_3x3(14*14 160),
        # inception_3c_double_3x3_2(14*14 96),
        # inception_3c_pool(14*14 64)
        self.conv2dnets = Conv2DNets(batch_size, sample)

        # global_pool3D<=Pooling3d<=res5b_bn
        self.global_pool3D_drop = Dropout(p=0.3)
        # global_pool<=Concat<=global_pool2D_reshape_consensus,global_pool3D
        self.fc_finall = Linear(input_dim=1536, output_dim=101)

    def forward(self, inputs, label=None):
        # dim 0 -> batch_size*sample
        inputs = fluid.layers.reshape(inputs, [-1, inputs.shape[2], inputs.shape[3], inputs.shape[4]])
        y = self.conv1(inputs)
        y = self.pool1(y)
        y = self.conv2_reduce(y)
        y = self.conv2(y)
        y = self.pool2(y)
        y = self.inception_3a(y)
        inception_3b = self.inception_3b(y)
        inception_3b = fluid.dygraph.to_variable(inception_3b)

        inception_3c_c3d = self.inception_3c_c3d(inception_3b)
        inception_3c_c3d_trans = inception_3c_c3d.numpy()
        inception_3c_c3d_trans = inception_3c_c3d_trans[np.newaxis, :]
        inception_3c_c3d_trans = fluid.dygraph.to_variable(inception_3c_c3d_trans)
        inception_3c_c3d_trans = fluid.layers.reshape(inception_3c_c3d_trans, 
            [self.batch_size, self.sample, inception_3c_c3d.shape[1], inception_3c_c3d.shape[2], inception_3c_c3d.shape[3]])
        inception_3c_c3d_trans = fluid.layers.transpose(inception_3c_c3d_trans, [0, 2, 1, 3, 4])  # [batch_size, channel, sample, 28, 28]  2D reshape-> 3D
        
        
        c3d = self.c3d(inception_3c_c3d_trans)

        y = self.inception_3c_3x3_reduce(inception_3b)
        inception_3c_3x3 = self.inception_3c_3x3(y)

        inception_3c_double_3x3_2 = self.inception_3c_double_3x3_2(inception_3c_c3d)

        inception_3c_pool = self.inception_3c_pool(inception_3b)
        # inception_3c_output  <=Concat<=
        # inception_3c_3x3_bn,inception_3c_double_3x3_2_bn,inception_3c_pool
        inception_3c_3x3 = inception_3c_3x3.numpy()
        inception_3c_double_3x3_2 = inception_3c_double_3x3_2.numpy()
        inception_3c_pool = inception_3c_pool.numpy()
        inception_3c_output = np.concatenate((inception_3c_3x3, inception_3c_double_3x3_2, inception_3c_pool), axis=1)
        inception_3c_output = fluid.dygraph.to_variable(inception_3c_output)  # [16, 576, 14, 14]

        conv2dnets = self.conv2dnets(inception_3c_output)

        global_pool3D = pool3d(c3d, pool_size=(self.sample//4, 7, 7), pool_type='avg')  # [1, 512, 1, 1, 1]
        global_pool3D_drop = self.global_pool3D_drop(global_pool3D)

        # global_pool<=Concat<=global_pool2D_reshape_consensus,global_pool3D
        conv2dnets = conv2dnets.numpy()
        global_pool3D_drop = global_pool3D_drop.numpy()
        global_pool = np.concatenate((conv2dnets, global_pool3D_drop), axis=1)
        global_pool = fluid.dygraph.to_variable(global_pool)
        global_pool = fluid.layers.reshape(global_pool, [self.batch_size, -1])
        fc_finall = self.fc_finall(global_pool)

        fc_finall = fluid.layers.softmax(fc_finall, axis=1)

        if label is not None:
            acc = fluid.layers.accuracy(fc_finall, label)  # label.shape  ->  (batch_size, 1)
            return fc_finall, acc
        else:
            return fc_finall


if __name__ == '__main__':
    # place = fluid.CUDAPlace(0)
    # with fluid.dygraph.guard(place):
    #     network = TSN(5, 32)
    #     imgs = np.zeros([5, 32, 3, 224, 224]).astype('float32')
    #     imgs = fluid.dygraph.to_variable(imgs)
        
    #     label = np.zeros([5, 1])
    #     label = fluid.dygraph.to_variable(label).astype('int64')
    #     outs = network(imgs, label)
    #     print(outs)
    label_dic = np.load('work/UCF-101_jpg/label_dir.npy', allow_pickle=True).item()
    print(label_dic)


        # ---------------for test------------------------


