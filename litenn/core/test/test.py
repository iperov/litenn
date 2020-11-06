import tempfile
from pathlib import Path

import numpy as np

import litenn as nn
import litenn.core as nc

from litenn.core.op.concat import concat_test
from litenn.core.op.depthwise_conv2D import depthwise_conv2d_test
from litenn.core.op.dropout import dropout_test
from litenn.core.op.conv2D import conv2d_test
from litenn.core.op.conv2DTranspose import conv2DTranspose_test
from litenn.core.op.element_wise_op import element_wise_op_test
from litenn.core.op.dual_wise_op import dual_wise_op_test
from litenn.core.op.matmul import matmul_test
from litenn.core.op.pool2D import pool2d_test
from litenn.core.op.reduce import reduce_test
from litenn.core.op.resize2D_bilinear import resize2D_bilinear_test
from litenn.core.op.resize2D_nearest import resize2D_nearest_test
from litenn.core.op.slice import slice_test
from litenn.core.op.stack import stack_test
from litenn.core.op.ssim import ssim_test, dssim_test
from litenn.core.op.tile import tile_test
from litenn.core.op.transpose import transpose_test

from litenn.core.module.BatchNorm2D import BatchNorm2D_test
from litenn.core.module.Conv2D import Conv2D_test
from litenn.core.module.Conv2DTranspose import Conv2DTranspose_test
from litenn.core.module.Dense import Dense_test
from litenn.core.module.DenseAffine import DenseAffine_test
from litenn.core.module.Dropout import Dropout_test
from litenn.core.module.InstanceNorm2D import InstanceNorm2D_test
from litenn.core.module.SeparableConv2D import SeparableConv2D_test

def Module_test():
    filepath = Path(tempfile.gettempdir()) / '123456789.bin'


    class Module1(nn.Module):
        def __init__(self, include_self=True):
            if include_self:
                self.mod = Module1(include_self=False)

            self.a = 1
            self.l = [1,1]

            self.conv1 = nn.Conv2D(1,1, 1, stride=1, padding='same', kernel_initializer=nn.initializer.Scalar(1.0), bias_initializer=nn.initializer.Scalar(1.0))
            self.list_of_convs = [
                        nn.Conv2D(1,1, 1, stride=1, padding='same', kernel_initializer=nn.initializer.Scalar(1.0), bias_initializer=nn.initializer.Scalar(1.0)),
                        nn.Conv2D(1,1, 1, stride=1, padding='same', kernel_initializer=nn.initializer.Scalar(1.0), bias_initializer=nn.initializer.Scalar(1.0))
                            ]
            self.dict_of_convs = {
                        0     : nn.Conv2D(1,1, 1, stride=1, padding='same', kernel_initializer=nn.initializer.Scalar(1.0), bias_initializer=nn.initializer.Scalar(1.0)),
                        'asd' : nn.Conv2D(1,1, 1, stride=1, padding='same', kernel_initializer=nn.initializer.Scalar(1.0), bias_initializer=nn.initializer.Scalar(1.0))
                        }

    class Trainer(nn.Module):
        def __init__(self):

            self.mod1 = Module1()
            self.opt1 = nn.optimizer.RMSprop(self.mod1.trainables() )

    trainer = Trainer()
    trainer2 = Trainer()

    if trainer.mod1.mod.conv1.kernel.get_name() != \
      'Trainer.mod1.mod.conv1.kernel':
        raise Exception(f'wrong Tensor name')

    if trainer.mod1.mod.list_of_convs[0].kernel.get_name() != \
      'Trainer.mod1.mod.list_of_convs[0].kernel':
        raise Exception(f'wrong Tensor name')

    if trainer.mod1.mod.dict_of_convs['asd'].kernel.get_name() != \
      'Trainer.mod1.mod.dict_of_convs[asd].kernel':
        raise Exception(f'wrong Tensor name')

    trainer.save(filepath)
    trainer.set_training(False)

    # Change tensors data
    for m in [trainer.mod1, trainer.mod1.mod]:
        for conv in [m.conv1, m.list_of_convs[0], m.list_of_convs[1],
                     m.dict_of_convs[0], m.dict_of_convs[0], m.dict_of_convs['asd'], m.dict_of_convs['asd']
                    ]:

            conv.kernel.fill(2.0)
            conv.bias.fill(3.0)

    # Change variables data
    trainer.mod1.a = 2
    trainer.mod1.l[0] = 2
    trainer.mod1.l[1] = 2
    trainer.mod1.mod.a = 2
    trainer.mod1.mod.l[0] = 2
    trainer.mod1.mod.l[1] = 2

    trainer.set_training(True)
    trainer.load(filepath)

    # Check loaded data
    for m in [trainer.mod1, trainer.mod1.mod]:
        for conv in [m.conv1, m.list_of_convs[0], m.list_of_convs[1],
                     m.dict_of_convs[0], m.dict_of_convs[0], m.dict_of_convs['asd'], m.dict_of_convs['asd']
                    ]:

            if not all( conv.kernel.np() == np.ones(conv.kernel.shape) ):
                raise Exception("tensor data invalid")
            if not all ( conv.bias.np() == np.ones(conv.bias.shape) ):
                raise Exception("tensor data invalid")

    # Load the same data from trainer to trainer2
    trainer2.load(filepath)

    # Change trainer.opt1._accs
    accs = trainer.opt1._accs
    acc_t = accs[list(accs.keys())[0]]
    acc_t.fill(5.0)

    # Load submodule by dump submodule
    trainer.opt1.load_state(trainer2.opt1.dump_state())

    accs = trainer.opt1._accs
    acc_t = accs[list(accs.keys())[0]]
    if not all( np.ndarray.flatten(acc_t.np() == np.zeros(acc_t.shape)) ):
        raise Exception("acc_t tensor data invalid")


    if trainer.mod1.a != 1 or \
       trainer.mod1.l[0] != 1 or \
       trainer.mod1.l[1] != 1 or \
       trainer.mod1.mod.a != 1 or \
       trainer.mod1.mod.l[0] != 1 or \
       trainer.mod1.mod.l[1] != 1 or \
       not trainer.mod1.mod.is_training():
        raise Exception("variable data invalid")

    filepath.unlink()

def shallow_mode_test():
    class Module1(nn.Module):
        def __init__(self, include_self=True):
            self.conv = nn.Conv2D(1,1, 3, stride=2)
            
        def forward(self, x):
            x = self.conv(x)
            return x

    m = Module1()
    m.set_training(True)

    out = m.shallow_forward ( nn.Tensor( (1,1,16,16) ) )

    dev = nn.devices.get_current()[0]

    if dev.get_used_memory() != 0:
        raise Exception('Memory is allocated in shallow_forward')

def backward_test():
    inp_t = nn.Tensor( (2,2,8,8), init=nn.initializer.Scalar(1.0) )
    kernel_t = nn.Tensor( (4,2,3,3), init=nn.initializer.Scalar(0.0) )

    kernel_2_t = nn.Tensor( (4,2,3,3), init=nn.initializer.Scalar(0.0) )

    opt = nn.optimizer.RMSprop([inp_t, kernel_t, kernel_2_t])
    opt.zero_grad()

    with nn.optimizer.freeze():
        r = nn.conv2D(inp_t, kernel_2_t)
    r.backward()

    if kernel_2_t.has_grad():
        raise Exception("kernel_2_t has grad, but used inside nn.optimizer.freeze()")

    r = nn.conv2D(inp_t, kernel_t)
    r.backward()

    if not kernel_t.has_grad():
        raise Exception("kernel_t has no grad")

    kernel_grad_t = kernel_t.get_grad()
    if all ( np.ndarray.flatten( kernel_grad_t.np() ) == 0 ):
         raise Exception("kernel_grad_t is not changed after backward step.")

    opt.step()

    if all ( np.ndarray.flatten( kernel_t.np() ) == 0 ):
         raise Exception("kernel_t is not changed after optimization step.")

def MultiGPU_test_():
    devices_count = len(nn.devices.get_current())

    x = nn.Tensor_sliced_from_value ( np.arange(0,devices_count).reshape( (devices_count,) ) )

    weight_t = nn.Tensor( (1,), init=nn.initializer.Scalar(0.0) )
    weight_t.get_grad().set( x )

    opt = nn.optimizer.RMSprop([weight_t], rho=1)
    opt.step()
    if all(np.ndarray.flatten(weight_t.np(0)) == \
           np.ndarray.flatten(weight_t.np(1)) ):
        raise Exception("weight_t is equal on both GPUs")


    weight_t = nn.Tensor( (1,), init=nn.initializer.Scalar(0.0) )
    weight_t.get_grad().set( x )
    opt = nn.optimizer.RMSprop([weight_t], rho=1)
    opt.step(multi_gpu_step=True)
    if all(np.ndarray.flatten(weight_t.np(0)) != \
           np.ndarray.flatten(weight_t.np(1)) ):
        raise Exception("weight_t is not equal on both GPUs")

def MultiGPU_test():
    devices = nn.devices.get_all()
    if len(devices) >= 2:
        saved_devices = nn.devices.get_current()


        devices = devices[0:2]
        nn.devices.set_current(devices)

        MultiGPU_test_()

        nn.devices.set_current(saved_devices)
    else:
        print("Unable to test MultiGPU, because you have only one GPU.")

def Initializers_test():
    nn.Tensor( (128,), init=nn.initializer.Scalar(1.0) ).np()
    nn.Tensor( (128,), init=nn.initializer.GlorotUniform(1.0, 1.0, 1.0)).np()
    nn.Tensor( (128,), init=nn.initializer.GlorotNormal(1.0, 1.0, 1.0)).np()
    nn.Tensor( (128,), init=nn.initializer.RandomNormal() ).np()
    nn.Tensor( (128,), init=nn.initializer.RandomUniform() ).np()

def Adam_test():
    weight_t = nn.Tensor( (16,), init=nn.initializer.Scalar(0.0) )
    weight_t.get_grad().fill(1.0)

    opt = nn.optimizer.Adam([weight_t], lr_decay=0.1, lr_dropout=0.7, clipnorm=0.1)
    opt.step()

def RMSprop_test():
    weight_t = nn.Tensor( (16,), init=nn.initializer.Scalar(0.0) )
    weight_t.get_grad().fill(1.0)

    opt = nn.optimizer.RMSprop([weight_t], lr_decay=0.1, lr_dropout=0.7, clipnorm=0.1)
    opt.step()

def SGD_test():
    weight_t = nn.Tensor( (16,), init=nn.initializer.Scalar(0.0) )
    weight_t.get_grad().fill(1.0)

    opt = nn.optimizer.SGD([weight_t], lr_decay=0.1, lr_dropout=0.7, clipnorm=0.1)
    opt.step()

def test_all(iterations=1):
    """
    Test library.
    """

    if nn.Tensor._object_count != 0:
        raise Exception('Unable to run test_all() while any Tensor object exists.')

    nn.devices.set_current(nn.devices.ask_to_choose(choose_only_one=True))

    test_funcs = [
        
        dropout_test,
        element_wise_op_test,
        dual_wise_op_test,
        transpose_test,
        matmul_test,
        slice_test,
        reduce_test,
        concat_test,
        stack_test,
        tile_test,
        conv2d_test,
        depthwise_conv2d_test,
        conv2DTranspose_test,
        pool2d_test,
        resize2D_bilinear_test,
        resize2D_nearest_test,
        
        ssim_test,
        dssim_test,
        
        backward_test,
        MultiGPU_test,
        Initializers_test,
        
        Adam_test,
        RMSprop_test,
        SGD_test,
        
        Module_test,
        BatchNorm2D_test,
        Conv2D_test,
        Conv2DTranspose_test,
        Dense_test,
        DenseAffine_test,
        Dropout_test,
        InstanceNorm2D_test,
        SeparableConv2D_test,

        shallow_mode_test,
        ]
    for _ in range(iterations):
        for test_func in test_funcs:
            print(f'{test_func.__name__}()')
            test_func()
            nn.cleanup()

    print('Done.')














