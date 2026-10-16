# -*- coding: utf-8 -*-
import tensorflow as tf#.compat.v1
# tf.disable_v2_behavior()
import tensorflow.contrib.slim as slim
# import tf_slim as slim
import numpy as np
import time
import os
from utils import *



class RDN(object):

    def __init__(self,
                 sess,  
                 is_train,
                 is_eval,  
                 image_size, 
                 c_dim,  
                 batch_size,
                 D,  
                 C,  
                 G,
                 P_dim,  
                 Pc_dim,  
                 PD,  
                 PC,  
                 PG,
                 kernel_size  
                 ):

        self.sess = sess
        self.is_train = is_train
        self.is_eval = is_eval
        self.image_size = image_size
        self.c_dim = c_dim
        self.P_dim = P_dim
        self.Pc_dim = Pc_dim
        self.batch_size = batch_size
        self.D = D
        self.C = C
        self.G = G
        self.PD = PD
        self.PC = PC
        self.PG = PG
        self.kernel_size = kernel_size

    def RDBs(self, input_layer):
        rdb_concat = list()
        rdb_in = input_layer
        for i in range(1, self.D + 1):
            x = rdb_in
            for j in range(1, self.C + 1):
                tmp = slim.conv2d(x, self.G, [3, 3], rate=1, activation_fn=lrelu,
                                  weights_initializer=tf.contrib.layers.variance_scaling_initializer())
                x = tf.concat([x, tmp], axis=3)

            # local feature fusion
            x = slim.conv2d(x, self.G, [1, 1], rate=1, activation_fn=None,
                            weights_initializer=tf.contrib.layers.variance_scaling_initializer())
            # local residual learning
            rdb_in = tf.add(x, rdb_in)
            rdb_concat.append(rdb_in)

        return tf.concat(rdb_concat, axis=3)
    

    
    # def SEblock(self, input_tensor, reduction):
    #     """
    #     Squeeze-and-Excitation block.
    #     Args:
    #         input_tensor: Input feature map (4D tensor).
    #         reduction: Reduction ratio for channel squeeze.
    #     Returns:
    #         A tensor with recalibrated channel-wise features.
    #     """
    #     channel = int(input_tensor.get_shape()[-1])  # 显式转换为 int
    #     squeeze = tf.reduce_mean(input_tensor, axis=[1, 2], keepdims=True)  # Global average pooling
    #     excitation = slim.fully_connected(squeeze, channel // reduction, activation_fn=lrelu)
    #     excitation = slim.fully_connected(excitation, channel, activation_fn=tf.nn.sigmoid)
    #     scale = input_tensor * excitation  # Scale the input
    #     return scale
        
    def Merge(self, input, s0, dop, aop):
        with tf.variable_scope("merge", reuse=tf.AUTO_REUSE):
            # 下采样模块
            conv1 = slim.conv2d(input, self.PG * 2, [3, 3], rate=1, activation_fn=lrelu, scope='conv1_1')
            conv1 = slim.conv2d(conv1, self.PG * 2, [3, 3], rate=1, activation_fn=lrelu, scope='conv1_2')
            pool1 = slim.max_pool2d(conv1, [2, 2], padding='SAME')

            conv2 = slim.conv2d(pool1, self.PG * 4, [3, 3], rate=1, activation_fn=lrelu, scope='conv2_1')
            conv2 = slim.conv2d(conv2, self.PG * 4, [3, 3], rate=1, activation_fn=lrelu, scope='conv2_2')
            pool2 = slim.max_pool2d(conv2, [2, 2], padding='SAME')

            conv3 = slim.conv2d(pool2, self.PG * 8, [3, 3], rate=1, activation_fn=lrelu, scope='conv3_1')
            conv3 = slim.conv2d(conv3, self.PG * 8, [3, 3], rate=1, activation_fn=lrelu, scope='conv3_2')
            pool3 = slim.max_pool2d(conv3, [2, 2], padding='SAME')

            # 上采样模块
            up4 = upsample_and_concat(pool3, conv3, self.PG * 8, self.PG * 8, 'up4')
            conv4 = slim.conv2d(up4, self.PG * 8, [3, 3], rate=1, activation_fn=lrelu, scope='conv4_1')
            conv4 = slim.conv2d(conv4, self.PG * 8, [3, 3], rate=1, activation_fn=lrelu, scope='conv4_2')

            up5 = upsample_and_concat(conv4, conv2, self.PG * 4, self.PG * 4, 'up5')
            conv5 = slim.conv2d(up5, self.PG * 4, [3, 3], rate=1, activation_fn=lrelu, scope='conv5_1')
            conv5 = slim.conv2d(conv5, self.PG * 4, [3, 3], rate=1, activation_fn=lrelu, scope='conv5_2')

            up6 = upsample_and_concat(conv5, conv1, self.PG * 2, self.PG * 2, 'up6')
            conv6 = slim.conv2d(up6, self.PG * 2, [3, 3], rate=1, activation_fn=lrelu, scope='conv6_1')
            conv6 = slim.conv2d(conv6, self.PG * 2, [3, 3], rate=1, activation_fn=lrelu, scope='conv6_2')

            # 拼接
            conv6 = tf.concat([conv6, s0, dop, aop], 3)

            out = slim.conv2d(conv6, self.Pc_dim, [1, 1], rate=1, activation_fn=None, scope='conv10')

            return out




    
    def model(self):
        image_0 = self.images[:, 1:self.images.shape[1]:2, 1:self.images.shape[2]:2,:]
        image_45 = self.images[:, 0:self.images.shape[1]:2, 1:self.images.shape[2]:2,:]
        image_90 = self.images[:, 0:self.images.shape[1]:2, 0:self.images.shape[2]:2,:]
        image_135 = self.images[:, 1:self.images.shape[1]:2, 0:self.images.shape[2]:2,:]


        gt_0 = self.labels[:, 1:self.labels.shape[1]:2, 1:self.labels.shape[2]:2,:]
        gt_45 = self.labels[:, 0:self.labels.shape[1]:2, 1:self.labels.shape[2]:2, :]
        gt_90 = self.labels[:, 0:self.labels.shape[1]:2, 0:self.labels.shape[2]:2, :]
        gt_135 = self.labels[:, 1:self.labels.shape[1]:2, 0:self.labels.shape[2]:2, :]


        F_1_0 = slim.conv2d(image_0, self.G, [3, 3], rate=1, activation_fn=lrelu,
                          weights_initializer=tf.contrib.layers.variance_scaling_initializer())
        F0_0 = slim.conv2d(F_1_0, self.G, [3, 3], rate=1, activation_fn=lrelu,
                         weights_initializer=tf.contrib.layers.variance_scaling_initializer())
        F0 = slim.conv2d(F0_0, self.G, [3, 3], rate=1, activation_fn=lrelu,
                          weights_initializer=tf.contrib.layers.variance_scaling_initializer())

        FD_0 = self.RDBs(F0)

        FGF1_0 = slim.conv2d(FD_0, self.G, [1, 1], rate=1, activation_fn=None,
                           weights_initializer=tf.contrib.layers.variance_scaling_initializer())
        FGF2_0 = slim.conv2d(FGF1_0, self.G, [3, 3], rate=1, activation_fn=lrelu,
                           weights_initializer=tf.contrib.layers.variance_scaling_initializer())

        # FGF2_0_se = self.SEblock(FGF2_0, 16)
        
        # FDF_0 = tf.add(FGF2_0, F_1_0)
        FDF_0 = tf.concat([FGF2_0, F_1_0, F0_0, F0],axis=-1)

        IHR_0 = slim.conv2d(FDF_0, self.c_dim, [3, 3], rate=1, activation_fn=lrelu,
                          weights_initializer=tf.contrib.layers.variance_scaling_initializer())
        # PSNR_0 = psnr(IHR_0, gt_0)
        # SSIM_0 = ssim(IHR_0, gt_0)


        F_1_45 = slim.conv2d(image_45, self.G, [3, 3], rate=1, activation_fn=lrelu,
                          weights_initializer=tf.contrib.layers.variance_scaling_initializer())
        F0_45 = slim.conv2d(F_1_45, self.G, [3, 3], rate=1, activation_fn=lrelu,
                         weights_initializer=tf.contrib.layers.variance_scaling_initializer())
        F45 = slim.conv2d(F0_45, self.G, [3, 3], rate=1, activation_fn=lrelu,
                          weights_initializer=tf.contrib.layers.variance_scaling_initializer())

        FD_45 = self.RDBs(F45)

        FGF1_45 = slim.conv2d(FD_45, self.G, [1, 1], rate=1, activation_fn=None,
                           weights_initializer=tf.contrib.layers.variance_scaling_initializer())
        FGF2_45 = slim.conv2d(FGF1_45, self.G, [3, 3], rate=1, activation_fn=lrelu,
                           weights_initializer=tf.contrib.layers.variance_scaling_initializer())

        # FGF2_45_se = self.SEblock(FGF2_45, 16)
        
        # FDF_45 = tf.add(FGF2_45, F_1_45)
        FDF_45 = tf.concat([FGF2_45, F_1_45, F0_45, F45],axis=-1)

        IHR_45 = slim.conv2d(FDF_45, self.c_dim, [3, 3], rate=1, activation_fn=lrelu,
                          weights_initializer=tf.contrib.layers.variance_scaling_initializer())
        
        # PSNR_45 = psnr(IHR_45, gt_45)
        # SSIM_45 = ssim(IHR_45, gt_45)


        F_1_90 = slim.conv2d(image_90, self.G, [3, 3], rate=1, activation_fn=lrelu,
                          weights_initializer=tf.contrib.layers.variance_scaling_initializer())
        F0_90 = slim.conv2d(F_1_90, self.G, [3, 3], rate=1, activation_fn=lrelu,
                         weights_initializer=tf.contrib.layers.variance_scaling_initializer())
        F90 = slim.conv2d(F0_90, self.G, [3, 3], rate=1, activation_fn=lrelu,
                          weights_initializer=tf.contrib.layers.variance_scaling_initializer())

        FD_90 = self.RDBs(F90)

        FGF1_90 = slim.conv2d(FD_90, self.G, [1, 1], rate=1, activation_fn=None,
                           weights_initializer=tf.contrib.layers.variance_scaling_initializer())
        FGF2_90 = slim.conv2d(FGF1_90, self.G, [3, 3], rate=1, activation_fn=lrelu,
                           weights_initializer=tf.contrib.layers.variance_scaling_initializer())
        
        # FGF2_90_se = self.SEblock(FGF2_90, 16)
        # FDF_90 = tf.add(FGF2_90, F_1_90)

        FDF_90 = tf.concat([FGF2_90, F_1_90, F0_90, F90],axis=-1)

        IHR_90 = slim.conv2d(FDF_90, self.c_dim, [3, 3], rate=1, activation_fn=lrelu,
                          weights_initializer=tf.contrib.layers.variance_scaling_initializer())
        # PSNR_90 = psnr(IHR_90, gt_90)
        # SSIM_90 = ssim(IHR_90, gt_90)


        F_1_135 = slim.conv2d(image_135, self.G, [3, 3], rate=1, activation_fn=lrelu,
                          weights_initializer=tf.contrib.layers.variance_scaling_initializer())
        F0_135 = slim.conv2d(F_1_135, self.G, [3, 3], rate=1, activation_fn=lrelu,
                         weights_initializer=tf.contrib.layers.variance_scaling_initializer())
        F135 = slim.conv2d(F0_135, self.G, [3, 3], rate=1, activation_fn=lrelu,
                          weights_initializer=tf.contrib.layers.variance_scaling_initializer())

        FD_135 = self.RDBs(F135)

        FGF1_135 = slim.conv2d(FD_135, self.G, [1, 1], rate=1, activation_fn=None,
                           weights_initializer=tf.contrib.layers.variance_scaling_initializer())
        FGF2_135 = slim.conv2d(FGF1_135, self.G, [3, 3], rate=1, activation_fn=lrelu,
                           weights_initializer=tf.contrib.layers.variance_scaling_initializer())

        # FGF2_135_se = self.SEblock(FGF2_135, 16)
        # FDF_135 = tf.add(FGF2_135, F_1_135)

        FDF_135 = tf.concat([FGF2_135, F_1_135, F0_135, F135],axis=-1)

        IHR_135 = slim.conv2d(FDF_135, self.c_dim, [3, 3], rate=1, activation_fn=lrelu,
                          weights_initializer=tf.contrib.layers.variance_scaling_initializer())

        # PSNR_135 = psnr(IHR_135, gt_135)
        # SSIM_135 = ssim(IHR_135, gt_135)


        p_s0 = (IHR_0+IHR_45+IHR_90+IHR_135)/2
        p_s1 = IHR_0 - IHR_90
        p_s2 = IHR_45 - IHR_135

        

        # p_s0 = tf.concat([IHR_0,IHR_45,IHR_90,IHR_135],axis=-1)
        # p_s1 = tf.concat([IHR_0,IHR_90],axis=-1)
        # p_s2 = tf.concat([IHR_45,IHR_135],axis=-1)

        # p_dop = tf.div(tf.sqrt(tf.square(p_s1) + tf.square(p_s2)), (p_s0 + 0.00001))
        # p_dop = tf.clip_by_value(p_dop, 0, 1)
        # p_aop = 1 / 2 * tf.atan2(p_s2, p_s1)
        # p_aop = (p_aop + math.pi / 2.0) / (math.pi)



        # I0
        input_0 = tf.concat([IHR_0, p_s0, p_s1, p_s2], 3)
        FIHR_0 = self.Merge(input_0, p_s0, p_s1, p_s2)
        PSNR_0 = psnr(FIHR_0, gt_0)
        SSIM_0 = ssim(FIHR_0, gt_0)

        # I45
        input_45 = tf.concat([IHR_45, p_s0, p_s1, p_s2], 3)
        FIHR_45 = self.Merge(input_45, p_s0, p_s1, p_s2)
        PSNR_45 = psnr(FIHR_45, gt_45)
        SSIM_45 = ssim(FIHR_45, gt_45)

        # I90
        input_90 = tf.concat([IHR_90, p_s0, p_s1, p_s2], 3)
        FIHR_90 = self.Merge(input_90, p_s0, p_s1, p_s2)
        PSNR_90 = psnr(FIHR_90, gt_90)
        SSIM_90 = ssim(FIHR_90, gt_90)

        # I135
        input_135 = tf.concat([IHR_135, p_s0, p_s1, p_s2], 3)
        FIHR_135 = self.Merge(input_135, p_s0, p_s1, p_s2)
        PSNR_135 = psnr(FIHR_135, gt_135)
        SSIM_135 = ssim(FIHR_135, gt_135)

        Pred_S0 = (FIHR_0 + FIHR_45 + FIHR_90 + FIHR_135) / 2
        Pred_S1 = FIHR_0 - FIHR_90
        Pred_S2 = FIHR_45 - FIHR_135

        gt_s0 = (gt_0 + gt_45 + gt_90 + gt_135)/2
        gt_s1 = gt_0 - gt_90
        gt_s2 = gt_45 - gt_135

        PSNR_pre_s0 = psnr(p_s0/2, gt_s0/2)
        SSIM_pre_s0 = ssim(p_s0/2, gt_s0/2)

        PSNR_s0 = psnr(Pred_S0/2, gt_s0/2)
        SSIM_s0 = ssim(Pred_S0/2, gt_s0/2)

        ##训练
        # GT_DOLP = tf.div(tf.sqrt(tf.square(gt_s1) + tf.square(gt_s2)), (gt_s0 + 0.00001))
        # GT_DOLP = tf.clip_by_value(GT_DOLP, 0, 1)
        # GT_AOP = 1 / 2 * tf.atan2(gt_s2, gt_s1)
        # GT_AOP = (GT_AOP + math.pi / 2.0) / (math.pi)

        # PRED_DOLP = tf.div(tf.sqrt(tf.square(Pred_S1) + tf.square(Pred_S2)), (Pred_S0 + 0.00001))
        # PRED_DOLP = tf.clip_by_value(PRED_DOLP, 0, 1)
        # PRED_AOP = 1 / 2 * tf.atan2(Pred_S2, Pred_S1)
        # PRED_AOP = (PRED_AOP + math.pi / 2.0) / (math.pi)

        

        ##测试
        GT_DOLP = tf.div(tf.sqrt(tf.square(gt_s1) + tf.square(gt_s2)), (gt_s0 + 0.00001))
        GT_DOLP = tf.clip_by_value(GT_DOLP, 0, 1)
        GT_DOLP = (GT_DOLP[:,:,:,0]+GT_DOLP[:,:,:,1]+GT_DOLP[:,:,:,2])/3
        GT_DOLP = tf.squeeze(GT_DOLP)
        GT_DOLP = GT_DOLP[:,:,np.newaxis]

        GT_AOP = 1 / 2 * tf.atan2(gt_s2, gt_s1)
        GT_AOP = (GT_AOP + math.pi / 2.0) / (math.pi)
        GT_AOP = (GT_AOP[:,:,:,0]+GT_AOP[:,:,:,1]+GT_AOP[:,:,:,2])/3
        GT_AOP = tf.squeeze(GT_AOP)
        GT_AOP = GT_AOP[:,:,np.newaxis]


        PRED_DOLP = tf.div(tf.sqrt(tf.square(Pred_S1) + tf.square(Pred_S2)), (Pred_S0 + 0.00001))
        PRED_DOLP = tf.clip_by_value(PRED_DOLP, 0, 1)
        PRED_DOLP = (PRED_DOLP[:,:,:,0]+PRED_DOLP[:,:,:,1]+PRED_DOLP[:,:,:,2])/3
        PRED_DOLP = tf.squeeze(PRED_DOLP)
        PRED_DOLP = PRED_DOLP[:,:,np.newaxis]

        PRED_AOP = 1 / 2 * tf.atan2(Pred_S2, Pred_S1)
        PRED_AOP = (PRED_AOP + math.pi / 2.0) / (math.pi)
        PRED_AOP = (PRED_AOP[:,:,:,0]+PRED_AOP[:,:,:,1]+PRED_AOP[:,:,:,2])/3
        PRED_AOP = tf.squeeze(PRED_AOP)
        PRED_AOP = PRED_AOP[:,:,np.newaxis]

        PSNR_aop = psnr(PRED_AOP, GT_AOP)
        SSIM_aop = ssim(PRED_AOP, GT_AOP)
        PSNR_dolp = psnr(PRED_DOLP, GT_DOLP)
        SSIM_dolp = ssim(PRED_DOLP,GT_DOLP)


        return p_s2, FIHR_0, FIHR_45, FIHR_90, FIHR_135, gt_0, gt_45, gt_90, gt_135, Pred_S0, PRED_DOLP, PRED_AOP, gt_s0, GT_DOLP, GT_AOP, PSNR_pre_s0, PSNR_s0, PSNR_dolp, PSNR_aop, SSIM_pre_s0, SSIM_s0, SSIM_dolp, SSIM_aop


    def build_model(self, images_shape, labels_shape):
        self.images = tf.placeholder(tf.float32, images_shape, name='images')
        self.labels = tf.placeholder(tf.float32, labels_shape, name='labels')

        self.pre_s2, self.pred_0, self.pred_45, self.pred_90, self.pred_135, self.gt_0, self.gt_45, self.gt_90,self.gt_135, self.pred_s0, self.pred_dolp, self.pred_aop, self.gt_s0, self.gt_dolp, self.gt_aop, self.psnr_pre_s0, self.psnr_s0, self.psnr_dop, self.psnr_aop, self. ssim_pre_s0, self.ssim_s0, self.ssim_dop, self.ssim_aop = self.model()


        

        self.loss_0 = tf.reduce_mean(tf.abs(self.gt_0 - self.pred_0))+0.0000001
        self.loss_45 = tf.reduce_mean(tf.abs(self.gt_45 - self.pred_45))+0.0000001
        self.loss_90 = tf.reduce_mean(tf.abs(self.gt_90 - self.pred_90))+0.0000001
        self.loss_135 = tf.reduce_mean(tf.abs(self.gt_135 - self.pred_135))+0.0000001
        w0 = self.loss_0/(self.loss_0 + self.loss_45 + self.loss_90 + self.loss_135)
        w45 = self.loss_45/(self.loss_0 + self.loss_45 + self.loss_90 + self.loss_135)
        w90 = self.loss_90/(self.loss_0 + self.loss_45 + self.loss_90 + self.loss_135)
        w135 = self.loss_135/(self.loss_0 + self.loss_45 + self.loss_90 + self.loss_135)
        self.I_loss = w0*self.loss_0 + w45*self.loss_45 + w90*self.loss_90 + w135*self.loss_135


        # self.I1_loss = tf.reduce_mean(tf.abs(self.gt_0 - self.pred_0))+0.0000001
        # self.I2_loss = tf.reduce_mean(tf.abs(self.gt_45 - self.pred_45))+0.0000001
        # self.I3_loss = tf.reduce_mean(tf.abs(self.gt_90 - self.pred_90))+0.0000001
        # self.I4_loss = tf.reduce_mean(tf.abs(self.gt_135 - self.pred_135))+0.0000001
        self.dolp_loss = 10*tf.reduce_mean(tf.abs(self.gt_dolp - self.pred_dolp)) + 0.0000001  
        self.aop_loss = 2*tf.reduce_mean(tf.abs(self.gt_aop - self.pred_aop)) + 0.0000001  
        # self.S0_loss =  tf.reduce_mean(tf.abs(self.gt_s0 - self.pred_s0))+0.0000001
        # # self.pre_s0_loss = tf.reduce_mean(tf.abs(self.gt_s0 - self.pre_s0))+0.0000001

        # # wps = self.pre_s0_loss / (self.aop_loss + self.dolp_loss + self.S0_loss + self.pre_s0_loss)
        # wS = self.S0_loss / (self.aop_loss + self.dolp_loss + self.S0_loss)
        wd = self.dolp_loss / (self.aop_loss + self.dolp_loss) # + self.S0_loss)
        wa = self.aop_loss / (self.aop_loss + self.dolp_loss) # + self.S0_loss)
        self.Spad_loss = wd * self.dolp_loss + wa * self.aop_loss #+ wS*self.S0_loss #+ wps * self.pre_s0_loss

        self.loss = 0.7* self.I_loss + 0.3 * self.Spad_loss

        
        
        # I1dy_true, I1dx_true = tf.image.image_gradients(self.gt_0)
        # I1dy_pred, I1dx_pred = tf.image.image_gradients(self.pred_0)
        # I2dy_true, I2dx_true = tf.image.image_gradients(self.gt_45)
        # I2dy_pred, I2dx_pred = tf.image.image_gradients(self.pred_45)
        # I3dy_true, I3dx_true = tf.image.image_gradients(self.gt_90)
        # I3dy_pred, I3dx_pred = tf.image.image_gradients(self.pred_90)
        # I4dy_true, I4dx_true = tf.image.image_gradients(self.gt_135)
        # I4dy_pred, I4dx_pred = tf.image.image_gradients(self.pred_135)
        # self.grad1loss = self.I1_loss + 0.5*tf.reduce_mean(tf.abs(I1dy_pred - I1dy_true) + tf.abs(I1dx_pred - I1dx_true))+0.0000001
        # self.grad2loss = self.I2_loss + 0.5*tf.reduce_mean(tf.abs(I2dy_pred - I2dy_true) + tf.abs(I2dx_pred - I2dx_true))+0.0000001
        # self.grad3loss = self.I3_loss + 0.5*tf.reduce_mean(tf.abs(I3dy_pred - I3dy_true) + tf.abs(I3dx_pred - I3dx_true))+0.0000001
        # self.grad4loss = self.I4_loss + 0.5*tf.reduce_mean(tf.abs(I4dy_pred - I4dy_true) + tf.abs(I4dx_pred - I4dx_true))+0.0000001
        

        
        # # wgs= self.gradsloss / (self.gradpsloss + self.gradsloss)
        # wg1 = self.grad1loss / (self.grad1loss + self.grad2loss + self.grad3loss + self.grad4loss)
        # wg2 = self.grad2loss / (self.grad1loss + self.grad2loss + self.grad3loss + self.grad4loss)
        # wg3 = self.grad3loss / (self.grad1loss + self.grad2loss + self.grad3loss + self.grad4loss)
        # wg4 = self.grad4loss / (self.grad1loss + self.grad2loss + self.grad3loss + self.grad4loss)
        # self.gradloss = wg1 * self.grad1loss + wg2 * self.grad2loss + wg3 * self.grad3loss + wg4 * self.grad4loss

        # self.loss = 0.7*self.gradloss + 0.3*self.Spad_loss

        
        
        #
        # self.gradloss = ( self.grads1loss +  self.grads2loss + self.grads0loss)*0.5+0.0000001

        
        

        tf.summary.scalar('loss', self.loss)
        # tf.summary.scalar('closs', self.n_loss)
        # tf.summary.scalar('Pre_PSNR_s0', self.pre_psnr_s0)
        # tf.summary.scalar('Pre_PSNR_dop', self.pre_psnr_dop)
        # tf.summary.scalar('Pre_PSNR_aop', self.pre_psnr_aop)
        tf.summary.scalar('PSNR_s0', self.psnr_s0)
        tf.summary.scalar('PSNR_dop', self.psnr_dop)
        tf.summary.scalar('PSNR_aop', self.psnr_aop)
        # tf.summary.scalar('Pre_SSIM_s0', self.pre_ssim_s0)
        # tf.summary.scalar('Pre_SSIM_dop', self.pre_ssim_dop)
        # tf.summary.scalar('Pre_SSIM_aop', self.pre_ssim_aop)
        tf.summary.scalar('SSIM_s0', self.ssim_s0)
        tf.summary.scalar('SSIM_dop', self.ssim_dop)
        tf.summary.scalar('SSIM_aop', self.ssim_aop)

        self.saver = tf.train.Saver(max_to_keep=5)

    def train(self, config):

        input_setup(config) 

        train_data_dir = get_data_dir(config.is_train)
        train_data_num = get_data_num(train_data_dir)
        batch_num = train_data_num // config.batch_size
        # eval_data_num = get_data_num(eval_data_dir)


        images_shape = [None, self.image_size, self.image_size , self.c_dim]
        labels_shape = [None, self.image_size, self.image_size, self.c_dim]

        self.build_model(images_shape, labels_shape)

        epoch, counter = self.load(config.checkpoint_dir)
        global_step = tf.Variable(counter, trainable=False)
        # learning_rate = config.learning_rate
        learning_rate = tf.train.exponential_decay(config.learning_rate, global_step, config.lr_decay_steps * batch_num,
                                                   config.lr_decay_rate, staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        learning_step = optimizer.minimize(self.loss, global_step=global_step)

        tf.global_variables_initializer().run(session=self.sess)

        merged_summary_op = tf.summary.merge_all()
        summary_train_path = os.path.join(config.checkpoint_dir, "train_%s_%s_%s" % (self.D, self.C, self.G))
        # summary_eval_path = os.path.join(config.checkpoint_dir, "eval_%s_%s_%s" % (self.D, self.C, self.G))

        summary_writer_train = tf.summary.FileWriter(summary_train_path, self.sess.graph)
        # summary_writer_validate = tf.summary.FileWriter(summary_eval_path)

        time_all = time.time()
        print("\nNow Start Training...\n")
        model_dir = "%s_%s_%s_%s" % ("rdn", self.D, self.C, self.G)
        checkpoint_dir = os.path.join(config.checkpoint_dir, model_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

        if ckpt and ckpt.model_checkpoint_path:
            ckpt_path = str(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(os.getcwd(), ckpt_path))

        for ep in range(epoch, config.epoch):
            # Run by batch images

            for idx in range(0, batch_num):
                batch_images, batch_labels = get_batch(train_data_dir, train_data_num, config.batch_size)
                # eval_batch_images, eval_batch_labels = get_batch(eval_data_dir, eval_data_num, config.batch_size)
                counter += 1

                assert batch_images.shape == batch_labels.shape
                # assert eval_batch_images.shape == eval_batch_labels.shape
                # _, loss, lr, pre_psnr_s0, pre_psnr_aop, pre_psnr_dolp, psnr_0, psnr_45, psnr_90, psnr_135, psnr_s0, psnr_aop,psnr_dolp, pre_ssim_s0, pre_ssim_aop, pre_ssim_dolp, ssim_0,ssim_45, ssim_90, ssim_135, ssim_s0,ssim_aop,ssim_dolp = self.sess.run(
                #     [learning_step, self.loss, learning_rate, self.pre_psnr_s0, self.pre_psnr_aop, self.pre_psnr_dop, self.psnr_0, self.psnr_45,
                #      self.psnr_90, self.psnr_135,self.psnr_s0,self.psnr_aop,self.psnr_dop,
                #      self.pre_ssim_s0, self.pre_ssim_aop, self.pre_ssim_dop, self.ssim_0, self.ssim_45, self.ssim_90, self.ssim_135,self.ssim_s0,self.ssim_aop,self.ssim_dop],
                #     feed_dict={self.images: batch_images, self.labels: batch_labels})
                # _, loss, lr,  psnr_s0, psnr_aop,psnr_dolp, ssim_s0,ssim_aop,ssim_dolp = self.sess.run(
                #     [learning_step, self.loss, learning_rate,  self.psnr_s0,self.psnr_aop,self.psnr_dop,
                #      self.ssim_s0,self.ssim_aop,self.ssim_dop],
                #     feed_dict={self.images: batch_images, self.labels: batch_labels})
                _, loss, psnr_s0, psnr_aop, psnr_dolp, ssim_s0, ssim_aop, ssim_dolp = self.sess.run(
                    [learning_step, self.loss, self.psnr_s0, self.psnr_aop, self.psnr_dop,
                    self.ssim_s0, self.ssim_aop, self.ssim_dop],
                    feed_dict={self.images: batch_images, self.labels: batch_labels})

                # eval_loss = self.sess.run(self.loss,
                #                           feed_dict={self.images: eval_batch_images,
                #                                      self.labels: eval_batch_labels})
                if counter % 10 == 0:
                    # print(
                    #     "Epoch: [%2d], batch: [%2d/%2d], step: [%2d], time: [%d]min, psnr_0:[%2.2f], psnr_45:[%2.2f], psnr_90:[%2.2f], psnr_135:[%2.2f],psnr_s0:[%2.2f],psnr_s1:[%2.2f],psnr_s2:[%2.2f],psnr_aop:[%2.2f],psnr_dolp:[%2.2f], train_loss: [%.4f]" % (
                    #         ep + 1, idx, batch_num, counter, int((time.time()-time_all)/60), psnr_0, psnr_45, psnr_90,psnr_135,psnr_s0,psnr_s1,psnr_s2,psnr_aop,psnr_dolp,
                    #          loss))
                    # print(
                    #     "Epoch: [%2d], batch: [%2d/%2d], step: [%2d], time: [%d]min, pre_psnr_s0:[%2.2f], pre_psnr_aop:[%2.2f], pre_psnr_dolp:[%2.2f], psnr_0:[%2.2f], psnr_45:[%2.2f], psnr_90:[%2.2f], psnr_135:[%2.2f], psnr_s0:[%2.2f], psnr_aop:[%2.2f], psnr_dolp:[%2.2f], pre_ssim_s0:[%2.2f], pre_ssim_aop:[%2.2f], pre_ssim_dolp:[%2.2f], ssim_0:[%2.2f], ssim_45:[%2.2f], ssim_90:[%2.2f], ssim_135:[%2.2f], ssim_s0:[%2.2f], ssim_aop:[%2.2f], ssim_dolp:[%2.2f], train_loss: [%.4f]" % (
                    #         ep + 1, idx, batch_num, counter, int((time.time()-time_all)/60), pre_psnr_s0, pre_psnr_aop, pre_psnr_dolp, psnr_0, psnr_45, psnr_90, psnr_135, psnr_s0,psnr_aop,psnr_dolp,pre_ssim_s0, pre_ssim_aop, pre_ssim_dolp, ssim_0,ssim_45, ssim_90, ssim_135, ssim_s0,ssim_aop,ssim_dolp,
                    #          loss))
                    
                    print(
                        "Epoch: [%2d], batch: [%2d/%2d], step: [%2d], time: [%d]min,  psnr_s0:[%2.2f], psnr_aop:[%2.2f], psnr_dolp:[%2.2f], ssim_s0:[%2.2f], ssim_aop:[%2.2f], ssim_dolp:[%2.2f], train_loss: [%.4f]" % (
                            ep + 1, idx, batch_num, counter, int((time.time()-time_all)/60),  psnr_s0,psnr_aop,psnr_dolp, ssim_s0, ssim_aop, ssim_dolp,
                             loss))

                if counter % 100 == 0:
                    print(int((time.time() - time_all) / 60))
                    self.save(config.checkpoint_dir, ep + 1, counter)
                    summary_train = self.sess.run(merged_summary_op,
                                                  feed_dict={self.images: batch_images, self.labels: batch_labels})
                    summary_writer_train.add_summary(summary_train, counter)
                    # summary_eval = self.sess.run(merged_summary_op,
                    #                              feed_dict={self.images: eval_batch_images,
                    #                                         self.labels: eval_batch_labels})
                    # summary_writer_validate.add_summary(summary_eval, counter)
                if counter > 0 and counter == batch_num * config.epoch:
                    print("Congratulation !  Train Finished.")
                    print("Congratulation !  Train Finished.")
                    print("Congratulation !  Train Finished.")
                    return

    def test(self, config):
        print("\nPrepare Testing Data...\n")
        input_paths, label_paths = prepare_data(config)  
        data_num = len(input_paths)

        psnr_pre_s0_list = []
        psnr_s0_list = []
        psnr_aop_list = []
        psnr_dop_list = []
        ssim_pre_s0_list = []
        ssim_s0_list = []
        ssim_aop_list = []
        ssim_dop_list = []

        print("\nNow Start Testing...\n")
        times = []
        s = time.time()
        for idx in range(data_num):
            output_name = input_paths[idx].split("/")[-1].split('.')[0]
            ratio = float(30)
            input_ = cv2.imread(input_paths[idx], -1)
            input_ = input_ * ratio
            input_ = input_[np.newaxis, :]
            label_ = cv2.imread(label_paths[idx], -1)
            label_ = label_[np.newaxis, :]

            test_ = np.zeros([input_.shape[0], input_.shape[1], input_.shape[2], input_.shape[3]])
            images_shape = input_.shape
            labels_shape = label_.shape

            self.build_model(images_shape, labels_shape)
            tf.global_variables_initializer().run(session=self.sess)

            self.load(config.checkpoint_dir)
            s = time.time()
            pre_s2, result_0, result_45, result_90, result_135, result_s0, result_dolp, result_aop, psnr_pre_s0, psnr_s0, psnr_aop, psnr_dop, ssim_pre_s0, ssim_s0, ssim_aop, ssim_dop = self.sess.run([self.pre_s2, self.pred_0, self.pred_45, self.pred_90, self.pred_135, self.pred_s0, self.pred_dolp, self.pred_aop, self.psnr_pre_s0, self.psnr_s0, self.psnr_aop,self.psnr_dop,self.ssim_s0, self.ssim_pre_s0, self.ssim_aop,self.ssim_dop],
                                                              feed_dict={self.images: input_/ 255,
                                                                         self.labels: label_/ 255})

            tot = time.time() - s
            self.sess.close()
            tf.reset_default_graph()
            self.sess = tf.Session()


            times.append(tot)

            # print(result_0.shape)
            # print(pre_s0.shape)
            # print(result_s0.shape)
            


            # img_0 = np.squeeze(result_0, axis=0)
            # img_45 = np.squeeze(result_45, axis=0)
            # img_90 = np.squeeze(result_90, axis=0)
            # img_135 = np.squeeze(result_135, axis=0)
            img_pre_s2 = np.squeeze(pre_s2)
            img_s0 = np.squeeze(result_s0)
            img_dolp = np.squeeze(result_dolp)
            img_aop = np.squeeze(result_aop)

            psnr_pre_s0_list.append(psnr_pre_s0)
            psnr_s0_list.append(psnr_s0)
            psnr_aop_list.append(psnr_aop)
            psnr_dop_list.append(psnr_dop)
            ssim_pre_s0_list.append(ssim_pre_s0)
            ssim_s0_list.append(ssim_s0)
            ssim_aop_list.append(ssim_aop)
            ssim_dop_list.append(ssim_dop)
            
            save_our(config, img_pre_s2, img_s0,img_dolp,img_aop,output_name)

        avg_psnr_pre_s0 = np.mean(psnr_pre_s0_list)
        avg_psnr_s0 = np.mean(psnr_s0_list)
        avg_psnr_aop = np.mean(psnr_aop_list)
        avg_psnr_dop = np.mean(psnr_dop_list)
        avg_ssim_pre_s0 = np.mean(ssim_pre_s0_list)
        avg_ssim_s0 = np.mean(ssim_s0_list)
        avg_ssim_aop = np.mean(ssim_aop_list)
        avg_ssim_dop = np.mean(ssim_dop_list)

        print("\nAverage PSNR (pre_s0):", avg_psnr_pre_s0)
        print("\nAverage PSNR (s0):", avg_psnr_s0)
        print("Average PSNR (aop):", avg_psnr_aop)
        print("Average PSNR (dop):", avg_psnr_dop)
        print("Average SSIM (pre_s0):", avg_ssim_pre_s0)
        print("Average SSIM (s0):", avg_ssim_s0)
        print("Average SSIM (aop):", avg_ssim_aop)
        print("Average SSIM (dop):", avg_ssim_dop)


        print("\n All Done ! ")
        print("\nTotal images: {0}".format(data_num))
        Ttime, Mtime = np.sum(times[1:]), np.mean(times[1:])
        print("Time taken: {0} sec at {1} fps".format(Ttime, 1./Mtime))

    def load(self, checkpoint_dir):
        print("\nReading Checkpoints.....\n")
        model_dir = "%s_%s_%s_%s" % ("rdn", self.D, self.C, self.G)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

        if ckpt and ckpt.model_checkpoint_path:
            ckpt_path = str(ckpt.model_checkpoint_path)
            print(os.path.join(os.getcwd(), ckpt_path))
            self.saver.restore(self.sess, os.path.join(os.getcwd(), ckpt_path))
            step = int(ckpt_path.split('-')[-1])
            epoch = int(ckpt_path.split('-')[1])
            print("\nCheckpoint Loading Success! %s\n" % ckpt_path)
        else:
            step = 0
            epoch = 0
            print("\nCheckpoint Loading Failed! \n")

        return epoch, step

    def save(self, checkpoint_dir, epoch, step):
        model_name = "RDN.model"
        model_dir = "%s_%s_%s_%s" % ("rdn", self.D, self.C, self.G)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        save_name = os.path.join(checkpoint_dir, model_name + '-{}'.format(epoch + 1))
        self.saver.save(self.sess,
                        save_name,
                        global_step=step)
