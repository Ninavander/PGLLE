# -*- coding: utf-8 -*-
import cv2
import tensorflow as tf
import numpy as np
import h5py
import math
import glob
import os
# import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim as slim
# import tensorflow.contrib.slim.nets


def psnr(img1, img2):
    return tf.reduce_mean(tf.image.psnr(img1, img2, 1.0))

def ssim(img1, img2):
    return tf.reduce_mean(tf.image.ssim(img1, img2, 1.0))
    # return tf.reduce_mean(tf.image.psnr(img1, img2, 1.0))


def lrelu(x):
    return tf.maximum(x * 0.2, x)

# def tvloss(x, TVLoss_weight=1):
#     # Compute Total Variation loss
#     tv_loss = tf.reduce_sum(tf.image.total_variation(x))
    
#     # Return weighted TV loss
#     return TVLoss_weight * tv_loss

def upsample_and_concat(x1, x2, output_channels, in_channels, scope):
    deconv_output = slim.conv2d_transpose(x1, output_channels, [2, 2], stride=2, padding='SAME',
                                          weights_initializer=tf.truncated_normal_initializer(stddev=0.02), scope=scope)
    H, W = tf.shape(deconv_output)[1], tf.shape(deconv_output)[2]
    h, w = tf.shape(x2)[1], tf.shape(x2)[2]
    H = tf.cast(H, tf.float32)
    W = tf.cast(W, tf.float32)
    h = tf.cast(h, tf.float32)
    w = tf.cast(w, tf.float32)
    x1 = tf.cast(tf.round((W - w) / 2), tf.int32)
    y1 = tf.cast(tf.round((H - h) / 2), tf.int32)
    h = tf.cast(h, tf.int32)
    w = tf.cast(w, tf.int32)
    deconv_output = deconv_output[:, y1: y1 + h, x1: x1 + w, :]
    deconv_output = tf.concat([deconv_output, x2], 3)
    #deconv_output.set_shape([None, None, None, output_channels * 2])
    return deconv_output



def prepare_data(config):

    if config.is_train:

        input_dir = os.path.join(os.path.join(os.getcwd(), config.train_set_input))
        input_list = glob.glob(os.path.join(input_dir, "*png"))
        input_list.sort()
        label_dir = os.path.join(os.path.join(os.getcwd(), config.train_set_label))
        label_list = glob.glob(os.path.join(label_dir, "*png"))
        label_list.sort()

        # eval_input_dir = os.path.join(os.path.join(os.getcwd(), config.eval_set_input))
        # eval_input_list = glob.glob(os.path.join(eval_input_dir, "*png"))
        # eval_input_list.sort()
        # eval_label_dir = os.path.join(os.path.join(os.getcwd(), config.eval_set_label))
        # eval_label_list = glob.glob(os.path.join(eval_label_dir, "*png"))
        # eval_label_list.sort()
        return input_list, label_list #, eval_input_list, eval_label_list

    else:

        test_dir = os.path.join(os.getcwd(), config.test_set)
        test_list = glob.glob(os.path.join(test_dir, "*.png"))
        test_list.sort()
        label_dir = os.path.join(os.path.join(os.getcwd(), config.test_set_label))
        label_list = glob.glob(os.path.join(label_dir, "*png"))
        label_list.sort()

        return test_list, label_list


def input_setup(config):

    input_list, label_list = prepare_data(config)
    print('Prepare training data...')
    make_sub_data(input_list, label_list, config, 'train')
    # print('Prepare evaluating data...')
    # make_sub_data(eval_input_list, eval_label_list, config, 'eval')


def make_data_hf(input_, label_, config, str, times):

    assert input_.shape == label_.shape
    #assert input_.shape[1]*2 == label_.shape[1]
    if not os.path.isdir(os.path.join(os.getcwd(), config.checkpoint_dir)):
        os.makedirs(os.path.join(os.getcwd(), "checkpoint"))
    if str == 'train':
        savepath = os.path.join(os.path.join(os.getcwd(), "checkpoint"), 'train.h5')
    elif str == 'eval':
        savepath = os.path.join(os.path.join(os.getcwd(), "checkpoint"), 'eval.h5')

    else:
        savepath = os.path.join(os.path.join(os.getcwd(), config.checkpoint_dir), 'test.h5')

    if times == 0:  
        if os.path.exists(savepath):
            print("\n%s have existed!\n" % (savepath))
            return False
        else:
            hf = h5py.File(savepath, 'w')
            if config.is_train:
                input_h5 = hf.create_dataset("input", (1, config.image_size, config.image_size, config.c_dim),
                                             maxshape=(None, config.image_size, config.image_size, config.c_dim),
                                             chunks=(1, config.image_size, config.image_size, config.c_dim),
                                             dtype='float32')
                # input_h5 = hf.create_dataset("input", (1, int(config.image_size/2), int(config.image_size/2), config.c_dim),
                #                              maxshape=(None, int(config.image_size/2), int(config.image_size/2), config.c_dim),
                #                              chunks=(1, int(config.image_size/2), int(config.image_size/2), config.c_dim),
                #                              dtype='float32')
                #

                label_h5 = hf.create_dataset("label", (1, config.image_size, config.image_size, config.c_dim),
                                             maxshape=(None, config.image_size, config.image_size, config.c_dim),
                                             chunks=(1, config.image_size, config.image_size, config.c_dim),
                                             dtype='float32')


            else:
                input_h5 = hf.create_dataset("input", (1, input_.shape[0], input_.shape[1], input_.shape[2]),
                                             maxshape=(None, input_.shape[0], input_.shape[1], input_.shape[2]),
                                             chunks=(1, input_.shape[0], input_.shape[1], input_.shape[2]),
                                             dtype='float32')
                label_h5 = hf.create_dataset("label", (1, label_.shape[0], label_.shape[1], label_.shape[2]),
                                             maxshape=(None, label_.shape[0], label_.shape[1], label_.shape[2]),
                                             chunks=(1, label_.shape[0], label_.shape[1], label_.shape[2]),
                                             dtype='float32')
    else:  
        hf = h5py.File(savepath, 'a')
        input_h5 = hf["input"]
        label_h5 = hf["label"]


    if config.is_train:
        # input_h5.resize([times + 1, int(config.image_size/2), int(config.image_size/2), config.c_dim])
        input_h5.resize([times + 1, config.image_size, config.image_size, config.c_dim])
        input_h5[times: times + 1] = input_
        label_h5.resize([times + 1, config.image_size, config.image_size, config.c_dim])
        label_h5[times: times + 1] = label_

    else:
        input_h5.resize([times + 1, input_.shape[0], input_.shape[1], input_.shape[2]])
        input_h5[times: times + 1] = input_
        label_h5.resize([times + 1, label_.shape[0], label_.shape[1], label_.shape[2]])
        label_h5[times: times + 1] = label_

    hf.close()
    return True


def make_sub_data(input_list, label_list, config, str):
    assert len(input_list) == len(label_list)
    times = 0  
    for i in range(len(input_list)):
        # name =  os.path.basename(input_list[i])
        ratio = float(30)#name[4:6])
        input_ = cv2.imread(input_list[i], -1)
        input_ = input_ * ratio
        label_ = cv2.imread(label_list[i], -1)

        # print(label.shape)
        assert input_.shape== label_.shape#*2

        if len(input_.shape) == 3:
            h, w, c = input_.shape
        else:
            h, w = input_.shape


        for x in range(0, h - config.image_size + 1, config.stride):
            for y in range(0, w - config.image_size + 1, config.stride):
                # x_ = int(x / 2)
                # y_ = int(y / 2)
                # sub_input = input_[x_:x_ + int(config.image_size/2), y_:y_ + int(config.image_size/2)]
                sub_input = input_[x:x + config.image_size, y:y + config.image_size]
                sub_input = sub_input / 255.0
                sub_label = label_[x:x + config.image_size, y:y + config.image_size]
                sub_label = sub_label / 255.0 

                save_flag = make_data_hf(sub_input, sub_label, config, str, times)
                if not save_flag:
                    return input_list, label_list
                times += 1
        print("image: [%2d], total: [%2d]" % (i, len(input_list)))
    return input_list, label_list



def get_data_num(path):
    with h5py.File(path, 'r') as hf:
        input_ = hf['input']
        return input_.shape[0]



def get_data_dir(is_train):#checkpoint_dir,
    if is_train:
        return os.path.join(os.path.join(os.getcwd(), "checkpoint"), 'train.h5')
    #, os.path.join(os.path.join(os.getcwd(), "checkpoint_pro"), 'eval.h5')

    else:
        return os.path.join(os.path.join(os.getcwd(), "checkpoint"), 'test.h5')


def get_batch(path, data_num, batch_size):
    with h5py.File(path, 'r')as hf:
        input_ = hf["input"]
        label_ = hf["label"]
        random_batch = np.random.rand(batch_size) * (data_num - 1)

        batch_images = np.zeros([batch_size, input_[0].shape[0], input_[0].shape[1], input_[0].shape[2]])
        batch_labels = np.zeros([batch_size, label_[0].shape[0], label_[0].shape[1], label_[0].shape[2]])

        for i in range(batch_size):
            batch_images[i, :, :, :] = np.asarray(input_[int(random_batch[i])])
            batch_labels[i, :, :, :] = np.asarray(label_[int(random_batch[i])])

        random_aug = np.random.rand(2)  
        batch_images = augmentation(batch_images, random_aug)
        batch_labels = augmentation(batch_labels, random_aug)
        return batch_images, batch_labels


def augmentation(batch, random):
    if random[0] < 0.3:

        batch_flip = np.flip(batch, 1)
    elif random[0] > 0.7:

        batch_flip = np.flip(batch, 2)
    else:

        batch_flip = batch

    if random[1] < 0.5:

        batch_rot = np.rot90(batch_flip, 1, [1, 2])
    else:

        batch_rot = batch_flip

    return batch_rot




def imsave(image, path):
    cv2.imwrite(os.path.join(os.getcwd(), path), image, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

# def save_our(config, img_s0, img_s1, img_s2, output_name):
#     path_s0 = os.path.join(config.output_dir, "our_sad")
#     if not os.path.isdir(path_s0):
#         os.mkdir(path_s0)
#     imsave(img_s0*255/2, path_s0 + '/%s_s0.png' % output_name)
#     img_s0_1 = img_s0[:,:,0]+img_s0[:,:,1]+img_s0[:,:,2]
#     img_s1_1 = img_s1[:, :, 0] + img_s1[:, :, 1] + img_s1[:, :, 2]
#     img_s2_1 = img_s2[:, :, 0] + img_s2[:, :, 1] + img_s2[:, :, 2]
#     layer_a = 1 / 2 * np.arctan2(img_s2_1, img_s1_1)  #
#     layer_aop = ((layer_a + np.pi/2) / np.pi) * 255.0  # 范围0-1
#     layer_dolp = np.divide(np.sqrt(np.add(np.square(img_s1_1), np.square(img_s2_1))) * 255,
#                            img_s0_1.astype(np.float32))  # [32,32,32,1]
#     img_good_aop = layer_aop.astype(np.uint8)
#     img_good_aop_color = cv2.applyColorMap(img_good_aop, 12)
#     cv2.imwrite(os.path.join(path_s0, ("%s_aop.png" % output_name)), img_good_aop_color,
#                     [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
#     img_good_dolp = layer_dolp.astype(np.uint8)
#     img_good_dolp_color = cv2.applyColorMap(img_good_dolp, 12)
#     cv2.imwrite(os.path.join(path_s0, ("%s_dolp.png" % output_name)), img_good_dolp_color,
#                 [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    
# def save_our(config, img_0,img_45,img_90,img_135, output_name):
#     path_s0 = os.path.join(config.output_dir, "our_sad")
#     if not os.path.isdir(path_s0):
#         os.mkdir(path_s0)
#     imsave(img_0*255*float(30), path_s0 + '/%s_0.png' % output_name)
#     imsave(img_45*255, path_s0 + '/%s_45.png' % output_name)
#     imsave(img_90*255, path_s0 + '/%s_90.png' % output_name)
#     imsave(img_135*255, path_s0 + '/%s_135.png' % output_name)


def save_our(config, img_s0, img_dolp, img_aop, output_name):
    path_s0 = os.path.join(config.output_dir, "our_sad")
    if not os.path.isdir(path_s0):
        os.mkdir(path_s0)
    imsave(img_s0*255/2, path_s0 + '/%s_s0.png' % output_name)
    aop_color_gray = cv2.applyColorMap(cv2.cvtColor(np.uint8(img_aop* 255), cv2.COLOR_RGB2GRAY), cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(path_s0, ("%s_aop.png" % output_name)), aop_color_gray)
    dop_color_gray = cv2.applyColorMap(cv2.cvtColor(np.uint8(img_dolp* 255), cv2.COLOR_RGB2GRAY), cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(path_s0, ("%s_dolp.png" % output_name)), dop_color_gray)




    
# def save_our(config, img_s0, img_dolp, img_aop, output_name):
#     path_s0 = os.path.join(config.output_dir, "our_sad")
#     if not os.path.isdir(path_s0):
#         os.mkdir(path_s0)
#     imsave(img_s0*255/2, path_s0 + '/%s_s0.png' % output_name)
#     img_aop = img_aop * 255.0
#     img_good_aop = img_aop.astype(np.uint8)
#     img_good_aop_color = cv2.applyColorMap(img_good_aop, 12)
#     cv2.imwrite(os.path.join(path_s0, ("%s_aop.png" % output_name)), img_good_aop_color,
#                     [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
#     img_dolp = img_dolp * 255.0
#     img_good_dolp = img_dolp.astype(np.uint8)
#     img_good_dolp_color = cv2.applyColorMap(img_good_dolp, 12)
#     cv2.imwrite(os.path.join(path_s0, ("%s_dolp.png" % output_name)), img_good_dolp_color,
#                 [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
