import tensorflow as tf
import numpy as np
import time
from tensorflow.contrib import slim
from utils import *

# ========== 激活函数 ==========
def lrelu(x, alpha=0.2):
    return tf.maximum(alpha * x, x)

# ========== 评估指标 ==========
def psnr(pred, gt):
    mse = tf.reduce_mean(tf.square(pred - gt))
    return 20 * tf.log(1.0 / tf.sqrt(mse)) / tf.log(10.0)

def ssim(img1, img2):
    return tf.image.ssim(img1, img2, max_val=1.0)

# ========== 模型类（用你的RDN定义） ==========
from model_s_2 import RDN # 假设你把RDN类存到 your_rdn_file.py

# ========== 统计参数数量 ==========
print(1)
def count_parameters():
    return np.sum([np.prod(v.shape.as_list()) for v in tf.trainable_variables()])

# ========== FLOPs 计算 ==========
def get_flops(sess, graph):
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    return flops.total_float_ops

# ========== 测试推理时间 ==========
def measure_inference_time(sess, output_tensor, feed_dict, runs=50):
    # 预热
    for _ in range(10):
        sess.run(output_tensor, feed_dict=feed_dict)

    start = time.time()
    for _ in range(runs):
        sess.run(output_tensor, feed_dict=feed_dict)
    end = time.time()

    return (end - start) / runs


if __name__ == "__main__":
    tf.reset_default_graph()

    # 模型输入大小
    batch_size = 1
    image_size = 128
    c_dim = 3

    # 输入占位符
    images = tf.placeholder(tf.float32, [batch_size, image_size, image_size, c_dim], name="images")
    labels = tf.placeholder(tf.float32, [batch_size, image_size, image_size, c_dim], name="labels")

    # 创建 Session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # 构建模型
    model = RDN(sess=sess,
                is_train=False,
                is_eval=True,
                image_size=image_size,
                c_dim=c_dim,
                batch_size=batch_size,
                D=3, C=3, G=32,  # 参数可调
                P_dim=64, Pc_dim=3, PD=3, PC=3, PG=32,
                kernel_size=3)

    model.images = images
    model.labels = labels
    output = model.model()  # 构建计算图

    # 初始化
    sess.run(tf.global_variables_initializer())

    # # 打印参数数量
    # print("Total Parameters: {:,}".format(count_parameters()))

    # 打印 FLOPs
    flops = get_flops(sess, tf.get_default_graph())
    print("Total FLOPs: {:,}".format(flops))

    # 打印参数数量
    print("Total Parameters: {:,}".format(count_parameters()))


    # 随机输入测试推理时间
    dummy_input = np.random.rand(batch_size, image_size, image_size, c_dim).astype(np.float32)
    dummy_label = np.random.rand(batch_size, image_size, image_size, c_dim).astype(np.float32)
    inference_time = measure_inference_time(sess, output, {images: dummy_input, labels: dummy_label})
    print("Average Inference Time: {:.6f} sec".format(inference_time))



# import tensorflow as tf
# from tensorflow.contrib import slim

# # -------------------------------
# # 单个轻量化 MSRB (L-MSRB)
# # -------------------------------
# def L_MSRB(input_layer, G=64, lrelu=tf.nn.leaky_relu, name='L_MSRB'):
#     with tf.variable_scope(name):
#         # 第一阶段
#         conv_3 = slim.separable_conv2d(input_layer, G, [3,3], rate=1, activation_fn=lrelu,
#                                        depth_multiplier=1,
#                                        weights_initializer=tf.contrib.layers.variance_scaling_initializer())
#         conv_5 = slim.separable_conv2d(input_layer, G, [5,5], rate=1, activation_fn=lrelu,
#                                        depth_multiplier=1,
#                                        weights_initializer=tf.contrib.layers.variance_scaling_initializer())
#         concat1 = tf.concat([conv_3, conv_5], axis=3)

#         # 第二阶段
#         conv_3_2 = slim.separable_conv2d(concat1, G*2, [3,3], rate=1, activation_fn=lrelu,
#                                         depth_multiplier=1,
#                                         weights_initializer=tf.contrib.layers.variance_scaling_initializer())
#         conv_5_2 = slim.separable_conv2d(concat1, G*2, [5,5], rate=1, activation_fn=lrelu,
#                                         depth_multiplier=1,
#                                         weights_initializer=tf.contrib.layers.variance_scaling_initializer())
#         concat2 = tf.concat([conv_3_2, conv_5_2], axis=3)

#         # 通道融合
#         fused = slim.conv2d(concat2, G, [1,1], activation_fn=None,
#                             weights_initializer=tf.contrib.layers.variance_scaling_initializer())
#         fused_feat = slim.conv2d(fused, G, [1,1], activation_fn=None,
#                                  weights_initializer=tf.contrib.layers.variance_scaling_initializer())

#         # 残差连接
#         out = tf.add(fused_feat, input_layer)
#         return out

# # -------------------------------
# # 单个原始 MSRB
# # -------------------------------
# def MSRB(input_layer, G=64, lrelu=tf.nn.leaky_relu, name='MSRB'):
#     with tf.variable_scope(name):
#         # 第一阶段
#         conv_3 = slim.conv2d(input_layer, G, [3,3], rate=1, activation_fn=lrelu,
#                              weights_initializer=tf.contrib.layers.variance_scaling_initializer())
#         conv_5 = slim.conv2d(input_layer, G, [5,5], rate=1, activation_fn=lrelu,
#                              weights_initializer=tf.contrib.layers.variance_scaling_initializer())
#         concat1 = tf.concat([conv_3, conv_5], axis=3)

#         # 第二阶段
#         conv_3_2 = slim.conv2d(concat1, G*2, [3,3], rate=1, activation_fn=lrelu,
#                                weights_initializer=tf.contrib.layers.variance_scaling_initializer())
#         conv_5_2 = slim.conv2d(concat1, G*2, [5,5], rate=1, activation_fn=lrelu,
#                                weights_initializer=tf.contrib.layers.variance_scaling_initializer())
#         concat2 = tf.concat([conv_3_2, conv_5_2], axis=3)

#         # 通道融合
#         fused = slim.conv2d(concat2, G, [1,1], activation_fn=None,
#                             weights_initializer=tf.contrib.layers.variance_scaling_initializer())
#         fused_feat = slim.conv2d(fused, G, [1,1], activation_fn=None,
#                                  weights_initializer=tf.contrib.layers.variance_scaling_initializer())

#         # 残差连接
#         out = tf.add(fused_feat, input_layer)
#         return out

# # -------------------------------
# # 测试单个模块参数量和 FLOPs
# # -------------------------------
# if __name__ == "__main__":
#     tf.reset_default_graph()
#     input_tensor = tf.placeholder(tf.float32, [1, 64, 64, 64])

#     msrb_out = MSRB(input_tensor)
#     # l_msrb_out = L_MSRB(input_tensor)

#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
        
#         # FLOPs
#         profiler = tf.compat.v1.profiler.profile(
#             sess.graph,
#             options=tf.compat.v1.profiler.ProfileOptionBuilder.float_operation())
#         print("Total FLOPs (all modules): ", profiler.total_float_ops)

#         # 参数量
#         total_params = sum([tf.size(v).eval() for v in tf.trainable_variables() if 'MSRB' in v.name])
#         # total_params_lw = sum([tf.size(v).eval() for v in tf.trainable_variables() if 'L_MSRB' in v.name])
#         print("MSRB Params: ", total_params)
#         # print("L-MSRB Params: ", total_params_lw)


