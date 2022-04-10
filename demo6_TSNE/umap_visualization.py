import os
import math
import numpy as np

from scipy.linalg import sqrtm

import umap
import matplotlib.pyplot as plt
from cycler import cycler

import torch
import torch.nn as nn

import cv2

'''
function:
    数据降维，将一个多维的数据降维到一个低维度空间上

parameter：
    @input - 输入数据，array, shape (n_samples, n_features)
    @out_dim - 降维后的输出维度

return：
    array, shape (n_samples, n_components)
'''
def v_dim_reduce(input, out_dim=2):
    reducer = umap.UMAP(random_state=42, n_components=out_dim, n_neighbors=15)
    embedding = reducer.fit_transform(input)
    print("v_dim_reduce> embedding.shape={}".format(embedding.shape))

    return embedding

'''
function:
    数据降维，将一个多维的数据降维到一个2维度空间上，然后将其可视化并保存

parameter：
    @targets - 可视化原始数据, array, shape (n_samples, n_features)
    @labels - 可视化原始数据对应的类别属性
    @save_path - 可视化图片保存目录
    @save_name - 可视化图片保存名称

return：
    无
'''
def v_dim_reduce_plot2d(targets, labels, save_path="./", save_name="umap_drv2d.jpg"):
    print("v_dim_reduce_plot2d> save_name={}".format(save_name))
    print("v_dim_reduce_plot2d> targets.shape={}".format(targets.shape))

    reducer = umap.UMAP(random_state=42, init='random')
    embedding = reducer.fit_transform(targets)
    print("v_dim_reduce_plot2d> embedding.shape={}".format(embedding.shape))

    label_set = np.unique(labels)
    num_classes = len(label_set)
    fig = plt.figure()
    # plt.gca().set_prop_cycle(cycler("color", [plt.cm.nipy_spectral(i) for i in np.linspace(0.2, 0.9, num_classes)]))

    # maxx = math.ceil(embedding[:, 0].max())
    # maxy = math.ceil(embedding[:, 1].max())
    # for i in range(num_classes):
    #     plt.plot(maxx, maxy + i, "o", markersize=30)
    #     plt.text(maxx, maxy + i, val_data_set.birdnames[i], ha='left', va='bottom', fontsize=9)

    for i in range(num_classes):
        idx = labels == label_set[i]
        # plt.plot(embedding[idx, 0], embedding[idx, 1], ".", markersize=3)
        if i == 0:
            plt.scatter(embedding[idx, 0], embedding[idx, 1], c='b', s=20, marker='o', label='good')
        else:
            plt.scatter(embedding[idx, 0], embedding[idx, 1], c='r', s=20, marker='v', label='anomaly')

    plt.legend()
    plt.savefig(os.path.join(save_path, save_name))

'''
function:
    数据降维，将一个多维的数据降维到一个2维度空间上，然后将其可视化并保存

parameter：
    @targets - 可视化原始数据, array, shape (n_samples, n_features)
    @labels - 可视化原始数据对应的类别属性
    @save_path - 可视化图片保存目录
    @save_name - 可视化图片保存名称

return：
    无
'''
def v_dim_reduce_plot3d(targets, labels, save_path="./", save_name="umap_drv3d.jpg"):
    print("v_dim_reduce_plot3d> save_name={}".format(save_name))
    print("v_dim_reduce_plot3d> targets.shape={}".format(targets.shape))

    reducer = umap.UMAP(random_state=42, n_components=3, n_neighbors=15, init='random')
    embedding = reducer.fit_transform(targets)
    print("v_dim_reduce_plot3d> embedding.shape={}".format(embedding.shape))

    label_set = np.unique(labels)
    num_classes = len(label_set)
    # fig = plt.figure(figsize=(20, 20))
    fig = plt.figure()
    # plt.gca().set_prop_cycle(cycler("color", [plt.cm.nipy_spectral(i) for i in np.linspace(0.2, 0.9, num_classes)]))

    ax = fig.add_subplot(111, projection='3d')
    for i in range(num_classes):
        idx = labels == label_set[i]
        # ax.scatter(embedding[idx, 0], embedding[idx, 1], embedding[idx, 2], c='b', s=2, marker='o')

        if i == 0:
            ax.scatter(embedding[idx, 0], embedding[idx, 1], embedding[idx, 2], c='b', s=20, marker='o', label='good')
        else:
            ax.scatter(embedding[idx, 0], embedding[idx, 1], embedding[idx, 2], c='r', s=20, marker='v', label='anomaly')

    ax.legend()
    plt.savefig(os.path.join(save_path, save_name))



'''
function:
    数据降维，将一个多维的数据降维到一个2维度空间上，然后将其可视化并保存
    针对两个目标之间的特征比较

parameter：
    @target1 - 可视化原始数据, array, shape (n_samples, n_features)
    @target2 - 可视化原始数据, array, shape (n_samples, n_features)
    @save_path - 可视化图片保存目录
    @save_name - 可视化图片保存名称

return：
    无
'''
def v_dim_reduce_plot2d_2in(target1, target2, save_path="./", save_name="umap_drv_compare.jpg"):
    print("v_dim_reduce_plot2d_2in> target1.shape={}, target2.shape={}".format(target1.shape, target2.shape))

    reducer = umap.UMAP(random_state=42, init='random', n_neighbors=50)
    embedding1 = reducer.fit_transform(target1)
    embedding2 = reducer.fit_transform(target2)
    print("v_dim_reduce_plot2d_2in> embedding1.shape={}, embedding2.shape={}".format(embedding1.shape, embedding2.shape))

    fig = plt.figure(figsize=(20, 5))
    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)

    plt.subplot(131)
    plt.plot(embedding1[:, 0], embedding1[:, 1], ".", markersize=3, color='red')
    plt.plot(embedding2[:, 0], embedding2[:, 1], ".", markersize=3, color='blue')

    dismap = np.sqrt((embedding1[:, 0] - embedding2[:, 0]) ** 2 + (embedding1[:, 1] - embedding2[:, 1]) ** 2)
    # dismap = dismap.reshape(64, 64)

    # dismap = (embedding1[:, 0] - embedding2[:, 0]) ** 2 + (embedding1[:, 1] - embedding2[:, 1]) ** 2
    # dismap = dismap.reshape(64, 64)
    # dismap = cv2.resize(dismap, dsize=(640, 640), interpolation=cv2.INTER_LINEAR)
    print("v_dim_reduce_plot2d_2in> dismap.shape={}".format(dismap.shape))

    # 画散点图
    plt.subplot(132)
    x = np.linspace(0, 200, 4096)
    plt.plot(x, dismap, ".", markersize=3, color='blue')

    plt.subplot(133)
    # 画热力图
    dismap = dismap.reshape(64, 64)
    plt.imshow(dismap, vmax=dismap.max(), vmin=dismap.min())
    plt.title('Predicted max=%f' % dismap.max())

    plt.savefig(os.path.join(save_path, save_name))

    fig.clf()


# def save_score_result(self, img, path, img_name, scores):
#     # binary score
#     max_val = scores.max()
#     min_val = scores.min()
#     # print('max_val:%f, min_val:%f' % (max_val, min_val))
#     # plt.imsave(os.path.join(path, "{}".format(img_name)), scores, vmax=max_val, vmin=min_val)

#     fig_img, ax_img = plt.subplots(1, 2, figsize=(12, 4))
#     fig_img.subplots_adjust(left=0, right=0.9, bottom=0, top=0.9)

#     for ax_i in ax_img:
#         ax_i.axes.xaxis.set_visible(False)
#         ax_i.axes.yaxis.set_visible(False)

#     img = np.squeeze(img)
#     img = self.denormalization(img)
#     ax_img[0].imshow(img)
#     ax_img[0].title.set_text('Image')
#     ax_img[1].imshow(scores, vmax=max_val, vmin=min_val)
#     ax_img[1].title.set_text('Predicted anomalous image, max=%f' % max_val)
#     # ax_img[1].set_title('title test', fontsize=12, color='r')

#     fig_img.savefig\
#         (os.path.join(path, "{}".format(img_name)),  dpi=100)
#     fig_img.clf()
#     plt.close(fig_img)

'''
function:
    数据降维，将一个多维的数据降维到一个2维度空间上，然后将其可视化并保存
    针对两个目标之间的特征比较

parameter：
    @target1 - 可视化原始数据, array, shape (n_samples, n_features)
    @target2 - 可视化原始数据, array, shape (n_samples, n_features)
    @save_path - 可视化图片保存目录
    @save_name - 可视化图片保存名称

return：
    无
'''
def v_dim_reduce_plot3d_2in(target1, target2, save_path="./", save_name="umap_drv_compare.jpg"):
    print("v_dim_reduce_plot3d_2in> target1.shape={}, target2.shape={}".format(target1.shape, target2.shape))

    reducer = umap.UMAP(random_state=42, init='random', n_neighbors=50, n_components=3)
    embedding1 = reducer.fit_transform(target1)
    embedding2 = reducer.fit_transform(target2)
    print("v_dim_reduce_plot3d_2in> embedding1.shape={}, embedding2.shape={}".format(embedding1.shape, embedding2.shape))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(embedding1[:, 0], embedding1[:, 1], embedding1[:, 2], c='r', s=2, marker='o')
    # bx = fig.add_subplot(122, projection='3d')
    ax.scatter(embedding2[:, 0], embedding2[:, 1], embedding2[:, 2], c='b', s=2, marker='o')

    plt.savefig(os.path.join(save_path, save_name))

    fig.clf()
    
'''
function:
    数据降维，将一个多维的数据降维到一个2维度空间上，然后将其可视化并保存
    针对两个目标之间的特征比较

parameter：
    @targets - 可视化原始数据, list, shape (b, w, h)
    @save_path - 可视化图片保存目录
    @save_name - 可视化图片保存名称

return：
    无
'''
def v_plot2d_nin(targets, save_path="./", save_name="ae_plot.jpg"):
    # print("v_plot2d_nin> len={}, targetsX.shape={}".format(len(targets), targets[0].shape))

    n = len(targets)
    b = targets[0].shape[0]
    
    merge = np.concatenate(targets, 0)
    max_val = merge.max()
    min_val = merge.min()

    # print("v_plot2d_nin> n={}, b={}, max_val={}, min_val={}".format(n, b, max_val, min_val))

    fig = plt.figure(figsize=(20, 40))

    for bi in range(b):
        for ni in range(n):
            plt.subplot(b, n, bi * n + ni + 1)
            plt.gca().axes.xaxis.set_visible(False)
            plt.gca().axes.yaxis.set_visible(False)
            plt.gca().axes.title.set_text('min=%.2f, max=%.2f' % (targets[ni][bi, :, :].min(), targets[ni][bi, :, :].max()))
            plt.imshow(targets[ni][bi, :, :], vmax=max_val, vmin=min_val)

    plt.savefig(os.path.join(save_path, save_name))

    fig.clf()
    plt.close()