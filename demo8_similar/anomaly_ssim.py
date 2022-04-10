import cv2
import torch
import torchvision.models as models
from tqdm import tqdm
import MVTec.mvtec as mvtec
from MVTec.loadImages import LoadImages
from torch.utils.data import DataLoader
import numpy as np
import os
import model_net
# from skimage.metrics import structural_similarity
from similar_cosine import cosine_similarity1
from tqdm import tqdm
import shutil
from skimage.metrics import structural_similarity
import matplotlib.pyplot as plt

def main():
    feature_size = 128
    im_size = 640
    crop_size = 160
    patch_size = 160

    # dataset load
    # root_path = './datasets_AEresult/gray/part5/'
    root_path = './datasets_AEresult/white/notqm/'
    # root_path = './datasets_AEresult/white/qmandother/'
    # root_path = './datasets_AEresult/white/stretch/'
    img_path = os.path.join(root_path, 'image')
    lbl_path = os.path.join(root_path, 'labels')
    good_path = os.path.join(root_path, 'good')
    dataset = LoadImages(img_path, lbl_path, img_size=im_size, crop_size=crop_size, patch_size=patch_size)

    rm_cnt = 0
    for path, patch_img, patch_img0, patch_img1 in tqdm(dataset):
        patch_img = patch_img.transpose(1, 2, 0)
        patch_img0 = patch_img0.transpose(1, 2, 0)
        patch_img1 = patch_img1.transpose(1, 2, 0)

        patch_img = cv2.cvtColor(patch_img, cv2.COLOR_RGB2GRAY)
        patch_img0 = cv2.cvtColor(patch_img0, cv2.COLOR_RGB2GRAY)
        patch_img1 = cv2.cvtColor(patch_img1, cv2.COLOR_RGB2GRAY)

        plt.imshow(patch_img, cmap=plt.cm.gray)
        plt.show()
        plt.imshow(patch_img0, cmap=plt.cm.gray)
        plt.show()
        plt.imshow(patch_img1, cmap=plt.cm.gray)
        plt.show()

        # cv2.imshow('patch_img', patch_img)
        # cv2.imshow('patch_img0', patch_img0)
        # cv2.imshow('patch_img1', patch_img1)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # structural_similarity
        # similar0 = structural_similarity(patch_img, patch_img0, multichannel=True, gaussian_weights=True,
        #                                         sigma=1.5,
        #                                         use_sample_covariance=False, data_range=255)
        # similar1 = structural_similarity(patch_img, patch_img1, multichannel=True, gaussian_weights=True,
        #                                         sigma=1.5,
        #                                         use_sample_covariance=False, data_range=255)
        # similar01 = structural_similarity(patch_img0, patch_img1, multichannel=True, gaussian_weights=True,
        #                                         sigma=1.5,
        #                                         use_sample_covariance=False, data_range=255)
        similar0 = structural_similarity(patch_img, patch_img0, win_size=11)
        similar1 = structural_similarity(patch_img, patch_img1, win_size=11)
        similar01 = structural_similarity(patch_img0, patch_img1, win_size=11)
        similar_std = np.std(np.array([similar0, similar1, similar01]))
        print(f'similar: {similar0}, {similar1}, {similar01}', f'std: {similar_std}')

        thred_dist = 2.5
        thred_std = 0.25
        # if (dist_std < thred_std) and (dist0 < thred_dist) and (dist1 < thred_dist):   #
        #     # shutil.copy(path, good_path)
        #     rm_cnt += 1
    print('total number {}, remove number {}: '.format(len(dataset), rm_cnt))


def calc_dist(x, y):
    """Calculate Euclidean distance matrix with torch.tensor"""
    dist = torch.sqrt(torch.pow(x - y, 2).sum())
    return dist


if __name__ == '__main__':
    main()
