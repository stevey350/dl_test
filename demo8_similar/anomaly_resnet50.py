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
from simclr.resnet_simclr import ResNetSimCLR

def main():
    feature_size = 128
    im_size = 608
    crop_size = 120
    patch_size = 224

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    weight = 'checkpoint_0040.pth.tar'              # SimCLR表征学习训练的权重
    model = ResNetSimCLR(base_model='resnet50', out_dim=128)
    checkpoints = torch.load(r"./checkpoint_0040.pth.tar")
    model.load_state_dict(checkpoints['state_dict'])
    model.to(device)
    model.eval()
    print(model)

    # model = model_net.ResNetBased(feature_size=feature_size, im_size=patch_size)
    # model.to(device)
    # model.eval()
    # print(model)

    # weight = 'model_best.pth.tar'                 # metric learning训练的权重
    # weight = 'checkpoint.pth.tar'
    # if os.path.isfile(weight):
    #     print("=> loading checkpoint '{}'".format(weight))
    #     checkpoint = torch.load(weight)
    #     start_epoch = checkpoint['epoch']
    #     best_prec1 = checkpoint['best_prec1']
    #     model.load_state_dict(checkpoint['state_dict'])
    #     print("=> loaded checkpoint '{}' (epoch {})"
    #           .format(weight, checkpoint['epoch']))
    # else:
    #     print("=> no checkpoint found at '{}'".format(weight))

    outputs = []

    def hook(module, input, output):
        outputs.append(output)

    # model.resnet.avgpool.register_forward_hook(hook)
    model.backbone.avgpool.register_forward_hook(hook)

    # test based on toy data
    x = torch.zeros((1, 3, 224, 224))
    embedding = model(x.to(device))
    featuremap = outputs[0]
    # print('model: ', model)
    print('embedding shape: ', embedding.shape)     # [1, 128]
    print('featuremap shape', featuremap.shape)     # [1, 2048, 1, 1]

    # dataset load
    root_path = './datasets_AEresult/white_yamian_hengwen3/notqm/'      # 白色1
    img_path = os.path.join(root_path, 'image')
    lbl_path = os.path.join(root_path, 'labels')
    good_path = os.path.join(root_path, 'good')
    dataset = LoadImages(img_path, lbl_path, img_size=im_size, crop_size=crop_size, patch_size=patch_size)

    rm_cnt = 0
    for path, patch_img, patch_img0, patch_img1 in tqdm(dataset):
        # patch_img
        img = patch_img.to(device)
        # img = img.float()  # uint8 to fp32
        # img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        outputs = []
        with torch.no_grad():
            embedding = model(img).cpu().squeeze()
        featuremap = outputs[0].cpu().squeeze()
        # print('embedding shape: ', embedding.shape)     # [1, 128]
        # print('featuremap shape: ', featuremap.shape)   # [1, 2048, 1, 1]

        # patch_img0
        img0 = patch_img0.to(device)
        # img0 = img0.float()  # uint8 to fp32
        # img0 /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img0.ndimension() == 3:
            img0 = img0.unsqueeze(0)

        outputs = []
        with torch.no_grad():
            embedding0 = model(img0).cpu().squeeze()
        featuremap0 = outputs[0].cpu().squeeze()

        # patch_img1
        img1 = patch_img1.to(device)
        # img1 = img1.float()  # uint8 to fp32
        # img1 /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img1.ndimension() == 3:
            img1 = img1.unsqueeze(0)

        outputs = []
        with torch.no_grad():
            embedding1 = model(img1).cpu().squeeze()
        featuremap1 = outputs[0].cpu().squeeze()

        # 计算距离相似度 featuremap
        dist0 = calc_dist(featuremap, featuremap0)
        dist1 = calc_dist(featuremap, featuremap1)
        dist01 = calc_dist(featuremap0, featuremap1)
        dist_std = np.std(np.array([dist0.data.item(), dist1.data.item(), dist01.data.item()]))
        print(f'dist: {dist0}, {dist1}, {dist01}', f'std: {dist_std}')
        similar0 = cosine_similarity1(featuremap, featuremap0)
        similar1 = cosine_similarity1(featuremap, featuremap1)
        similar01 = cosine_similarity1(featuremap0, featuremap1)
        similar_std = np.std(np.array([similar0, similar1, similar01]))
        print(f'similar: {similar0}, {similar1}, {similar01}', f'std: {similar_std}')

        # embedding
        # dist0 = calc_dist(embedding, embedding0)
        # dist1 = calc_dist(embedding, embedding1)
        # dist01 = calc_dist(embedding0, embedding1)
        # dist_std = np.std(np.array([dist0.data.item(), dist1.data.item(), dist01.data.item()]))
        # print(f'dist: {dist0}, {dist1}, {dist01}', f'std: {dist_std}')
        # similar0 = cosine_similarity1(embedding, embedding0)
        # similar1 = cosine_similarity1(embedding, embedding1)
        # similar01 = cosine_similarity1(embedding0, embedding1)
        # similar_std = np.std(np.array([similar0, similar1, similar01]))
        # print(f'similar: {similar0}, {similar1}, {similar01}', f'std: {similar_std}')

        thred_similar = 0.80
        thred_std = 0.1
        if (similar0 > thred_similar) and (similar1 > thred_similar) and (similar01 > thred_similar) and (
                similar_std < thred_std):  #
            # shutil.copy(path, good_path)
            # shutil.move(path, good_path)
            rm_cnt += 1
    print('total number {}, remove number {}: '.format(len(dataset), rm_cnt))


def calc_dist(x, y):
    """Calculate Euclidean distance matrix with torch.tensor"""
    dist = torch.sqrt(torch.pow(x - y, 2).sum())
    return dist


if __name__ == '__main__':
    main()
