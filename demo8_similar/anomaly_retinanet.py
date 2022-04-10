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
from retinanet.model import resnet18, resnet50, resnet152, resnet34, resnet101

def main():

    im_size = 640
    crop_size = 224
    patch_size = 224

    pretrainWeight = "./retinanet/weights/colorjit_retinanet.pt"

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    half = device != 'cpu'

    # 载入后，直接在cuda?
    model = torch.load(pretrainWeight)
    cnn_layers = ('P3_x', 'P4_x', 'P5_x')
    model.eval()
    # print(feature)

    # test based on toy data
    x = torch.zeros((1, 3, 224, 224))
    feature_maps = model(x.to(device), feature_layers=cnn_layers)
    # print('model: ', model)
    print('embedding shape: ', len(feature_maps))                   # [1, 128]
    print('featuremap shape', feature_maps['P3_x'].shape, feature_maps['P4_x'].shape, feature_maps['P5_x'].shape)           # [1, 2048, 1, 1]
    feature_map = torch.mean(feature_maps['P3_x'], dim=1)
    print('feature_map', feature_map.shape)

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
        # patch_img
        img = torch.from_numpy(patch_img).to(device)
        img = img.float()  # uint8 to fp32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        featuremap = model(img, feature_layers=cnn_layers)['P3_x']
        featuremap = torch.mean(featuremap, dim=1).cpu().detach()
        print('featuremap shape: ', featuremap.shape)   # [1, 28, 28]

        # patch_img0
        img0 = torch.from_numpy(patch_img0).to(device)
        img0 = img0.float()  # uint8 to fp32
        img0 /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img0.ndimension() == 3:
            img0 = img0.unsqueeze(0)

        featuremap0 = model(img0, feature_layers=cnn_layers)['P3_x']
        featuremap0 = torch.mean(featuremap0, dim=1).cpu().detach()

        # patch_img1
        img1 = torch.from_numpy(patch_img1).to(device)
        img1 = img1.float()  # uint8 to fp32
        img1 /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img1.ndimension() == 3:
            img1 = img1.unsqueeze(0)

        featuremap1 = model(img1, feature_layers=cnn_layers)['P3_x']
        featuremap1 = torch.mean(featuremap1, dim=1).cpu().detach()
        print('featuremap1 flatten', torch.flatten(featuremap1).shape)

        # 计算距离相似度
        dist0 = calc_dist(torch.flatten(featuremap), torch.flatten(featuremap0))
        dist1 = calc_dist(torch.flatten(featuremap), torch.flatten(featuremap1))
        dist01 = calc_dist(torch.flatten(featuremap0), torch.flatten(featuremap1))
        dist_std = np.std(np.array([dist0.data.item(), dist1.data.item(), dist01.data.item()]))
        print(f'dist: {dist0}, {dist1}, {dist01}', f'std: {dist_std}')
        similar0 = cosine_similarity1(torch.flatten(featuremap), torch.flatten(featuremap0))
        similar1 = cosine_similarity1(torch.flatten(featuremap), torch.flatten(featuremap1))
        similar01 = cosine_similarity1(torch.flatten(featuremap0), torch.flatten(featuremap1))
        # similar0 = cosine_similarity1(embedding.numpy(), embedding0.numpy())
        # similar1 = cosine_similarity1(embedding.numpy(), embedding1.numpy())
        # similar01 = cosine_similarity1(embedding0.numpy(), embedding1.numpy())
        similar_std = np.std(np.array([similar0, similar1, similar01]))
        print(f'similar: {similar0}, {similar1}, {similar01}', f'std: {similar_std}')

        thred_dist = 0.26
        thred_std = 0.026
        if (dist_std < thred_std) and (dist0 < thred_dist) and (dist1 < thred_dist):   #
            # shutil.copy(path, good_path)
            rm_cnt += 1
    print('total number {}, remove number {}: '.format(len(dataset), rm_cnt))


def calc_dist(x, y):
    """Calculate Euclidean distance matrix with torch.tensor"""
    dist = torch.sqrt(torch.pow(x - y, 2).sum())
    return dist


if __name__ == '__main__':
    main()
