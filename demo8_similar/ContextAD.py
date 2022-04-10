import torch
import torch.nn as nn
import torchvision.models as models
from MVTec.loadImages import LoadImages
from torch.utils.data import DataLoader
import numpy as np
import os
from similar_cosine import cosine_similarity1
from tqdm import tqdm
import shutil
from efficientnet.model import EfficientNet
import time
import matplotlib.pyplot as plt


class ContextAD():
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.device       # torch.device(
        self.class_num = cfg.class_num

        # 
        self.img_size = cfg.img_size
        self.crop_size = cfg.crop_size
        self.patch_size = cfg.patch_size

        # model
        self.model = EfficientNet.from_name(cfg.net_name)

        num_ftrs = self.model._fc.in_features  # b7: 2560  # 修改全连接层
        self.model._fc = nn.Linear(num_ftrs, self.class_num)

        net_weight = os.path.join(cfg.wight_path, cfg.net_name + cfg.weight_version + '.pth')   # efficientnet-b0_v1.0.pth'
        best_model_wts = torch.load(net_weight)
        self.model.load_state_dict(best_model_wts.state_dict())
        self.model.to(self.device)
        self.model.eval()
        # print(model)


    def infer(self, infer_path, lbl_path, filter_out_path, thred_similar, thred_std):

        outputs = []
        def hook(module, input, output):
            outputs.append(output)
        self.model._avg_pooling.register_forward_hook(hook)

        # test based on toy data
        x = torch.zeros((1, 3, 224, 224))
        embedding = self.model(x.to(self.device))
        featuremap = outputs[0]

        # print('model: ', model)
        print('embedding shape: ', embedding.shape)         # [1, 2]
        print('feature_map', featuremap.shape)              # b7: [1, 2560, 1, 1]  b0: [1, 1280, 1, 1]

        # good_path = './datasets_AEresult/white/good'
        dataset = LoadImages(infer_path, lbl_path, img_size=self.img_size, crop_size=self.crop_size, patch_size=self.patch_size)

        # if not os.path.exists(good_path):
        #     os.makedirs(good_path)
        # else:
        #     del_file_by_path(good_path)

        rm_num = 0
        for path, patch_img, patch_img0, patch_img1 in tqdm(dataset):
            # print('filename: ', path)
            time_start = time.time()
            # patch_img
            img = patch_img.to(self.device).unsqueeze(0)
            img0 = patch_img0.to(self.device).unsqueeze(0)
            img1 = patch_img1.to(self.device).unsqueeze(0)

            # inference
            outputs = []
            imgs = torch.cat((img, img0, img1), dim=0)
            with torch.no_grad():
                preds = self.model(imgs).cpu()
            featuremaps = outputs[0].cpu()

            # probability
            prob = torch.softmax(preds[0].unsqueeze(0), dim=1).tolist()[0]
            prob0 = torch.softmax(preds[1].unsqueeze(0), dim=1).tolist()[0]
            prob1 = torch.softmax(preds[2].unsqueeze(0), dim=1).tolist()[0]

            # featuremaps
            featuremap = featuremaps[0].squeeze()
            featuremap0 = featuremaps[1].squeeze()
            featuremap1 = featuremaps[2].squeeze()

            time_end1 = time.time()

            # 计算距离相似度
            # dist0 = calc_dist(torch.flatten(featuremap), torch.flatten(featuremap0))
            # dist1 = calc_dist(torch.flatten(featuremap), torch.flatten(featuremap1))
            # dist01 = calc_dist(torch.flatten(featuremap0), torch.flatten(featuremap1))
            # dist_std = np.std(np.array([dist0.data.item(), dist1.data.item(), dist01.data.item()]))
            # print(f'dist: {dist0}, {dist1}, {dist01}', f'std: {dist_std}')
            # print('Anomlay prob:', prob[0], prob0[0], prob1[0])
            similar0 = cosine_similarity1(torch.flatten(featuremap), torch.flatten(featuremap0))
            similar1 = cosine_similarity1(torch.flatten(featuremap), torch.flatten(featuremap1))
            similar01 = cosine_similarity1(torch.flatten(featuremap0), torch.flatten(featuremap1))
            similar_std = np.std(np.array([similar0, similar1, similar01]))
            # print(f'similar: {similar0}, {similar1}, {similar01}', f'std: {similar_std}')
            # time_end2 = time.time()
            # # print('time1: ', time_end - time_start)
            # print('time2: ', time_end1 - time_start)
            # print('time3: ', time_end2 - time_end1)
            # a = prob[1]
            # b = prob0[1]
            # c = prob1[1]
            # thred_prob = 0.98
            # if (prob[1] > thred_prob) and (prob0[1] > thred_prob) and (prob1[1] > thred_prob):
            #     rm_cnt += 1

            # thred_similar = 0.9
            # thred_std = 0.1
            if (similar0 > thred_similar) and (similar1 > thred_similar) and (similar01 > thred_similar) and (similar_std < thred_std):   #
                # shutil.copy(path, filter_out_path)
                # shutil.move(path, filter_out_path)
                rm_num += 1
        
        return len(dataset), rm_num
        


    def del_file_by_path(self, path):
        for i in os.listdir(path):
            path_file = os.path.join(path, i)
            if os.path.isfile(path_file):
                os.remove(path_file)
            else:
                for f in os.listdir(path_file):
                    path_file2 = os.path.join(path_file, f)
                    if os.path.isfile(path_file2):
                        os.remove(path_file2)


    def calc_dist(self, x, y):
        """Calculate Euclidean distance matrix with torch.tensor"""
        dist = torch.sqrt(torch.pow(x - y, 2).sum())
        return dist

    def thred_estimate_with_fr(self, eval_path, expect_fr=0.75):
        outputs = []
        def hook(module, input, output):
            outputs.append(output)
        self.model._avg_pooling.register_forward_hook(hook)

        # test based on toy data
        x = torch.zeros((1, 3, 224, 224))
        embedding = self.model(x.to(self.device))
        featuremap = outputs[0]

        # Note: lbl_path = None for eval_path
        dataset = LoadImages(eval_path, lbl_path=None, img_size=self.img_size, crop_size=self.crop_size, patch_size=self.patch_size)
        similar_list0 = []
        similar_list1 = []
        similar_list01 = []
        std_list = []
        for path, patch_img, patch_img0, patch_img1 in tqdm(dataset):
            # print('filename: ', path)
            time_start = time.time()
            # patch_img
            img = patch_img.to(self.device).unsqueeze(0)
            img0 = patch_img0.to(self.device).unsqueeze(0)
            img1 = patch_img1.to(self.device).unsqueeze(0)

            # inference
            outputs = []
            imgs = torch.cat((img, img0, img1), dim=0)
            with torch.no_grad():
                preds = self.model(imgs).cpu()
            featuremaps = outputs[0].cpu()

            # probability
            prob = torch.softmax(preds[0].unsqueeze(0), dim=1).tolist()[0]
            prob0 = torch.softmax(preds[1].unsqueeze(0), dim=1).tolist()[0]
            prob1 = torch.softmax(preds[2].unsqueeze(0), dim=1).tolist()[0]

            # featuremaps
            featuremap = featuremaps[0].squeeze()
            featuremap0 = featuremaps[1].squeeze()
            featuremap1 = featuremaps[2].squeeze()

            time_end1 = time.time()

            # 计算距离相似度
            similar0 = cosine_similarity1(torch.flatten(featuremap), torch.flatten(featuremap0))
            similar1 = cosine_similarity1(torch.flatten(featuremap), torch.flatten(featuremap1))
            similar01 = cosine_similarity1(torch.flatten(featuremap0), torch.flatten(featuremap1))
            similar_std = np.std(np.array([similar0, similar1, similar01]))

            similar_list0.append(similar0.item())
            similar_list1.append(similar1.item())
            similar_list01.append(similar01.item())
            std_list.append(similar_std)
        
        K = int(len(dataset)*(1-expect_fr))
        print("K=", K)
        similar0_topk, _ = torch.topk(torch.Tensor(similar_list0), K, largest=False)        # 最小K个值
        similar1_topk, _ = torch.topk(torch.Tensor(similar_list1), K, largest=False)
        similar01_topk, _ = torch.topk(torch.Tensor(similar_list01), K, largest=False)
        std_topk, _ = torch.topk(torch.Tensor(std_list), K)     # 最大K个值

        # print("std_topk: ", std_topk[-20:-1])
        # print("similar0_topk: ", similar0_topk[-20:-1])
        # print("similar1_topk: ", similar1_topk[-20:-1])
        # print("similar01_topk: ", similar01_topk[-20:-1])

        thred_similar = (similar0_topk[-1] + similar1_topk[-1] + similar01_topk[-1])/3
        thred_std = std_topk[-1]

        plt.figure()
        n, bins, patches = plt.hist(x=similar_list0, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
        print("n={}, bins={}, len={}".format(n, bins, len(n)) )
        for i in range(len(n)):
            plt.text(bins[i]*1.0, n[i]*1.01, int(n[i]))  # 打标签，在合适的位置标注每个直方图上面样本数
        plt.show()


        return thred_similar, thred_std


if __name__ == '__main__':
    # main()
    pass
