import torch
import torchvision.models as models
from tqdm import tqdm
import MVTec.mvtec as mvtec
from torch.utils.data import DataLoader
import numpy as np
from skimage.metrics import structural_similarity
from similar_cosine import cosine_similarity1
class_name = 'all_yamian_hengwen3'     # all_yamian_hengwen3  gray_yamian_hengwen3
# class_name = 'gray_sj'

if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # model = models.resnet18(pretrained=True)
    model = models.wide_resnet50_2(pretrained=True, progress=True)
    model.to(device)
    model.eval()
    # print(model)

    outputs = []

    def hook(module, input, output):
        outputs.append(output)

    model.avgpool.register_forward_hook(hook)

    # test based on toy data
    x = torch.zeros((1, 3, 224, 224))
    y = model(x.to(device))
    embedding = outputs[0]
    outputs = []
    print('model: ', model)
    print('embedding shape: ', embedding.shape)
    print('x shape', torch.squeeze(embedding).shape)
    print('y shape: ', y.shape)

    ref_dataset = mvtec.MVTecDataset(class_name=class_name, is_train=True)      # load data from train.good
    ref_dataloader = DataLoader(ref_dataset, batch_size=1, pin_memory=True)
    test_dataset = mvtec.MVTecDataset(class_name=class_name, is_train=False)    # load data from test.anomaly and test.good
    test_dataloader = DataLoader(test_dataset, batch_size=1, pin_memory=True)

    ref_imgs_list = []
    ref_embedding_list = []
    for (x, y) in tqdm(ref_dataloader, '| feature extraction | reference | %s |' % class_name):
        ref_img = x.cpu().detach().numpy()
        ref_img = ref_img.squeeze().transpose(1, 2, 0)
        ref_imgs_list.append(ref_img)

        with torch.no_grad():
            pred = model(x.to(device))

        # get intermediate layer outputs
        ref_embedding_list.append(outputs[0])

        # initialize hook outputs
        outputs = []
    print('ref_embedding_list length: ', len(ref_embedding_list))

    for i, (x, y) in enumerate(test_dataloader):
        if i == 192:    # 337
            # break
            print("index = ", i)

        test_img = x.cpu().detach().numpy()         # [1, 3, 160, 160]
        test_img = test_img.squeeze().transpose(1, 2, 0)    # [160, 160, 3]
        # model prediction
        with torch.no_grad():
            pred = model(x.to(device))
        # get intermediate layer outputs
        test_embedding = outputs[0]
        # initialize hook outputs
        outputs = []

        # calculate the cosine similar score
        cosine_similar_score_list = []
        for ref_embedding in ref_embedding_list:
            test_embedding = torch.squeeze(test_embedding).cpu()
            ref_embedding = torch.squeeze(ref_embedding).cpu()
            cosine_similar_score_list.append(cosine_similarity1(test_embedding, ref_embedding))
        print('cosine similar score: ', np.mean(cosine_similar_score_list))

        # calculate the ssim similar score
        # ssim_similar_score_list = []
        # for ref_img in ref_imgs_list:
        #     ssim_similar_score = structural_similarity(ref_img, test_img, multichannel=True, gaussian_weights=True,
        #                                                sigma=1.5,
        #                                                use_sample_covariance=False, data_range=1.0)
        #     ssim_similar_score_list.append(ssim_similar_score)
        # print('ssim similar score: ', np.mean(ssim_similar_score_list))
