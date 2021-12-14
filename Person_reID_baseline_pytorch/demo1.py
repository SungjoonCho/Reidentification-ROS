import argparse
import scipy.io
import torch
import numpy as np
import os
from torchvision import datasets
import matplotlib
# matplotlib.use('agg')
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
#######################################################################
# Evaluate
parser = argparse.ArgumentParser(description='Demo')
parser.add_argument('--query_index', default=777, type=int, help='test_image_index')
parser.add_argument('--test_dir',default='../Market-1501/pytorch',type=str, help='./test_data')
opts = parser.parse_args()

# data_dir = opts.test_dir
image_datasets = {'gallery' : datasets.ImageFolder('./runs1')}

#####################################################################
#Show result
def imshow(path, title=None):
    """Imshow for Tensor."""
    im = plt.imread(path)
    plt.imshow(im)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

######################################################################
result = scipy.io.loadmat('output1_extract_result.mat')
# query_feature = torch.FloatTensor(result['query_f'])
# query_cam = result['query_cam'][0]
# query_label = result['query_label'][0]

gallery_feature = torch.FloatTensor(result['img_f'])

# gallery_cam = result['gallery_cam'][0]
# gallery_label = result['gallery_label'][0]

query_feature = gallery_feature[2] # 여기서 쿼리 번호 지정

query_feature = query_feature.cuda()
gallery_feature = gallery_feature.cuda()

# #######################################################################
# # sort the images
# def sort_img(qf, ql, qc, gf, gl, gc):
#     query = qf.view(-1,1)
#     # print(query.shape)
#     score = torch.mm(gf,query)
#     score = score.squeeze(1).cpu()
#     score = score.numpy()
#     # predict index
#     index = np.argsort(score)  #from small to large
#     index = index[::-1]
#     # index = index[0:2000]
#     # good index
#     query_index = np.argwhere(gl==ql)
#     #same camera
#     camera_index = np.argwhere(gc==qc)

#     #good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
#     junk_index1 = np.argwhere(gl==-1)
#     junk_index2 = np.intersect1d(query_index, camera_index)
#     junk_index = np.append(junk_index2, junk_index1) 

#     mask = np.in1d(index, junk_index, invert=True)
#     index = index[mask]
#     return index

def sort_img(qf, gf):
    query = qf.view(-1,1)
    # print(query.shape)
    score = torch.mm(gf,query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    # print(gf.shape, qf.shape, score.shape)
    # print(score, end = '\n')

    # predict index
    index = np.argsort(score)  #from small to large
    index = index[::-1]
    # index = index[0:2000]
    # print(score)
    # print(index)


    # # good index
    # query_index = np.argwhere(gl==ql)
    # #same camera
    # camera_index = np.argwhere(gc==qc)

    # #good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    # junk_index1 = np.argwhere(gl==-1)
    # junk_index2 = np.intersect1d(query_index, camera_index)
    # junk_index = np.append(junk_index2, junk_index1) 

    # mask = np.in1d(index, junk_index, invert=True)
    # index = index[mask]
    return score, index

# i = opts.query_index
# index = sort_img(query_feature[i],query_label[i],query_cam[i],gallery_feature,gallery_label,gallery_cam)
score, index = sort_img(query_feature, gallery_feature)

# ########################################################################
# # Visualize the rank result

# query_path, _ = image_datasets['query'].imgs[i]
# query_label = query_label[i]
# print(query_path)
print('images are as follow:')
# try: # Visualize Ranking Result 
#     # Graphical User Interface is needed
#     fig = plt.figure(figsize=(16,4))
#     ax = plt.subplot(1,11,1)
#     ax.axis('off')
#     # imshow(query_path,'query')
#     for i in range(50):
#         ax = plt.subplot(1,21,i+2)
#         ax.axis('off')
#         img_path, _ = image_datasets['gallery'].imgs[index[i]]

#         # label = gallery_label[index[i]]
#         imshow(img_path)  # ax에 추가
        
#         # if label == query_label:
#         #     ax.set_title('%d'%(i+1), color='green')
#         # else:
#         #     ax.set_title('%d'%(i+1), color='red')

#         print(img_path, 'score :', score[index[i]]) # 최종 결과(유사 인물) 출력

# except RuntimeError:
#     for i in range(10):
#         img_path = image_datasets.imgs[index[i]]
#         print(img_path[0])
#     print('If you want to see the visualization of the ranking result, graphical user interface is needed.')

# fig.savefig("show_test.png")


i = 0
result = []
while True:
    if score[index[i]] < 0.8:
        break
    img_path, _ = image_datasets['gallery'].imgs[index[i]]
    # print(img_path, 'score :', score[index[i]]) # 최종 결과(유사 인물) 출력
    result.append(img_path +'_' + str(score[index[i]]))
    i += 1

# 동일 프레임이 또 있으면 제거(동일 프레임에 동일한 인물이 두명일수는 없으니까)
for idx, each_time in enumerate(result):
    if each_time != '0':
        timestamp = each_time[14:28]
        count = 0
        for compare in result[idx+1:]:
            count += 1
            if compare == '0':
                continue
            if timestamp == compare[14:28]:
                result[idx+count] = '0'
    
# '0' 모두 지우기
while True:
    try:
        result.remove('0')
    except:
        break

# 시간에 따라 정렬
sorted_result = sorted(result, key = lambda x : x[14:28])


for i in sorted_result:
    print(i)
