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
import sys

import cv2
import os


#######################################################################
# Evaluate
parser = argparse.ArgumentParser(description='Demo')
parser.add_argument('--query_index', default=777, type=int, help='test_image_index')
parser.add_argument('--test_dir',default='../Market-1501/pytorch',type=str, help='./test_data')
opts = parser.parse_args()

# galley image
image_datasets = {'gallery' : datasets.ImageFolder('./runs2')}

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

# load gallery feature

result = scipy.io.loadmat('./runs2/output2_extract_result.mat')
# query_feature = torch.FloatTensor(result['query_f'])
# query_cam = result['query_cam'][0]
# query_label = result['query_label'][0]


gallery_feature = torch.FloatTensor(result['img_f'])

# gallery_cam = result['gallery_cam'][0]
# gallery_label = result['gallery_label'][0]

######################################################################

# 한 프레임 사진 여기서 던져주기 (여기에 코드 쓰기, 지금은 이미 폴더에 넣어놨고)

#### RUN query_extractor2.py
os.system('python query_extractor2.py')

print('Complete query extract')

######################################################################

# load query feature

detect_path = './test_result/detect'

fileList = os.listdir(detect_path)
fileList = sorted(fileList)
# print('fileList', fileList)
filenum = len(fileList)
# print(filenum)


query_result = scipy.io.loadmat('query2_extract_result.mat')
query_feature_list = torch.FloatTensor(query_result['img_f'])


######################################################################

# 쿼리에 있는 사람 한명씩 돌아가면서 gallery 사람들과 비교
final_result = []

for person_num in range(filenum):
    query_feature = query_feature_list[person_num] # 여기서 쿼리 번호 지정 

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
    
    # for i in index:
    #     print(score[i])

    print()

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
        if score[index[i]] < 0.9:
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

    # 결과 출력
    print("query :", fileList[person_num])
    for i in sorted_result:
        print(i) 
    print()

    final_result.append(sorted_result)




# fileList = fileList[::-1]
fig = plt.figure(figsize=(20,10))

rows = filenum + 1
cols = 15

# 원본 프레임 출력
ax = fig.add_subplot(rows,1,1)
frame_dir = os.listdir('./test_frame/')
frame_path = './test_frame/' + frame_dir[0]
frame_time = frame_dir[0][:14]
frame_month = frame_time[4:6]
frame_day = frame_time[6:8]
frame_hour = frame_time[8:10]
frame_min = frame_time[10:12]
frame_sec = frame_time[12:14]

frame_file = cv2.imread(frame_path)
frame_file = cv2.resize(frame_file, dsize=(500, 500), interpolation=cv2.INTER_LINEAR)
ax.imshow(cv2.cvtColor(frame_file, cv2.COLOR_BGR2RGB))
ax.axis('off')
ax.set_title('Frame \n' + frame_month + '/' + frame_day+ '\n' + frame_hour+ ':' +frame_min+ ':' + frame_sec)


person_idx = 65 # 아스키

# 쿼리, 도출된 결과 출력
for i in range(1,rows):
    ax = fig.add_subplot(rows,cols, i*cols+1)
    query_path = './test_result/detect/' + fileList[i-1]
    query_file = cv2.imread(query_path)
    query_file = cv2.resize(query_file, dsize=(480, 1280), interpolation=cv2.INTER_AREA)
    ax.imshow(cv2.cvtColor(query_file, cv2.COLOR_BGR2RGB))
    ax.axis('off')
    ax.set_title('Person ' + chr(person_idx))
    person_idx += 1
    for j in range(cols):
        try:
            candidate_path = final_result[i-1][j][:35]
            # print(candidate_path)
            candidate_file = cv2.imread(candidate_path)
            candidate_file = cv2.resize(candidate_file, dsize=(480, 1280), interpolation=cv2.INTER_AREA)
            ax = fig.add_subplot(rows,cols,i*cols+j+2)            
            ax.imshow(cv2.cvtColor(candidate_file, cv2.COLOR_BGR2RGB))
            ax.axis('off')
            month = candidate_path[19:21]
            day = candidate_path[21:23]
            hour = candidate_path[23:25]
            minute = candidate_path[25:27]
            second = candidate_path[27:29]
            ax.set_title(month + '/' + day+ '\n' + hour+ ':' +minute+ ':' + second)
        except:
            break

fig.tight_layout()
plt.show()


    