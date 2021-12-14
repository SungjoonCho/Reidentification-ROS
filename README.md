# Person Re-Identification을 활용한 CCTV 속 용의자 추적

![image](https://user-images.githubusercontent.com/80872528/145921687-e0805b3e-e99b-4eb2-bbc5-36b3e2822606.png)



* [test 1 영상 다운로드](https://github.com/Achilleas/cctv-prototype-deepsort/blob/master/videos/town_short.mp4)

  <p align="center">
    <img width="500" height="300" src="https://user-images.githubusercontent.com/80872528/145817522-b7b12a6b-b970-4fb7-89e2-5ba0f4c22ee3.png">
  </p>

* [test 2 영상 다운로드](http://www.cvg.reading.ac.uk/PETS2009/a.html)
  
  <p align="center">
    <img width="500" height="300" src="https://user-images.githubusercontent.com/80872528/145817577-c10c2da0-6f01-46f8-acf5-31c27b681d83.png">
  </p>
  



* 개발 환경

  <pre>
	Ubuntu 18.04 LTS 64 bit
	GPU : Geforce RTX 2080ti
	Cuda : 10.2
	driver : 440
	cudnn : 8.0.3
	ROS 1 Melodic
	Python : 3.6 (virtual env)	
	torch : 1.7.0
	torchvision : 0.8.1
	opencv-python : 4.1.2
  </pre>
  
  
* 가상환경 구성  
  
  가상환경 생성 후 requirementx.txt 모듈 다운로드
  
  <pre>
  $ conda activate (virtualenv)
  </pre>
  

* CCTV 영상 송신부(Ros Pubisher), CCTV 영상 수신부(Ros Subscriber)
  
  [ROS 설명 참조](https://github.com/SungjoonCho/ROS-study), [ROS wiki 설명](http://wiki.ros.org/ROS/Tutorials)
  
  Terminal 1
  <pre>
  $ mkdir -p catkin_ws/src
  $ cd catkin_ws/src
  $ catkin_create_pkg catkin_ws std_msgs rospy roscpp
  $ cd ..
  $ catkin_make
  
  ros_opencv_pub, subscriber 디렉토리 catkin_ws/src로 복사
  
  </pre>

  Terminal 2
  <pre>
  $ roscore
  </pre>
  
  Terminal 3 (Publisher) - test 영상 publsih
  <pre>
  $catkin_ws source ./bash
  $catkin_ws ros_opencv_pub ros_opencv_pub2 
  </pre>
  
  Terminal 4 (Subscriber) - test 영상 subscribe
  <pre>
  $/catkin_ws/src/catkin_ws/subscriber/ mkdir -p output2/video
  $/catkin_ws/src/catkin_ws/subscriber/ python ros_sub2.py
  </pre>
   


* 훈련 데이터셋 다운로드

  [market 1501](https://www.kaggle.com/pengcw1/market-1501/data)
  
* Training & Test
  <pre>
  python train.py --gpu_ids 0 --name ft_ResNet50 --train_all --batchsize 32  --data_dir your_data_path
  </pre>
  
  <pre>
  python test.py --gpu_ids 0 --name ft_ResNet50 --test_dir your_data_path  --batchsize 32 --which_epoch 59
  </pre>
  
* Demo
  <pre>
  python demo.py --query_index 777
  </pre>
  
  Reference : [Person_reID_baseline_pytorch](https://github.com/layumi/Person_reID_baseline_pytorch/blob/master/tutorial/README.md)


* Person detection (yolo v5) & Feature Extract
  
  <pre>
    $python extractor2.py
  </pre>

  detect2.py의 yolo v5로 사람 탐지 후 반환된 bounding box 좌표의 사람 feature 추출
  
  각 사람 ROI는 runs2에 저장됨
  
  추출된 feature는 /runs2/output2_extract_result.mat에 저장됨
  
  
* 사람 찾기(Query Frame)

  <pre>
    $python demo2_full.py 
  </pre>
  
  test_frame에 사진 저장해두고 실행시, query_extractor2.py에서 frame의 feature 추출 후 
  
  test_result/detect/에 각 인물 ROI 저장, ./test_result/detect에 feature 값 query2_extract_result.mat 저장
  
  그리고 output2_extract_result과 query2_extract_result 비교하며 파일명(촬영시간 및 ID)로 사람 분별
  
  

* 실험 과정 영상 - [Google Drive](https://drive.google.com/drive/folders/1nzh57DeMqPWge6X7Q2PlNvUwW7iobQ2W?usp=sharing)

![image](https://user-images.githubusercontent.com/80872528/145921687-e0805b3e-e99b-4eb2-bbc5-36b3e2822606.png)

