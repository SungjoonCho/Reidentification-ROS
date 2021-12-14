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
  OS : Ubuntu 18.04
  ROS ver : Melodic
  GPU : RTX 2080ti
  cuda : 10.2
  driver : 440
  cudnn : 8.0.3
  python : 3.7  
  tf-gpu : 1.5  
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

* Person detection

* Feature extract(Person)

* 사람 찾기(Query Frame)

* Visualization


* 실험 과정 영상 - [Google Drive](https://drive.google.com/drive/folders/1nzh57DeMqPWge6X7Q2PlNvUwW7iobQ2W?usp=sharing)

![image](https://user-images.githubusercontent.com/80872528/145921687-e0805b3e-e99b-4eb2-bbc5-36b3e2822606.png)

