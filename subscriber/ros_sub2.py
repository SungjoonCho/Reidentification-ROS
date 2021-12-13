#!/usr/bin/env python
from __future__ import print_function

import roslib
# roslib.load_manifest('my_package')
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
# from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import os
import time


class image_converter:

  def __init__(self):
    self.image_sub = rospy.Subscriber("video/test2",Image,self.callback1)
    self.width1 = 1920
    self.height1 = 1080
    self.fps1 = 10
    self.fourcc1 = cv2.VideoWriter_fourcc(*'MPEG')
    self.out1 = cv2.VideoWriter('./output2/video/output.avi', self.fourcc1, self.fps1, (int(self.width1), int(self.height1)))
    self.counter1 = 0    
    self.save_frame1 = './output2'

    self.interval = 3 # 3초에 한 프레임

  def callback1(self,image_data):
    rospy.loginfo("Subscribed %d frame", self.counter1)
    try:
      # cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
      im = np.frombuffer(image_data.data, dtype=np.uint8).reshape(image_data.height, image_data.width, -1)
    except:
      print("exit")

    # cv2.imshow("Image window", im)
    self.out1.write(im)
    cv2.waitKey(3)

    if self.counter1 % (self.fps1 * self.interval) == 0: 
      # now = rospy.get_rostime()
      # secs = now.secs
      # second = secs%60
      # secs /= 60
      # minute = secs % 60
      # secs /= 60
      # hour = secs
      # print(hour, minute, second)
      now = time.localtime(time.time())
      # str(now.tm_year) + str(now.tm_mon) + str(now.tm_mday) + str(now.tm_hour) + str(now.tm_min) + str(now.tm_sec)

      second = str(now.tm_sec)
      if len(second) == 1:
          second = '0' + second

      minute = str(now.tm_min)
      if len(minute) == 1:
          minute = '0' + minute

      hour = str(now.tm_hour)
      if len(hour) == 1:
          hour = '0' + hour

      day = str(now.tm_mday)
      if len(day) == 1:
          day = '0' + day

      month = str(now.tm_mon)
      if len(month) == 1:
          month = '0' + month

      year = str(now.tm_year)
      filename = year + month + day + hour + minute + second
      cv2.imwrite(os.path.join(self.save_frame1,filename+'.jpg'), im)

    self.counter1 += 1



def main(args):
  ic = image_converter()
  rospy.init_node('image_converter', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")

  # cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
