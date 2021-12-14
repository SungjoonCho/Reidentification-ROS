# video2.py
import cv2

# cap = cv2.VideoCapture('/home/jskimlab/graduate/src/graduate/ros_opencv_pub/video/test2.mp4')

# #재생할 파일의 넓이와 높이
# width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
# height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
# fps = cap.get(cv2.CAP_PROP_FPS)
# print("재생할 파일 넓이, 높이 : %d, %d"%(width, height))

# fourcc = cv2.VideoWriter_fourcc(*'DIVX')
# out = cv2.VideoWriter('output.avi', fourcc, fps, (int(width), int(height)))
# print(fps, int(width), int(height))
# counter = 0

# while(cap.isOpened()):
#     ret, frame = cap.read()
    
#     if ret == False:
#         break
    
#     cv2.imshow('frame',frame)
#     out.write(frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#     counter += 1
#     if counter ==  100:
#         break
    
# cap.release()
# out.release()
# cv2.destroyAllWindows()



img = cv2.imread('/home/jskimlab/Downloads/DukeMTMC-reID/bounding_box_test/0002_c1_f0048718.jpg')

print(type(img))
print(img.shape)
print(type(img.shape))