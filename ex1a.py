import cv2

cap = cv2.VideoCapture('../inputs/ebu7240_hand.mp4')

img_array = []

if (cap.isOpened() == False):
    print("Error opening video stream or file")

#--------------------------------- WRITE YOUR CODE HERE ---------------------------------#

frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
for a in range(0, frame_num):
    ret, frame = cap.read()
    img_array.append(frame)
for b in range(len(img_array)):
    if b < 30:
        continue
    elif b >= 30 and b < 50:
        for m in range(0, width):
            for n in range(0, height):
                img_array[b].itemset((n, m, 0), 0)
                img_array[b].itemset((n, m, 1), 0)
    elif b >= 50 and b < 70:
        for m in range(0, width):
            for n in range(0, height):
                img_array[b].itemset((n, m, 0), 0)
                img_array[b].itemset((n, m, 2), 0)
    else:
        for m in range(0, width):
            for n in range(0, height):
                img_array[b].itemset((n, m, 1), 0)
                img_array[b].itemset((n, m, 2), 0)
# video_capture = cv2.VideoCapture('../inputs/name_QMnumber.mp4')
# if (video_capture.isOpened() == False):
#     print("Error opening video stream or file")
# width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter('../inputs/ebu7240_hand.mp4', fourcc, 30, (360, 640))
# for i in range(0, 90):
#     ret, frame = video_capture.read()
#     frame1 = cv2.resize(frame, (360, 640), interpolation=cv2.INTER_LINEAR)
#     out.write(frame1)
# out.release()

# cap = cv2.VideoCapture('../inputs/ebu7240_hand.mp4')
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = int(cap.get(cv2.CAP_PROP_FPS))
# frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# print(width, height, fps, frame_num)

##########################################################################################

out = cv2.VideoWriter('../results/ex1_a_hand_rgbtest.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (360, 640))
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()

# cv2.imshow('frame', img_array[89])
# cv2.waitKey(0)



