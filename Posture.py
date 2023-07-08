{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "71c8b844",
   "metadata": {
    "scrolled": false
   },
   "outputs": [],
   "source": [
    "import cv2\n",
    "import numpy as np\n",
    "import time\n",
    "import math\n",
    "import mediapipe as mp\n",
    "\n",
    "#mp_holistic = mp.solutions.holistic\n",
    "#holistic_model = mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) \n",
    "#mp_face_detection = mp.solutions.face_detection\n",
    "mp_pose = mp.solutions.pose\n",
    "pose_video = mp_pose.Pose(static_image_mode=False,model_complexity=1,min_detection_confidence=0.7,min_tracking_confidence=0.7)\n",
    "mp_drawing = mp.solutions.drawing_utils\n",
    "\n",
    "#with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:\n",
    "cap = cv2.VideoCapture(\"single.mp4\")\n",
    "def gradient(pt1,pt2):\n",
    "    if (pt1[0]-pt2[0])==0:\n",
    "        return (pt1[1]-pt2[1])\n",
    "    else:\n",
    "        return (pt1[1]-pt2[1])/(pt1[0]-pt2[0])\n",
    "def getAngle(pt1,pt2,pt3):\n",
    "    m1 = gradient(pt1,pt2)\n",
    "    m2 = gradient(pt2,pt3)\n",
    "    if 1+(m2*m1)!=0:\n",
    "        angR = math.atan((m2-m1)/(1+(m2*m1)))\n",
    "    else:\n",
    "        angR = math.atan(m2-m1)\n",
    "    angD = round(math.degrees(angR))\n",
    "    return angD\n",
    "dis_list=[]\n",
    "while True:\n",
    "    success,img = cap.read()\n",
    "    img = cv2.flip(img, 0)\n",
    "    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\n",
    "    results = pose_video.process(imgRGB)\n",
    "    #rint(results.pose_landmarks)\n",
    "    if results.pose_landmarks:\n",
    "        mp_drawing.draw_landmarks(image=img,landmark_list=results.pose_landmarks,connections=mp_pose.POSE_CONNECTIONS,landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255,255,255),thickness=3,circle_radius=3),connection_drawing_spec=mp_drawing.DrawingSpec(color=(49,125,237),thickness=2,circle_radius=2))\n",
    "    height,width,_=img.shape\n",
    "\n",
    "    shoulder_landmark = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x*width,results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y*height)\n",
    "    hip_landmark = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x*width,results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y*height)\n",
    "    knee_landmark = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].x*width,results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y*height)\n",
    "    ankle_landmark = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].x*width,results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y*height)\n",
    "    \n",
    "    dis_list.append((int(ankle_landmark[0]),int(ankle_landmark[1])))\n",
    "    #if (len(dis_list)>4):\n",
    "        #if(dis_list[-1]==dis_list[-2]==dis_list[-3]==dis_list[-4]):\n",
    "            #cv2.putText(img,\"Standing\",(190,30),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),3)\n",
    "    s=getAngle(shoulder_landmark,hip_landmark,knee_landmark)\n",
    "    a=getAngle(hip_landmark,knee_landmark,ankle_landmark)\n",
    "    if(a>75):\n",
    "            cv2.putText(img,\"correct\",(10,30),cv2.FONT_HERSHEY_PLAIN,2,(65,163,23),3)\n",
    "            cv2.putText(img,\"Box picked\",(190,30),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),2)\n",
    "    if(s<-53 and s>-70):\n",
    "            cv2.putText(img,\"incorrect\",(10,30),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),3)\n",
    "            cv2.putText(img,\"Box picked\",(190,30),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),2)\n",
    "        \n",
    "    cv2.imshow(\"image\",img)\n",
    "    cv2.waitKey(70)\n",
    "\n",
    "    \n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0b4561dd",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
