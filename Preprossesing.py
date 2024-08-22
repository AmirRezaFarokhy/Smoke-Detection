import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler, scale
from sklearn.model_selection import train_test_split

path_train = "data_train1"
path_valid = "data_valid"

CATEGORIS = ["ApplyEyeMakeup", 
			 "ApplyLipstick",
			 "BrushingTeeth",
			 "WalkingWithDog",
			 "BlowDryHair",
			 "smoking"]

IMG_SIZE_0 = 200
IMG_SIZE_1 = 200
def take_video_give_data(data_path, sizes=(50,50)):
	data = []
	y = []
	for category in CATEGORIS:
		path = os.path.join(data_path, category)
		class_num = CATEGORIS.index(category) 
		print(f"this {category} is complete!!!")
		for name_video in os.listdir(path):
			try:
				cap = cv2.VideoCapture(os.path.join(path, name_video))
				success, frame = cap.read()
				lst = []
				count = 0
				while success:
					frame = cv2.resize(frame, sizes).astype(np.float32)
					lst.append(frame / 255)
					success, frame = cap.read()
					
				y.append(class_num)
				data.append(np.array(lst))

			except Exception as e:
				print(e)
	return data, y

# Select Frames
def select_frames(frames_arr, n=20):
	videos=[]
	for i in range(len(frames_arr)):
		frames=[]
		for t in np.linspace(0, len(frames_arr[i])-1, num=n):
			frames.append(frames_arr[i][int(t)])
		videos.append(frames)
		
	videos = np.array(videos)
	return videos


def show_samples(X_frames):
	fig = plt.figure(figsize=(32,8))
	for i,image in enumerate(X_frames[-1]):
		ax = plt.subplot(2,5,i+1)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		plt.imshow(image)

	plt.show()

all_dataset, y = take_video_give_data(path_train, sizes=(IMG_SIZE_1, IMG_SIZE_0))
all_dataset = select_frames(all_dataset, n=10)
print(f"Shape of features is {all_dataset.shape}")
print(f"Shape of labels is {np.array(y).shape}")

with open("datasets/features_real.pickle", "wb") as file:
	pickle.dump(all_dataset ,file)

with open("datasets/labels_real.pickle", "wb") as file:
	pickle.dump(y ,file)

show_samples(all_dataset)
#print(y[-1]) # Show some targets

