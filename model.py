import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model 
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2

# Using Google Colab GPU's, all Data send to google deive and loaded here...
path_features = "/content/drive/MyDrive/YOURFILEPATH.pickle" # put your file path
path_labels = "/content/drive/MyDrive/YOURFILEPATH.pickle"

with open(path_features, "rb") as f:
	X = pickle.load(f)
	f.close()

with open(path_labels, "rb") as f:
	y = pickle.load(f)
	y = np.array(y)
	f.close()
	
def accuracy_each_group(y_true, y_pred):
	
    def func(data):
        y_index = [i for i in range(6)]
        cnt = 0
        y_tru = []
        for i in range(6):
            lst = []
            for j in data:
                if j==y_index[i]:
                    lst.append(j)

            y_tru.append(lst)
    
    	return y_tru

    actual = func(y_true)
    predict = func(y_pred)
    all_lst_names = ["ApplyEyeMakeup", 
                     "ApplyLipstick",
                     "BrushingTeeth",
                     "BlowDryHair",
                     "WalkingWithDog",
                     "smoking"]
    
    cnt = 0
    for ac, pre in zip(actual, predict):
        length_max = max(len(ac), len(pre))
        length_min = min(len(ac), len(pre)) 
        accuracy = length_min/length_max
        print(f"this {all_lst_names[cnt]} group accuracy is {accuracy}")
        cnt += 1
    

x_train, x_valid, y_train, y_valid_cn = train_test_split(X, y, test_size=0.2)

print(f"Shape of X_train {x_train.shape}")
print(f"Shape of the Y_train{y_train.shape}")
print(f"Shape of the X_valid{x_valid.shape}")
print(f"Shape of the Y_valid{y_valid.shape}")

y_train = to_categorical(y_train, 8)
y_valid = to_categorical(y_valid_cn, 8)


def list_to_stack(xs):
	xs=tf.stack(xs, axis=1)
	s = tf.shape(xs)
	return xs

ish=(10, 128, 192, 3)
xs=[]
inp = Input(ish)

for slice_indx in range(0,10,1):
	x = Lambda(lambda x: x[:, slice_indx])(inp)
	x = BatchNormalization(momentum=0.75)(x)
	x = Conv2D(filters=20, kernel_size=3, padding='same', activation='relu', kernel_regularizer=l2(0.001))(x)
	x = BatchNormalization(momentum=0.75)(x)
	x = MaxPooling2D(pool_size=2)(x)
	x = Conv2D(filters=30, kernel_size=3, padding='same', activation='relu', kernel_regularizer=l2(0.001))(x)
	x = BatchNormalization(momentum=0.75)(x)
	x = MaxPooling2D(pool_size=2)(x)
	x = Conv2D(filters=30, kernel_size=3, padding='same', activation='relu', kernel_regularizer=l2(0.001))(x)
	xs.append(x)


t = Lambda(list_to_stack)(xs)
t = Conv3D(50, 3, padding='same', activation='relu', kernel_regularizer=l2(0.001))(t)
t = BatchNormalization(momentum=0.75)(t)
t = Dropout(0.1)(t)
target_shape = (10,32*48*50)
t = Reshape(target_shape)(t)
t = GRU(25, return_sequences=True)(t)
t = Dropout(0.1)(t)
t = GRU(50, return_sequences=False)(t)
t = Dropout(0.4)(t)
t = Dense(100, 'relu')(t)
t = Dropout(0.2)(t)
out = Dense(8, activation='softmax')(t)

model = Model(inputs=inp, outputs=out)
opt = tf.keras.optimizers.SGD(lr=0.008)
model.compile(loss="categorical_crossentropy", 
							optimizer=opt, 
							metrics=['accuracy'])

model.summary()

hist = model.fit(x_train, y_train,
								 epochs=50,
								 batch_size=16,
								 shuffle=True,
								 validation_data=(x_valid, y_valid))


# plot loss function
plt.plot(hist.history["loss"], label="train_loss")
plt.plot(hist.history["val_loss"], label="val_loss")
plt.legend(loc="upper right")
plt.show()

# plot accuracy
plt.plot(hist.history["accuracy"], label="train_acc")
plt.plot(hist.history["val_accuracy"], label="val_acc")
plt.legend(loc="upper right")
plt.show()


y_pred = model.predict(x_valid)
y_pred_class = []
for i in y_pred:
    y_pred_class.append(np.argmax(i))

accuracy_each_group(y_valid_cn, y_pred_class)


