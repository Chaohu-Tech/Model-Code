import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"      #双显卡
import tensorflow as tf
from keras import layers, models
from tensorflow.python.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


#设置参数
path = 'E:/model/dead_model/'

dicClass = {'notdead': 0,
            'dead': 1}

classnum = 2
EPOCHS = 30


#下面都不用改
labelList = []
INIT_LR = 0.0001
Batch_Size = 128
np.random.seed(123)
datapath = path + 'train'

#加载图片
#和以前做法不同的是，这里不再处理图片，而是只返回图片路径的list列表
def loadImageData():
    imageList = []
    listClasses = os.listdir(datapath)    # 类别文件夹
    print(listClasses)
    for class_name in listClasses:
        label_id = dicClass[class_name]
        class_path = os.path.join(datapath, class_name)
        image_names = os.listdir(class_path)
        for image_name in image_names:
            image_full_path = os.path.join(class_path, image_name)
            labelList.append(label_id)
            imageList.append(image_full_path)
    return imageList


print("开始加载数据")
imageArr = loadImageData()
labelList = np.array(labelList)
print("加载数据完成")

#random split
trainX, valX, trainY, valY = train_test_split(imageArr, labelList, test_size=0.2, random_state=123)


#定义图像处理的方法
def generator(file_pathList, labels, batch_size, train_action = False):
    L = len(file_pathList)
    while True:
        input_labels = []
        input_samples = []
        for row in range(0, batch_size):
            temp = np.random.randint(0, L)
            X = file_pathList[temp]
            Y = labels[temp]
            image = cv2.imdecode(np.fromfile(X, dtype=np.uint8), -1)
            if image.shape[2] > 3:
                image = image[:, :, :3]
            # if train_action:
            #     image = train_transform(image=image)['image']
            # else:
            #     image = val_transform(image=image)['image']
            image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LANCZOS4)
            image = img_to_array(image)
            input_samples.append(image)
            input_labels.append(Y)
        batch_x = np.asarray(input_samples)
        batch_y = np.asarray(input_labels)
        yield (batch_x, batch_y)



# 构造网络模型
model = models.Sequential([
    layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(512, 512, 3)),
    layers.MaxPooling2D(2,2),
    # layers.Dropout(0.5),
    
    layers.Conv2D(32, (3,3), padding='same', activation='relu'),
    layers.MaxPooling2D(2,2),
    # layers.Dropout(0.5),
    
    layers.Conv2D(64, (3,3), padding='same', activation='relu'),
    layers.MaxPooling2D(2,2),
    # layers.Dropout(0.5),
    
    layers.Conv2D(64, (3,3), padding='same', activation='relu'),
    layers.MaxPooling2D(2,2),
    # layers.Dropout(0.5),
    
    # layers.Conv2D(128, (3,3), padding='same', activation='relu'),
    # layers.MaxPooling2D(2,2),
    # layers.Dropout(0.5),
     
    layers.Flatten(),
    # layers.Dense(256, activation='relu'),
    # layers.Dropout(0.5),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(classnum, activation='softmax'),
])
 

# 查看网络结构
model.summary()


#定义损失函数优化器
model.compile(optimizer = Adam(learning_rate = INIT_LR),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


checkpointer = ModelCheckpoint(filepath=path + 'best_model.hdf5',
                               monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

reduce = ReduceLROnPlateau(monitor='val_accuracy', patience=10,
                           verbose=1,
                           factor=0.5,
                           min_lr=1e-6)


#模型训练
history = model.fit(generator(trainX, trainY, Batch_Size, train_action=True),
                    steps_per_epoch = len(trainX) / Batch_Size,
                    validation_data = generator(valX, valY, Batch_Size, train_action=False),
                    epochs = EPOCHS,
                    validation_steps = len(valX) / Batch_Size,
                    callbacks = [checkpointer, reduce])

model.save(path + 'my_model.h5')
print(history)


#保留训练结果，并将其生成图片
loss_trend_graph_path = path + 'WW_loss.png'
acc_trend_graph_path = path + 'WW_acc.png'
print("开始绘图")
# summarize history for accuracy
fig = plt.figure(1)
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["train", "val"], loc="upper left")
plt.savefig(acc_trend_graph_path)
plt.close(1)
# summarize history for loss
fig = plt.figure(2)
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "val"], loc="upper left")
plt.savefig(loss_trend_graph_path)
plt.close(2)
print("绘图完成")



# ##批量预测
# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"      #双显卡
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import img_to_array
# import cv2
# import numpy as np
# from tqdm import tqdm
# import csv
# from keras import Model
# import pandas as pd

# path = 'G:/Downloads/pannuke_cut/'
# save_path = os.path.join(path, 'test')
# predict_dir = os.path.join(save_path, 'Dead')

# model = load_model(path + 'my_model.h5')
# #用model.summary()查看最后一层卷积层的名称
# last_conv_layer_name = 'conv2d_46'

# model_labels = {0: 'Neoplastic',
#                 1: 'Inflammatory',
#                 2: 'Connective',
#                 3: 'Dead',
#                 4: 'Epithelial'}


# imagelist = pd.DataFrame()
# denselist = pd.DataFrame()

# index = -3
# layer_model = Model(inputs=model.input, outputs=model.layers[index].output)


# test = os.listdir(predict_dir)
# for file in tqdm(test):
#     filepath=os.path.join(predict_dir,file)
#     image = cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), -1)
#     image = img_to_array(image)
#     image = image[np.newaxis,:,:,:]
    
#     out = model.predict(image)  
#     out = pd.DataFrame(out)
#     pre = np.argmax(out)
#     class_name_list = model_labels[pre]
    
#     out.insert(0, 'file', file) 
#     out.insert(out.shape[1], 'pre_label', pre) 
#     out.insert(out.shape[1], 'predict', class_name_list) 
    
#     imagelist = imagelist.append(out, ignore_index = True)
    
#     Dense = layer_model.predict(image)
#     Dense = pd.DataFrame(Dense)
#     Dense.insert(0, column = 'sample', value = file)
#     denselist = denselist.append(Dense, ignore_index = True)
    

# imagelist.to_csv(save_path + 'Dead_predict.csv')
# denselist.to_csv(save_path + 'Dead_tezheng.csv')


 
# #可以指定某一层的输出
# from keras import Model
# index = -3
# layer_model = Model(inputs=model.input, outputs=model.layers[index].output)
# predict = layer_model.predict(image)

# model.layers
# Out[65]: 
# [<keras.layers.convolutional.Conv2D at 0x1fa8b7ec4e0>,
#  <keras.layers.pooling.MaxPooling2D at 0x1fd63947240>,
#  <keras.layers.convolutional.Conv2D at 0x1fd6394c5c0>,
#  <keras.layers.pooling.MaxPooling2D at 0x1fd640b5630>,
#  <keras.layers.convolutional.Conv2D at 0x1fd640b5d68>,
#  <keras.layers.pooling.MaxPooling2D at 0x1fad61acfd0>,
#  <keras.layers.convolutional.Conv2D at 0x1fab9d93a90>,
#  <keras.layers.pooling.MaxPooling2D at 0x1fd63944278>,
#  <keras.layers.core.Flatten at 0x1fd6392b978>,
#  <keras.layers.core.Dense at 0x1fd63916898>,
#  <keras.layers.core.Dropout at 0x1fd640a8ac8>,
#  <keras.layers.core.Dense at 0x1fd640a1710>]

# model.layers[-1]
# Out[66]: <keras.layers.core.Dense at 0x1fd640a1710>

# model.layers[-2]
# Out[67]: <keras.layers.core.Dropout at 0x1fd640a8ac8>

# model.layers[-3]
# Out[68]: <keras.layers.core.Dense at 0x1fd63916898>

# model.layers[-4]
# Out[69]: <keras.layers.core.Flatten at 0x1fd6392b978>
