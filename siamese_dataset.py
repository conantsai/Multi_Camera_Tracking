import numpy as np 
import os
from skimage import io, color

# class Dataset:
#     def __init__(self, data, label):
#         self._index_in_epoch = 0
#         self._epochs_completed = 0
#         self._data = data
#         self._label = label
#         self._num_examples = data.shape[0]
#         pass

#     @property
#     def data(self):
#         return self._data
#     @property
#     def label(self):
#         return self._label

#     def next_batch(self, batch_size, shuffle=True):
#         start = self._index_in_epoch
#         if start == 0 and self._epochs_completed == 0:
#             idx = np.arange(0, self._num_examples)
#             np.random.shuffle(idx)  # shuffle indexe
#             self._data = self.data[idx]  # get the shuffled data
#             self._label = self.label[idx]  # get the shuffled data

#         # go to the data of next batch
#         if start + batch_size > self._num_examples:
#             '''
#             note: when start  == self._num_examples, data_rest_part = np.array([])
#             '''
#             self._epochs_completed += 1
#             # print(self.data)
#             rest_num_examples = self._num_examples - start
#             data_rest_part = self.data[start:self._num_examples]
#             label_rest_part = self.label[start:self._num_examples]
#             idx_update = np.arange(0, self._num_examples)
#             np.random.shuffle(idx_update)
#             self._data = self.data[idx_update]  # get another shuffled data
#             self._label = self.label[idx_update]  # get the shuffled data

#             start = 0
#             self._index_in_epoch = batch_size - rest_num_examples
#             end = self._index_in_epoch
#             data_new_part = self._data[start:end]
#             label_new_part = self._label[start:end]
#             return np.concatenate((data_rest_part, data_new_part), axis=0), np.concatenate((label_rest_part, label_new_part), axis=0)
#         else:
#             self._index_in_epoch += batch_size
#             end = self._index_in_epoch
#             return self._data[start:end], self._label[start:end]

class Dataset:
    def __init__(self, data1, label1, data2, label2):
        self._index_in_epoch = 0
        self._epochs_completed = 0
        self._data1 = data1
        self._label1 = label1
        self._data2 = data2
        self._label2 = label2
        # self._num_examples = lambda x:data1.shape[0] if (data1.shape[0] < data2.shape[0]) else data2.shape[0]
        if (data1.shape[0] < data2.shape[0]):
            self._num_examples = data1.shape[0]
        else:
            self._num_examples = data2.shape[0]
        pass

    @property
    def data1(self):
        return self._data1
    @property
    def label1(self):
        return self._label1
    @property
    def data2(self):
        return self._data2
    @property
    def label2(self):
        return self._label2

    def next_batch(self, batch_size, shuffle=True):
        start = self._index_in_epoch
        if start == 0 and self._epochs_completed == 0:
            idx = np.arange(0, self._num_examples)
            np.random.shuffle(idx)  # shuffle indexe
            self._data1 = self.data1[idx]  # get the shuffled data
            self._label1 = self.label1[idx]  # get the shuffled data
            self._data2 = self.data2[idx]  # get the shuffled data
            self._label2 = self.label2[idx]  # get the shuffled data

        # go to the data of next batch
        if start + batch_size > self._num_examples:
            '''
            note: when start  == self._num_examples, data_rest_part = np.array([])
            '''
            self._epochs_completed += 1
            # print(self.data)
            rest_num_examples = self._num_examples - start
            data1_rest_part = self.data1[start:self._num_examples]
            label1_rest_part = self.label1[start:self._num_examples]
            data2_rest_part = self.data2[start:self._num_examples]
            label2_rest_part = self.label2[start:self._num_examples]
            idx_update = np.arange(0, self._num_examples)
            np.random.shuffle(idx_update)
            self._data1 = self.data1[idx_update]  # get another shuffled data
            self._label1 = self.label1[idx_update]  # get the shuffled data
            self._data2 = self.data2[idx_update]  # get another shuffled data
            self._label2 = self.label2[idx_update]  # get the shuffled data

            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            data1_new_part = self._data1[start:end]
            label1_new_part = self._label1[start:end]
            data2_new_part = self._data2[start:end]
            label2_new_part = self._label2[start:end]
            return np.concatenate((data1_rest_part, data1_new_part), axis=0), np.concatenate((label1_rest_part, label1_new_part), axis=0), \
                   np.concatenate((data2_rest_part, data2_new_part), axis=0), np.concatenate((label2_rest_part, label2_new_part), axis=0)

        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._data1[start:end], self._label1[start:end], self._data2[start:end], self._label2[start:end]


def get_data(datapath, colorTrans=False):
    print('Getting images & labels ... ')

    images, labels = list(), list()
    source_flag = True

    for root, dirs, files in os.walk(datapath):
        if len(files) == 0: continue
        root = root.replace("\\", "/")
        for index, content in enumerate(files):
            try:
                image_value = io.imread(root + "/" + content)
                if colorTrans == True:
                    if source_flag == True: 
                        Slab = color.rgb2lab(image_value)
                        SLABmean = Slab.mean(0).mean(0)
                        SLABstd = Slab.std(0).std(0)
                        source_flag = False
                    else:
                        Tlab = color.rgb2lab(image_value)
                        TLABmean = Tlab.mean(0).mean(0)
                        TLABstd = Tlab.std(0).std(0)
                        height, width, channels = image_value.shape
                        for x in range(0, height):
                            for y in range(0, width):
                                Tlab[x][y][0] = (SLABstd[0]/TLABstd[0])*(Tlab[x][y][0] - TLABmean[0]) + SLABmean[0]
                                Tlab[x][y][1] = (SLABstd[1]/TLABstd[1])*(Tlab[x][y][1] - TLABmean[1]) + SLABmean[1]
                                Tlab[x][y][2] = (SLABstd[2]/TLABstd[2])*(Tlab[x][y][2] - TLABmean[2]) + SLABmean[2]
                        Trgb = color.lab2rgb(Tlab)
                        image_value = (255.0 / Trgb.max() * (Trgb - Trgb.min())).astype(np.uint8)
                images.append(image_value / 255)
                labels.append(content.split("person")[1].split("_")[0])
            except OSError as e:
                # print(root + "/" + content)
                continue

    return np.array(images), np.array(labels, dtype=int)

def get_data_test(datapath, colorTrans=False):
    print('Getting images & labels ... ')

    images, labels = list(), list()
    source_flag = True

    for root, dirs, files in os.walk(datapath):
        if len(files) == 0: continue
        root = root.replace("\\", "/")
        for index, content in enumerate(files):
            try:
                image_value = io.imread(root + "/" + content)
                if colorTrans == True:
                    if source_flag == True: 
                        Slab = color.rgb2lab(image_value)
                        SLABmean = Slab.mean(0).mean(0)
                        SLABstd = Slab.std(0).std(0)
                        source_flag = False
                    else:
                        Tlab = color.rgb2lab(image_value)
                        TLABmean = Tlab.mean(0).mean(0)
                        TLABstd = Tlab.std(0).std(0)
                        height, width, channels = image_value.shape
                        for x in range(0, height):
                            for y in range(0, width):
                                Tlab[x][y][0] = (SLABstd[0]/TLABstd[0])*(Tlab[x][y][0] - TLABmean[0]) + SLABmean[0]
                                Tlab[x][y][1] = (SLABstd[1]/TLABstd[1])*(Tlab[x][y][1] - TLABmean[1]) + SLABmean[1]
                                Tlab[x][y][2] = (SLABstd[2]/TLABstd[2])*(Tlab[x][y][2] - TLABmean[2]) + SLABmean[2]
                        Trgb = color.lab2rgb(Tlab)
                        image_value = (255.0 / Trgb.max() * (Trgb - Trgb.min())).astype(np.uint8)
                images.append(image_value / 255)
                labels.append(content.split("person")[1].split(".")[0])
            except OSError as e:
                # print(root + "/" + content)
                continue

    return np.array(images), np.array(labels, dtype=int)