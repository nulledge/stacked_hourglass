import numpy as np
class Image:
    Width = Height = 256
    Channel = 3
    
class Heatmap:
    Width = Height = 64
    Joints = 11


class Reader(object):
    __slots__ = [   '__train_stream', '__train_seeker', '__train_size',
                    '__test_stream', '__test_seeker', '__test_size']

    def __init__(self, train = 'train.dat', test = 'test.dat'):
        option = 'rb'
        self.__train_stream = open(train, option)
        self.__test_stream = open(test, option)

        self.__train_seeker = self.__test_seeker = 0
        self.__train_size = 4502
        self.__test_size = 501

    def __del__(self):
        self.__train_stream.close()
        self.__test_stream.close()

    def __read(self, index, is_train):
        stream = self.__train_stream if is_train else self.__test_stream
        data = np.array(list(stream.read(Image.Width * Image.Height * Image.Channel + Heatmap.Joints * Heatmap.Width * Heatmap.Height)))
        image = data[:Image.Width * Image.Height * Image.Channel]
        heatmap = data[Image.Width * Image.Height * Image.Channel:]

        image = image.reshape((Image.Height, Image.Width, Image.Channel))
        heatmap = heatmap.reshape((Heatmap.Height, Heatmap.Width, Heatmap.Joints))

        return image, heatmap

    def batch(self, size, is_train):
        images = []
        heatmaps = []

        for index in range(size):

            if is_train == True:
                image, heatmap = self.__read(self.__train_seeker, is_train = is_train)
                self.__train_seeker += 1
                if self.__train_seeker >= self.__train_size:
                    self.__train_seeker = 0
                    self.__train_stream.seek(0)
            else:
                image, heatmap = self.__read(self.__test_seeker, is_train = is_train)
                self.__test_seeker += 1
                if self.__test_seeker >= self.__test_size:
                    self.__test_seeker = 0
                    self.__test_stream.seek(0)
            
            images.append(image)
            heatmaps.append(heatmap)
        return images, heatmaps