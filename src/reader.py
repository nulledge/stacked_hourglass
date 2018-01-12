import numpy as np
class Image:
    Width = Height = 256
    Channel = 3
    
class Heatmap:
    Width = Height = 64
    Joints = 16


class Reader(object):
    __slots__ = [   '__train_stream', '__train_seeker', '__train_size',
                    '__test_stream', '__test_seeker', '__test_size']

    def __init__(self, train = 'train.dat', test = 'test.dat'):
        option = 'rb'
        self.__train_stream = open(train, option)
        self.__test_stream = open(test, option)

        self.__train_seeker = self.__test_seeker = 0
        self.__train_size = 19122
        self.__test_size = 2125

    def __del__(self):
        self.__train_stream.close()
        self.__test_stream.close()

    def batch(self, size, is_train):
        if is_train == True:
            if self.__train_seeker + size > self.__train_size:
                raise Exception()
            self.__train_stream.seek(Image.Width * Image.Height * Image.Channel * self.__train_seeker)
            image = self.__train_stream.read(Image.Width * Image.Height * Image.Channel * size)
            image = np.frombuffer(image, dtype = np.uint8, count = Image.Width * Image.Height * Image.Channel * size)
            image = np.reshape(image, newshape = (size, Image.Height, Image.Width, Image.Channel))
            
            self.__train_stream.seek(Image.Width * Image.Height * Image.Channel * self.__train_size
                                     + Heatmap.Width * Heatmap.Height * Heatmap.Joints * self.__train_seeker)
            heatmap = self.__train_stream.read(Heatmap.Width * Heatmap.Height * Heatmap.Joints * size)
            heatmap = np.frombuffer(heatmap, dtype = np.uint8, count = Heatmap.Width * Heatmap.Height * Heatmap.Joints * size)
            heatmap = np.reshape(heatmap, newshape = (size, Heatmap.Height, Heatmap.Width, Heatmap.Joints))
            
            self.__train_seeker += size
            
            if self.__train_seeker == self.__train_size:
                self.__train_seeker = 0
        else:
            if self.__test_seeker + size > self.__test_size:
                raise Exception()
            self.__test_stream.seek(Image.Width * Image.Height * Image.Channel * self.__test_seeker)
            image = self.__test_stream.read(Image.Width * Image.Height * Image.Channel * size)
            image = np.frombuffer(image, dtype = np.uint8, count = Image.Width * Image.Height * Image.Channel * size)
            image = np.reshape(image, newshape = (size, Image.Height, Image.Width, Image.Channel))
            
            self.__test_stream.seek(Image.Width * Image.Height * Image.Channel * self.__test_size
                                     + Heatmap.Width * Heatmap.Height * Heatmap.Joints * self.__test_seeker)
            heatmap = self.__test_stream.read(Heatmap.Width * Heatmap.Height * Heatmap.Joints * size)
            heatmap = np.frombuffer(heatmap, dtype = np.uint8, count = Heatmap.Width * Heatmap.Height * Heatmap.Joints * size)
            heatmap = np.reshape(heatmap, newshape = (size, Heatmap.Height, Heatmap.Width, Heatmap.Joints))
            
            self.__test_seeker += size
            
            if self.__test_seeker == self.__test_size:
                self.__test_seeker = 0
        return image, heatmap