import numpy as np
import pandas as pd

class data_io(object):
    def __init__(self):
        pass

    def read_train_data(self,path):
        with open(path,'rb') as contents:
            train_data=pd.read_csv(contents,header=1)
            print(train_data)

    def read_test_data(self,path):
        pass

    def write_data(self,path):
        pass

if __name__ == '__main__':
    io=data_io()
    io.read_train_data('./train.csv')
