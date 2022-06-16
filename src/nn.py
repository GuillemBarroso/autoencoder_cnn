from src.fcnn import FCNN
from src.cnn import CNN
from src.param_ae import PARAM_AE

class NN():
    def __init__(self, data, kw):
        if kw['ARCH'] == 'fcnn':
            self.model = FCNN(data,kw)
        elif kw['ARCH'] == 'cnn':
            self.model = CNN(data,kw)
        elif kw['ARCH'] == 'param_ae':
            self.model = PARAM_AE(data,kw)
        else:
            raise ValueError('dataset "{}" not implemented!'.format(kw['ARCH']))

        self.model.build(kw)