from src.fcnn import FCNN
from src.cnn import CNN

class NN():
    def __init__(self):
        pass

    def build(self, data, kw):

        if kw['ARCH'] == 'fcnn':
            nn = FCNN(data,kw['VERBOSE'], kw['SAVE_INFO'])
            nn.build(kw['CODE_SIZE'], kw['N_NEURONS'], kw['N_HID_LAY'], kw['REGULARISATION'])
        elif kw['ARCH'] == 'cnn':
            nn = CNN(data,kw['VERBOSE'], kw['SAVE_INFO'])
            nn.build(kw['N_CONV_BLOCKS'], kw['CODE_SIZE'], kw['REGULARISATION'], kw['N_FILTERS'], kw['KERNEL_SIZE'], kw['STRIDE'],
            kw['N_HID_LAY'], kw['N_NEURONS'])
        else:
            raise ValueError('dataset "{}" not implemented!'.format(kw['ARCH']))
        return nn