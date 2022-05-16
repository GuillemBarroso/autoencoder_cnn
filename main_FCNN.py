from common.training_data import Data
from common.fcnn import FCNN
from common.model import Model
from common.postprocessing import plottingPrediction


if __name__ == '__main__':
    dataset = 'beam_homog_x4'
    plotPredictions = True

    extrap9ang5pos = ['Fh0_Fv1_T0.5.txt', 'Fh0_Fv1_T0.5 copy.txt', 'Fh0_Fv1_T0.5 copy 2.txt', 'Fh0_Fv1_T0.5 copy 3.txt', 'Fh0.382683_Fv0.92388_T0.5.txt', 'Fh0.382683_Fv0.92388_T0.5 copy.txt', 'Fh0.382683_Fv0.92388_T0.5 copy 2.txt', 'Fh0.382683_Fv0.92388_T0.5 copy 3.txt', 'Fh0.707107_Fv0.707107_T0.5.txt', 'Fh0.707107_Fv0.707107_T0.5 copy.txt', 'Fh0.707107_Fv0.707107_T0.5 copy 2.txt', 'Fh0.707107_Fv0.707107_T0.5 copy 3.txt', 'Fh0.92388_Fv0.382683_T0.5.txt', 'Fh0.92388_Fv0.382683_T0.5 copy.txt', 'Fh0.92388_Fv0.382683_T0.5 copy 2.txt', 'Fh0.92388_Fv0.382683_T0.5 copy 3.txt', 'Fh1_Fv0_T0.5.txt', 'Fh1_Fv0_T0.5 copy.txt', 'Fh1_Fv0_T0.5 copy 2.txt', 'Fh1_Fv0_T0.5 copy 3.txt', 'Fh0.92388_Fv-0.382683_T0.5.txt', 'Fh0.92388_Fv-0.382683_T0.5 copy.txt', 'Fh0.92388_Fv-0.382683_T0.5 copy 2.txt', 'Fh0.92388_Fv-0.382683_T0.5 copy 3.txt', 'Fh0.707107_Fv-0.707107_T0.5.txt', 'Fh0.707107_Fv-0.707107_T0.5 copy.txt', 'Fh0.707107_Fv-0.707107_T0.5 copy 2.txt', 'Fh0.707107_Fv-0.707107_T0.5 copy 3.txt', 'Fh0.382683_Fv-0.92388_T0.5.txt', 'Fh0.382683_Fv-0.92388_T0.5 copy.txt', 'Fh0.382683_Fv-0.92388_T0.5 copy 2.txt', 'Fh0.382683_Fv-0.92388_T0.5 copy 3.txt', 'Fh0_Fv-1_T0.5.txt', 'Fh0_Fv-1_T0.5 copy.txt', 'Fh0_Fv-1_T0.5 copy 2.txt', 'Fh0_Fv-1_T0.5 copy 3.txt', 'Fh0_Fv1_T0.45.txt', 'Fh0_Fv1_T0.45 copy.txt', 'Fh0_Fv1_T0.45 copy 2.txt', 'Fh0_Fv1_T0.45 copy 3.txt', 'Fh0.382683_Fv0.92388_T0.45.txt', 'Fh0.382683_Fv0.92388_T0.45 copy.txt', 'Fh0.382683_Fv0.92388_T0.45 copy 2.txt', 'Fh0.382683_Fv0.92388_T0.45 copy 3.txt', 'Fh0.707107_Fv0.707107_T0.45.txt', 'Fh0.707107_Fv0.707107_T0.45 copy.txt', 'Fh0.707107_Fv0.707107_T0.45 copy 2.txt', 'Fh0.707107_Fv0.707107_T0.45 copy 3.txt', 'Fh0.92388_Fv0.382683_T0.45.txt', 'Fh0.92388_Fv0.382683_T0.45 copy.txt', 'Fh0.92388_Fv0.382683_T0.45 copy 2.txt', 'Fh0.92388_Fv0.382683_T0.45 copy 3.txt', 'Fh1_Fv0_T0.45.txt', 'Fh1_Fv0_T0.45 copy.txt', 'Fh1_Fv0_T0.45 copy 2.txt', 'Fh1_Fv0_T0.45 copy 3.txt', 'Fh0.92388_Fv-0.382683_T0.45.txt', 'Fh0.92388_Fv-0.382683_T0.45 copy.txt', 'Fh0.92388_Fv-0.382683_T0.45 copy 2.txt', 'Fh0.92388_Fv-0.382683_T0.45 copy 3.txt', 'Fh0.707107_Fv-0.707107_T0.45.txt', 'Fh0.707107_Fv-0.707107_T0.45 copy.txt', 'Fh0.707107_Fv-0.707107_T0.45 copy 2.txt', 'Fh0.707107_Fv-0.707107_T0.45 copy 3.txt', 'Fh0.382683_Fv-0.92388_T0.45.txt', 'Fh0.382683_Fv-0.92388_T0.45 copy.txt', 'Fh0.382683_Fv-0.92388_T0.45 copy 2.txt', 'Fh0.382683_Fv-0.92388_T0.45 copy 3.txt', 'Fh0_Fv-1_T0.45.txt', 'Fh0_Fv-1_T0.45 copy.txt', 'Fh0_Fv-1_T0.45 copy 2.txt', 'Fh0_Fv-1_T0.45 copy 3.txt', 'Fh0_Fv1_T0.4.txt', 'Fh0_Fv1_T0.4 copy.txt', 'Fh0_Fv1_T0.4 copy 2.txt', 'Fh0_Fv1_T0.4 copy 3.txt', 'Fh0.382683_Fv0.92388_T0.4.txt', 'Fh0.382683_Fv0.92388_T0.4 copy.txt', 'Fh0.382683_Fv0.92388_T0.4 copy 2.txt', 'Fh0.382683_Fv0.92388_T0.4 copy 3.txt', 'Fh0.707107_Fv0.707107_T0.4.txt', 'Fh0.707107_Fv0.707107_T0.4 copy.txt', 'Fh0.707107_Fv0.707107_T0.4 copy 2.txt', 'Fh0.707107_Fv0.707107_T0.4 copy 3.txt', 'Fh0.92388_Fv0.382683_T0.4.txt', 'Fh0.92388_Fv0.382683_T0.4 copy.txt', 'Fh0.92388_Fv0.382683_T0.4 copy 2.txt', 'Fh0.92388_Fv0.382683_T0.4 copy 3.txt', 'Fh1_Fv0_T0.4.txt', 'Fh1_Fv0_T0.4 copy.txt', 'Fh1_Fv0_T0.4 copy 2.txt', 'Fh1_Fv0_T0.4 copy 3.txt', 'Fh0.92388_Fv-0.382683_T0.4.txt', 'Fh0.92388_Fv-0.382683_T0.4 copy.txt', 'Fh0.92388_Fv-0.382683_T0.4 copy 2.txt', 'Fh0.92388_Fv-0.382683_T0.4 copy 3.txt', 'Fh0.707107_Fv-0.707107_T0.4.txt', 'Fh0.707107_Fv-0.707107_T0.4 copy.txt', 'Fh0.707107_Fv-0.707107_T0.4 copy 2.txt', 'Fh0.707107_Fv-0.707107_T0.4 copy 3.txt', 'Fh0.382683_Fv-0.92388_T0.4.txt', 'Fh0.382683_Fv-0.92388_T0.4 copy.txt', 'Fh0.382683_Fv-0.92388_T0.4 copy 2.txt', 'Fh0.382683_Fv-0.92388_T0.4 copy 3.txt', 'Fh0_Fv-1_T0.4.txt', 'Fh0_Fv-1_T0.4 copy.txt', 'Fh0_Fv-1_T0.4 copy 2.txt', 'Fh0_Fv-1_T0.4 copy 3.txt', 'Fh0_Fv1_T0.35.txt', 'Fh0_Fv1_T0.35 copy.txt', 'Fh0_Fv1_T0.35 copy 2.txt', 'Fh0_Fv1_T0.35 copy 3.txt', 'Fh0.382683_Fv0.92388_T0.35.txt', 'Fh0.382683_Fv0.92388_T0.35 copy.txt', 'Fh0.382683_Fv0.92388_T0.35 copy 2.txt', 'Fh0.382683_Fv0.92388_T0.35 copy 3.txt', 'Fh0.707107_Fv0.707107_T0.35.txt', 'Fh0.707107_Fv0.707107_T0.35 copy.txt', 'Fh0.707107_Fv0.707107_T0.35 copy 2.txt', 'Fh0.707107_Fv0.707107_T0.35 copy 3.txt', 'Fh0.92388_Fv0.382683_T0.35.txt', 'Fh0.92388_Fv0.382683_T0.35 copy.txt', 'Fh0.92388_Fv0.382683_T0.35 copy 2.txt', 'Fh0.92388_Fv0.382683_T0.35 copy 3.txt', 'Fh1_Fv0_T0.35.txt', 'Fh1_Fv0_T0.35 copy.txt', 'Fh1_Fv0_T0.35 copy 2.txt', 'Fh1_Fv0_T0.35 copy 3.txt', 'Fh0.92388_Fv-0.382683_T0.35.txt', 'Fh0.92388_Fv-0.382683_T0.35 copy.txt', 'Fh0.92388_Fv-0.382683_T0.35 copy 2.txt', 'Fh0.92388_Fv-0.382683_T0.35 copy 3.txt', 'Fh0.707107_Fv-0.707107_T0.35.txt', 'Fh0.707107_Fv-0.707107_T0.35 copy.txt', 'Fh0.707107_Fv-0.707107_T0.35 copy 2.txt', 'Fh0.707107_Fv-0.707107_T0.35 copy 3.txt', 'Fh0.382683_Fv-0.92388_T0.35.txt', 'Fh0.382683_Fv-0.92388_T0.35 copy.txt', 'Fh0.382683_Fv-0.92388_T0.35 copy 2.txt', 'Fh0.382683_Fv-0.92388_T0.35 copy 3.txt', 'Fh0_Fv-1_T0.35.txt', 'Fh0_Fv-1_T0.35 copy.txt', 'Fh0_Fv-1_T0.35 copy 2.txt', 'Fh0_Fv-1_T0.35 copy 3.txt', 'Fh0_Fv1_T0.3.txt', 'Fh0_Fv1_T0.3 copy.txt', 'Fh0_Fv1_T0.3 copy 2.txt', 'Fh0_Fv1_T0.3 copy 3.txt', 'Fh0.382683_Fv0.92388_T0.3.txt', 'Fh0.382683_Fv0.92388_T0.3 copy.txt', 'Fh0.382683_Fv0.92388_T0.3 copy 2.txt', 'Fh0.382683_Fv0.92388_T0.3 copy 3.txt', 'Fh0.707107_Fv0.707107_T0.3.txt', 'Fh0.707107_Fv0.707107_T0.3 copy.txt', 'Fh0.707107_Fv0.707107_T0.3 copy 2.txt', 'Fh0.707107_Fv0.707107_T0.3 copy 3.txt', 'Fh0.92388_Fv0.382683_T0.3.txt', 'Fh0.92388_Fv0.382683_T0.3 copy.txt', 'Fh0.92388_Fv0.382683_T0.3 copy 2.txt', 'Fh0.92388_Fv0.382683_T0.3 copy 3.txt', 'Fh1_Fv0_T0.3.txt', 'Fh1_Fv0_T0.3 copy.txt', 'Fh1_Fv0_T0.3 copy 2.txt', 'Fh1_Fv0_T0.3 copy 3.txt', 'Fh0.92388_Fv-0.382683_T0.3.txt', 'Fh0.92388_Fv-0.382683_T0.3 copy.txt', 'Fh0.92388_Fv-0.382683_T0.3 copy 2.txt', 'Fh0.92388_Fv-0.382683_T0.3 copy 3.txt', 'Fh0.707107_Fv-0.707107_T0.3.txt', 'Fh0.707107_Fv-0.707107_T0.3 copy.txt', 'Fh0.707107_Fv-0.707107_T0.3 copy 2.txt', 'Fh0.707107_Fv-0.707107_T0.3 copy 3.txt', 'Fh0.382683_Fv-0.92388_T0.3.txt', 'Fh0.382683_Fv-0.92388_T0.3 copy.txt', 'Fh0.382683_Fv-0.92388_T0.3 copy 2.txt', 'Fh0.382683_Fv-0.92388_T0.3 copy 3.txt', 'Fh0_Fv-1_T0.3.txt', 'Fh0_Fv-1_T0.3 copy.txt', 'Fh0_Fv-1_T0.3 copy 2.txt', 'Fh0_Fv-1_T0.3 copy 3.txt']
    interp5ang6pos = ['Fh0.707107_Fv0.707107_R0.txt', 'Fh0.707107_Fv0.707107_R0 copy.txt', 'Fh0.707107_Fv0.707107_R0 copy 2.txt', 'Fh0.707107_Fv0.707107_R0 copy 3.txt', 'Fh0.92388_Fv0.382683_R0.txt', 'Fh0.92388_Fv0.382683_R0 copy.txt', 'Fh0.92388_Fv0.382683_R0 copy 2.txt', 'Fh0.92388_Fv0.382683_R0 copy 3.txt', 'Fh1_Fv0_R0.txt', 'Fh1_Fv0_R0 copy.txt', 'Fh1_Fv0_R0 copy 2.txt', 'Fh1_Fv0_R0 copy 3.txt', 'Fh0.92388_Fv-0.382683_R0.txt', 'Fh0.92388_Fv-0.382683_R0 copy.txt', 'Fh0.92388_Fv-0.382683_R0 copy 2.txt', 'Fh0.92388_Fv-0.382683_R0 copy 3.txt', 'Fh0.707107_Fv-0.707107_R0.txt', 'Fh0.707107_Fv-0.707107_R0 copy.txt', 'Fh0.707107_Fv-0.707107_R0 copy 2.txt', 'Fh0.707107_Fv-0.707107_R0 copy 3.txt', 'Fh0.707107_Fv0.707107_R0.05.txt', 'Fh0.707107_Fv0.707107_R0.05 copy.txt', 'Fh0.707107_Fv0.707107_R0.05 copy 2.txt', 'Fh0.707107_Fv0.707107_R0.05 copy 3.txt', 'Fh0.92388_Fv0.382683_R0.05.txt', 'Fh0.92388_Fv0.382683_R0.05 copy.txt', 'Fh0.92388_Fv0.382683_R0.05 copy 2.txt', 'Fh0.92388_Fv0.382683_R0.05 copy 3.txt', 'Fh1_Fv0_R0.05.txt', 'Fh1_Fv0_R0.05 copy.txt', 'Fh1_Fv0_R0.05 copy 2.txt', 'Fh1_Fv0_R0.05 copy 3.txt', 'Fh0.92388_Fv-0.382683_R0.05.txt', 'Fh0.92388_Fv-0.382683_R0.05 copy.txt', 'Fh0.92388_Fv-0.382683_R0.05 copy 2.txt', 'Fh0.92388_Fv-0.382683_R0.05 copy 3.txt', 'Fh0.707107_Fv-0.707107_R0.05.txt', 'Fh0.707107_Fv-0.707107_R0.05 copy.txt', 'Fh0.707107_Fv-0.707107_R0.05 copy 2.txt', 'Fh0.707107_Fv-0.707107_R0.05 copy 3.txt', 'Fh0.707107_Fv0.707107_R0.1.txt', 'Fh0.707107_Fv0.707107_R0.1 copy.txt', 'Fh0.707107_Fv0.707107_R0.1 copy 2.txt', 'Fh0.707107_Fv0.707107_R0.1 copy 3.txt', 'Fh0.92388_Fv0.382683_R0.1.txt', 'Fh0.92388_Fv0.382683_R0.1 copy.txt', 'Fh0.92388_Fv0.382683_R0.1 copy 2.txt', 'Fh0.92388_Fv0.382683_R0.1 copy 3.txt', 'Fh1_Fv0_R0.1.txt', 'Fh1_Fv0_R0.1 copy.txt', 'Fh1_Fv0_R0.1 copy 2.txt', 'Fh1_Fv0_R0.1 copy 3.txt', 'Fh0.92388_Fv-0.382683_R0.1.txt', 'Fh0.92388_Fv-0.382683_R0.1 copy.txt', 'Fh0.92388_Fv-0.382683_R0.1 copy 2.txt', 'Fh0.92388_Fv-0.382683_R0.1 copy 3.txt', 'Fh0.707107_Fv-0.707107_R0.1.txt', 'Fh0.707107_Fv-0.707107_R0.1 copy.txt', 'Fh0.707107_Fv-0.707107_R0.1 copy 2.txt', 'Fh0.707107_Fv-0.707107_R0.1 copy 3.txt', 'Fh0.707107_Fv0.707107_R0.15.txt', 'Fh0.707107_Fv0.707107_R0.15 copy.txt', 'Fh0.707107_Fv0.707107_R0.15 copy 2.txt', 'Fh0.707107_Fv0.707107_R0.15 copy 3.txt', 'Fh0.92388_Fv0.382683_R0.15.txt', 'Fh0.92388_Fv0.382683_R0.15 copy.txt', 'Fh0.92388_Fv0.382683_R0.15 copy 2.txt', 'Fh0.92388_Fv0.382683_R0.15 copy 3.txt', 'Fh1_Fv0_R0.15.txt', 'Fh1_Fv0_R0.15 copy.txt', 'Fh1_Fv0_R0.15 copy 2.txt', 'Fh1_Fv0_R0.15 copy 3.txt', 'Fh0.92388_Fv-0.382683_R0.15.txt', 'Fh0.92388_Fv-0.382683_R0.15 copy.txt', 'Fh0.92388_Fv-0.382683_R0.15 copy 2.txt', 'Fh0.92388_Fv-0.382683_R0.15 copy 3.txt', 'Fh0.707107_Fv-0.707107_R0.15.txt', 'Fh0.707107_Fv-0.707107_R0.15 copy.txt', 'Fh0.707107_Fv-0.707107_R0.15 copy 2.txt', 'Fh0.707107_Fv-0.707107_R0.15 copy 3.txt', 'Fh0.707107_Fv0.707107_R0.2.txt', 'Fh0.707107_Fv0.707107_R0.2 copy.txt', 'Fh0.707107_Fv0.707107_R0.2 copy 2.txt', 'Fh0.707107_Fv0.707107_R0.2 copy 3.txt', 'Fh0.92388_Fv0.382683_R0.2.txt', 'Fh0.92388_Fv0.382683_R0.2 copy.txt', 'Fh0.92388_Fv0.382683_R0.2 copy 2.txt', 'Fh0.92388_Fv0.382683_R0.2 copy 3.txt', 'Fh1_Fv0_R0.2.txt', 'Fh1_Fv0_R0.2 copy.txt', 'Fh1_Fv0_R0.2 copy 2.txt', 'Fh1_Fv0_R0.2 copy 3.txt', 'Fh0.92388_Fv-0.382683_R0.2.txt', 'Fh0.92388_Fv-0.382683_R0.2 copy.txt', 'Fh0.92388_Fv-0.382683_R0.2 copy 2.txt', 'Fh0.92388_Fv-0.382683_R0.2 copy 3.txt', 'Fh0.707107_Fv-0.707107_R0.2.txt', 'Fh0.707107_Fv-0.707107_R0.2 copy.txt', 'Fh0.707107_Fv-0.707107_R0.2 copy 2.txt', 'Fh0.707107_Fv-0.707107_R0.2 copy 3.txt', 'Fh0.707107_Fv0.707107_R0.25.txt', 'Fh0.707107_Fv0.707107_R0.25 copy.txt', 'Fh0.707107_Fv0.707107_R0.25 copy 2.txt', 'Fh0.707107_Fv0.707107_R0.25 copy 3.txt', 'Fh0.92388_Fv0.382683_R0.25.txt', 'Fh0.92388_Fv0.382683_R0.25 copy.txt', 'Fh0.92388_Fv0.382683_R0.25 copy 2.txt', 'Fh0.92388_Fv0.382683_R0.25 copy 3.txt', 'Fh1_Fv0_R0.25.txt', 'Fh1_Fv0_R0.25 copy.txt', 'Fh1_Fv0_R0.25 copy 2.txt', 'Fh1_Fv0_R0.25 copy 3.txt', 'Fh0.92388_Fv-0.382683_R0.25.txt', 'Fh0.92388_Fv-0.382683_R0.25 copy.txt', 'Fh0.92388_Fv-0.382683_R0.25 copy 2.txt', 'Fh0.92388_Fv-0.382683_R0.25 copy 3.txt', 'Fh0.707107_Fv-0.707107_R0.25.txt', 'Fh0.707107_Fv-0.707107_R0.25 copy.txt', 'Fh0.707107_Fv-0.707107_R0.25 copy 2.txt', 'Fh0.707107_Fv-0.707107_R0.25 copy 3.txt']
    interp9ang6pos = ['Fh0_Fv1_R0.txt', 'Fh0_Fv1_R0 copy.txt', 'Fh0_Fv1_R0 copy 2.txt', 'Fh0_Fv1_R0 copy 3.txt', 'Fh0.382683_Fv0.92388_R0.txt', 'Fh0.382683_Fv0.92388_R0 copy.txt', 'Fh0.382683_Fv0.92388_R0 copy 2.txt', 'Fh0.382683_Fv0.92388_R0 copy 3.txt', 'Fh0.707107_Fv0.707107_R0.txt', 'Fh0.707107_Fv0.707107_R0 copy.txt', 'Fh0.707107_Fv0.707107_R0 copy 2.txt', 'Fh0.707107_Fv0.707107_R0 copy 3.txt', 'Fh0.92388_Fv0.382683_R0.txt', 'Fh0.92388_Fv0.382683_R0 copy.txt', 'Fh0.92388_Fv0.382683_R0 copy 2.txt', 'Fh0.92388_Fv0.382683_R0 copy 3.txt', 'Fh1_Fv0_R0.txt', 'Fh1_Fv0_R0 copy.txt', 'Fh1_Fv0_R0 copy 2.txt', 'Fh1_Fv0_R0 copy 3.txt', 'Fh0.92388_Fv-0.382683_R0.txt', 'Fh0.92388_Fv-0.382683_R0 copy.txt', 'Fh0.92388_Fv-0.382683_R0 copy 2.txt', 'Fh0.92388_Fv-0.382683_R0 copy 3.txt', 'Fh0.707107_Fv-0.707107_R0.txt', 'Fh0.707107_Fv-0.707107_R0 copy.txt', 'Fh0.707107_Fv-0.707107_R0 copy 2.txt', 'Fh0.707107_Fv-0.707107_R0 copy 3.txt', 'Fh0.382683_Fv-0.92388_R0.txt', 'Fh0.382683_Fv-0.92388_R0 copy.txt', 'Fh0.382683_Fv-0.92388_R0 copy 2.txt', 'Fh0.382683_Fv-0.92388_R0 copy 3.txt', 'Fh0_Fv-1_R0.txt', 'Fh0_Fv-1_R0 copy.txt', 'Fh0_Fv-1_R0 copy 2.txt', 'Fh0_Fv-1_R0 copy 3.txt', 'Fh0_Fv1_R0.05.txt', 'Fh0_Fv1_R0.05 copy.txt', 'Fh0_Fv1_R0.05 copy 2.txt', 'Fh0_Fv1_R0.05 copy 3.txt', 'Fh0.382683_Fv0.92388_R0.05.txt', 'Fh0.382683_Fv0.92388_R0.05 copy.txt', 'Fh0.382683_Fv0.92388_R0.05 copy 2.txt', 'Fh0.382683_Fv0.92388_R0.05 copy 3.txt', 'Fh0.707107_Fv0.707107_R0.05.txt', 'Fh0.707107_Fv0.707107_R0.05 copy.txt', 'Fh0.707107_Fv0.707107_R0.05 copy 2.txt', 'Fh0.707107_Fv0.707107_R0.05 copy 3.txt', 'Fh0.92388_Fv0.382683_R0.05.txt', 'Fh0.92388_Fv0.382683_R0.05 copy.txt', 'Fh0.92388_Fv0.382683_R0.05 copy 2.txt', 'Fh0.92388_Fv0.382683_R0.05 copy 3.txt', 'Fh1_Fv0_R0.05.txt', 'Fh1_Fv0_R0.05 copy.txt', 'Fh1_Fv0_R0.05 copy 2.txt', 'Fh1_Fv0_R0.05 copy 3.txt', 'Fh0.92388_Fv-0.382683_R0.05.txt', 'Fh0.92388_Fv-0.382683_R0.05 copy.txt', 'Fh0.92388_Fv-0.382683_R0.05 copy 2.txt', 'Fh0.92388_Fv-0.382683_R0.05 copy 3.txt', 'Fh0.707107_Fv-0.707107_R0.05.txt', 'Fh0.707107_Fv-0.707107_R0.05 copy.txt', 'Fh0.707107_Fv-0.707107_R0.05 copy 2.txt', 'Fh0.707107_Fv-0.707107_R0.05 copy 3.txt', 'Fh0.382683_Fv-0.92388_R0.05.txt', 'Fh0.382683_Fv-0.92388_R0.05 copy.txt', 'Fh0.382683_Fv-0.92388_R0.05 copy 2.txt', 'Fh0.382683_Fv-0.92388_R0.05 copy 3.txt', 'Fh0_Fv-1_R0.05.txt', 'Fh0_Fv-1_R0.05 copy.txt', 'Fh0_Fv-1_R0.05 copy 2.txt', 'Fh0_Fv-1_R0.05 copy 3.txt', 'Fh0_Fv1_R0.1.txt', 'Fh0_Fv1_R0.1 copy.txt', 'Fh0_Fv1_R0.1 copy 2.txt', 'Fh0_Fv1_R0.1 copy 3.txt', 'Fh0.382683_Fv0.92388_R0.1.txt', 'Fh0.382683_Fv0.92388_R0.1 copy.txt', 'Fh0.382683_Fv0.92388_R0.1 copy 2.txt', 'Fh0.382683_Fv0.92388_R0.1 copy 3.txt', 'Fh0.707107_Fv0.707107_R0.1.txt', 'Fh0.707107_Fv0.707107_R0.1 copy.txt', 'Fh0.707107_Fv0.707107_R0.1 copy 2.txt', 'Fh0.707107_Fv0.707107_R0.1 copy 3.txt', 'Fh0.92388_Fv0.382683_R0.1.txt', 'Fh0.92388_Fv0.382683_R0.1 copy.txt', 'Fh0.92388_Fv0.382683_R0.1 copy 2.txt', 'Fh0.92388_Fv0.382683_R0.1 copy 3.txt', 'Fh1_Fv0_R0.1.txt', 'Fh1_Fv0_R0.1 copy.txt', 'Fh1_Fv0_R0.1 copy 2.txt', 'Fh1_Fv0_R0.1 copy 3.txt', 'Fh0.92388_Fv-0.382683_R0.1.txt', 'Fh0.92388_Fv-0.382683_R0.1 copy.txt', 'Fh0.92388_Fv-0.382683_R0.1 copy 2.txt', 'Fh0.92388_Fv-0.382683_R0.1 copy 3.txt', 'Fh0.707107_Fv-0.707107_R0.1.txt', 'Fh0.707107_Fv-0.707107_R0.1 copy.txt', 'Fh0.707107_Fv-0.707107_R0.1 copy 2.txt', 'Fh0.707107_Fv-0.707107_R0.1 copy 3.txt', 'Fh0.382683_Fv-0.92388_R0.1.txt', 'Fh0.382683_Fv-0.92388_R0.1 copy.txt', 'Fh0.382683_Fv-0.92388_R0.1 copy 2.txt', 'Fh0.382683_Fv-0.92388_R0.1 copy 3.txt', 'Fh0_Fv-1_R0.1.txt', 'Fh0_Fv-1_R0.1 copy.txt', 'Fh0_Fv-1_R0.1 copy 2.txt', 'Fh0_Fv-1_R0.1 copy 3.txt', 'Fh0_Fv1_R0.15.txt', 'Fh0_Fv1_R0.15 copy.txt', 'Fh0_Fv1_R0.15 copy 2.txt', 'Fh0_Fv1_R0.15 copy 3.txt', 'Fh0.382683_Fv0.92388_R0.15.txt', 'Fh0.382683_Fv0.92388_R0.15 copy.txt', 'Fh0.382683_Fv0.92388_R0.15 copy 2.txt', 'Fh0.382683_Fv0.92388_R0.15 copy 3.txt', 'Fh0.707107_Fv0.707107_R0.15.txt', 'Fh0.707107_Fv0.707107_R0.15 copy.txt', 'Fh0.707107_Fv0.707107_R0.15 copy 2.txt', 'Fh0.707107_Fv0.707107_R0.15 copy 3.txt', 'Fh0.92388_Fv0.382683_R0.15.txt', 'Fh0.92388_Fv0.382683_R0.15 copy.txt', 'Fh0.92388_Fv0.382683_R0.15 copy 2.txt', 'Fh0.92388_Fv0.382683_R0.15 copy 3.txt', 'Fh1_Fv0_R0.15.txt', 'Fh1_Fv0_R0.15 copy.txt', 'Fh1_Fv0_R0.15 copy 2.txt', 'Fh1_Fv0_R0.15 copy 3.txt', 'Fh0.92388_Fv-0.382683_R0.15.txt', 'Fh0.92388_Fv-0.382683_R0.15 copy.txt', 'Fh0.92388_Fv-0.382683_R0.15 copy 2.txt', 'Fh0.92388_Fv-0.382683_R0.15 copy 3.txt', 'Fh0.707107_Fv-0.707107_R0.15.txt', 'Fh0.707107_Fv-0.707107_R0.15 copy.txt', 'Fh0.707107_Fv-0.707107_R0.15 copy 2.txt', 'Fh0.707107_Fv-0.707107_R0.15 copy 3.txt', 'Fh0.382683_Fv-0.92388_R0.15.txt', 'Fh0.382683_Fv-0.92388_R0.15 copy.txt', 'Fh0.382683_Fv-0.92388_R0.15 copy 2.txt', 'Fh0.382683_Fv-0.92388_R0.15 copy 3.txt', 'Fh0_Fv-1_R0.15.txt', 'Fh0_Fv-1_R0.15 copy.txt', 'Fh0_Fv-1_R0.15 copy 2.txt', 'Fh0_Fv-1_R0.15 copy 3.txt', 'Fh0_Fv1_R0.2.txt', 'Fh0_Fv1_R0.2 copy.txt', 'Fh0_Fv1_R0.2 copy 2.txt', 'Fh0_Fv1_R0.2 copy 3.txt', 'Fh0.382683_Fv0.92388_R0.2.txt', 'Fh0.382683_Fv0.92388_R0.2 copy.txt', 'Fh0.382683_Fv0.92388_R0.2 copy 2.txt', 'Fh0.382683_Fv0.92388_R0.2 copy 3.txt', 'Fh0.707107_Fv0.707107_R0.2.txt', 'Fh0.707107_Fv0.707107_R0.2 copy.txt', 'Fh0.707107_Fv0.707107_R0.2 copy 2.txt', 'Fh0.707107_Fv0.707107_R0.2 copy 3.txt', 'Fh0.92388_Fv0.382683_R0.2.txt', 'Fh0.92388_Fv0.382683_R0.2 copy.txt', 'Fh0.92388_Fv0.382683_R0.2 copy 2.txt', 'Fh0.92388_Fv0.382683_R0.2 copy 3.txt', 'Fh1_Fv0_R0.2.txt', 'Fh1_Fv0_R0.2 copy.txt', 'Fh1_Fv0_R0.2 copy 2.txt', 'Fh1_Fv0_R0.2 copy 3.txt', 'Fh0.92388_Fv-0.382683_R0.2.txt', 'Fh0.92388_Fv-0.382683_R0.2 copy.txt', 'Fh0.92388_Fv-0.382683_R0.2 copy 2.txt', 'Fh0.92388_Fv-0.382683_R0.2 copy 3.txt', 'Fh0.707107_Fv-0.707107_R0.2.txt', 'Fh0.707107_Fv-0.707107_R0.2 copy.txt', 'Fh0.707107_Fv-0.707107_R0.2 copy 2.txt', 'Fh0.707107_Fv-0.707107_R0.2 copy 3.txt', 'Fh0.382683_Fv-0.92388_R0.2.txt', 'Fh0.382683_Fv-0.92388_R0.2 copy.txt', 'Fh0.382683_Fv-0.92388_R0.2 copy 2.txt', 'Fh0.382683_Fv-0.92388_R0.2 copy 3.txt', 'Fh0_Fv-1_R0.2.txt', 'Fh0_Fv-1_R0.2 copy.txt', 'Fh0_Fv-1_R0.2 copy 2.txt', 'Fh0_Fv-1_R0.2 copy 3.txt', 'Fh0_Fv1_R0.25.txt', 'Fh0_Fv1_R0.25 copy.txt', 'Fh0_Fv1_R0.25 copy 2.txt', 'Fh0_Fv1_R0.25 copy 3.txt', 'Fh0.382683_Fv0.92388_R0.25.txt', 'Fh0.382683_Fv0.92388_R0.25 copy.txt', 'Fh0.382683_Fv0.92388_R0.25 copy 2.txt', 'Fh0.382683_Fv0.92388_R0.25 copy 3.txt', 'Fh0.707107_Fv0.707107_R0.25.txt', 'Fh0.707107_Fv0.707107_R0.25 copy.txt', 'Fh0.707107_Fv0.707107_R0.25 copy 2.txt', 'Fh0.707107_Fv0.707107_R0.25 copy 3.txt', 'Fh0.92388_Fv0.382683_R0.25.txt', 'Fh0.92388_Fv0.382683_R0.25 copy.txt', 'Fh0.92388_Fv0.382683_R0.25 copy 2.txt', 'Fh0.92388_Fv0.382683_R0.25 copy 3.txt', 'Fh1_Fv0_R0.25.txt', 'Fh1_Fv0_R0.25 copy.txt', 'Fh1_Fv0_R0.25 copy 2.txt', 'Fh1_Fv0_R0.25 copy 3.txt', 'Fh0.92388_Fv-0.382683_R0.25.txt', 'Fh0.92388_Fv-0.382683_R0.25 copy.txt', 'Fh0.92388_Fv-0.382683_R0.25 copy 2.txt', 'Fh0.92388_Fv-0.382683_R0.25 copy 3.txt', 'Fh0.707107_Fv-0.707107_R0.25.txt', 'Fh0.707107_Fv-0.707107_R0.25 copy.txt', 'Fh0.707107_Fv-0.707107_R0.25 copy 2.txt', 'Fh0.707107_Fv-0.707107_R0.25 copy 3.txt', 'Fh0.382683_Fv-0.92388_R0.25.txt', 'Fh0.382683_Fv-0.92388_R0.25 copy.txt', 'Fh0.382683_Fv-0.92388_R0.25 copy 2.txt', 'Fh0.382683_Fv-0.92388_R0.25 copy 3.txt', 'Fh0_Fv-1_R0.25.txt', 'Fh0_Fv-1_R0.25 copy.txt', 'Fh0_Fv-1_R0.25 copy 2.txt', 'Fh0_Fv-1_R0.25 copy 3.txt']
    # testData = ['Fh0.707107_Fv0.707107_R1.0.txt', 'Fh0.707107_Fv0.707107_R1.0 copy.txt', 'Fh0.707107_Fv0.707107_R1.0 copy 2.txt', 'Fh0.707107_Fv0.707107_R1.0 copy 3.txt', 'Fh0.92388_Fv0.382683_R1.0.txt', 'Fh0.92388_Fv0.382683_R1.0 copy.txt', 'Fh0.92388_Fv0.382683_R1.0 copy 2.txt', 'Fh0.92388_Fv0.382683_R1.0 copy 3.txt', 'Fh1_Fv0_R1.0.txt', 'Fh1_Fv0_R1.0 copy.txt', 'Fh1_Fv0_R1.0 copy 2.txt', 'Fh1_Fv0_R1.0 copy 3.txt', 'Fh0.92388_Fv-0.382683_R1.0.txt', 'Fh0.92388_Fv-0.382683_R1.0 copy.txt', 'Fh0.92388_Fv-0.382683_R1.0 copy 2.txt', 'Fh0.92388_Fv-0.382683_R1.0 copy 3.txt', 'Fh0.707107_Fv-0.707107_R1.0.txt', 'Fh0.707107_Fv-0.707107_R1.0 copy.txt', 'Fh0.707107_Fv-0.707107_R1.0 copy 2.txt', 'Fh0.707107_Fv-0.707107_R1.0 copy 3.txt', 'Fh0.707107_Fv0.707107_R0.95.txt', 'Fh0.707107_Fv0.707107_R0.95 copy.txt', 'Fh0.707107_Fv0.707107_R0.95 copy 2.txt', 'Fh0.707107_Fv0.707107_R0.95 copy 3.txt', 'Fh0.92388_Fv0.382683_R0.95.txt', 'Fh0.92388_Fv0.382683_R0.95 copy.txt', 'Fh0.92388_Fv0.382683_R0.95 copy 2.txt', 'Fh0.92388_Fv0.382683_R0.95 copy 3.txt', 'Fh1_Fv0_R0.95.txt', 'Fh1_Fv0_R0.95 copy.txt', 'Fh1_Fv0_R0.95 copy 2.txt', 'Fh1_Fv0_R0.95 copy 3.txt', 'Fh0.92388_Fv-0.382683_R0.95.txt', 'Fh0.92388_Fv-0.382683_R0.95 copy.txt', 'Fh0.92388_Fv-0.382683_R0.95 copy 2.txt', 'Fh0.92388_Fv-0.382683_R0.95 copy 3.txt', 'Fh0.707107_Fv-0.707107_R0.95.txt', 'Fh0.707107_Fv-0.707107_R0.95 copy.txt', 'Fh0.707107_Fv-0.707107_R0.95 copy 2.txt', 'Fh0.707107_Fv-0.707107_R0.95 copy 3.txt', 'Fh0.707107_Fv0.707107_R0.9.txt', 'Fh0.707107_Fv0.707107_R0.9 copy.txt', 'Fh0.707107_Fv0.707107_R0.9 copy 2.txt', 'Fh0.707107_Fv0.707107_R0.9 copy 3.txt', 'Fh0.92388_Fv0.382683_R0.9.txt', 'Fh0.92388_Fv0.382683_R0.9 copy.txt', 'Fh0.92388_Fv0.382683_R0.9 copy 2.txt', 'Fh0.92388_Fv0.382683_R0.9 copy 3.txt', 'Fh1_Fv0_R0.9.txt', 'Fh1_Fv0_R0.9 copy.txt', 'Fh1_Fv0_R0.9 copy 2.txt', 'Fh1_Fv0_R0.9 copy 3.txt', 'Fh0.92388_Fv-0.382683_R0.9.txt', 'Fh0.92388_Fv-0.382683_R0.9 copy.txt', 'Fh0.92388_Fv-0.382683_R0.9 copy 2.txt', 'Fh0.92388_Fv-0.382683_R0.9 copy 3.txt', 'Fh0.707107_Fv-0.707107_R0.9.txt', 'Fh0.707107_Fv-0.707107_R0.9 copy.txt', 'Fh0.707107_Fv-0.707107_R0.9 copy 2.txt', 'Fh0.707107_Fv-0.707107_R0.9 copy 3.txt', 'Fh0.707107_Fv0.707107_R0.85.txt', 'Fh0.707107_Fv0.707107_R0.85 copy.txt', 'Fh0.707107_Fv0.707107_R0.85 copy 2.txt', 'Fh0.707107_Fv0.707107_R0.85 copy 3.txt', 'Fh0.92388_Fv0.382683_R0.85.txt', 'Fh0.92388_Fv0.382683_R0.85 copy.txt', 'Fh0.92388_Fv0.382683_R0.85 copy 2.txt', 'Fh0.92388_Fv0.382683_R0.85 copy 3.txt', 'Fh1_Fv0_R0.85.txt', 'Fh1_Fv0_R0.85 copy.txt', 'Fh1_Fv0_R0.85 copy 2.txt', 'Fh1_Fv0_R0.85 copy 3.txt', 'Fh0.92388_Fv-0.382683_R0.85.txt', 'Fh0.92388_Fv-0.382683_R0.85 copy.txt', 'Fh0.92388_Fv-0.382683_R0.85 copy 2.txt', 'Fh0.92388_Fv-0.382683_R0.85 copy 3.txt', 'Fh0.707107_Fv-0.707107_R0.85.txt', 'Fh0.707107_Fv-0.707107_R0.85 copy.txt', 'Fh0.707107_Fv-0.707107_R0.85 copy 2.txt', 'Fh0.707107_Fv-0.707107_R0.85 copy 3.txt', 'Fh0.707107_Fv0.707107_R0.8.txt', 'Fh0.707107_Fv0.707107_R0.8 copy.txt', 'Fh0.707107_Fv0.707107_R0.8 copy 2.txt', 'Fh0.707107_Fv0.707107_R0.8 copy 3.txt', 'Fh0.92388_Fv0.382683_R0.8.txt', 'Fh0.92388_Fv0.382683_R0.8 copy.txt', 'Fh0.92388_Fv0.382683_R0.8 copy 2.txt', 'Fh0.92388_Fv0.382683_R0.8 copy 3.txt', 'Fh1_Fv0_R0.8.txt', 'Fh1_Fv0_R0.8 copy.txt', 'Fh1_Fv0_R0.8 copy 2.txt', 'Fh1_Fv0_R0.8 copy 3.txt', 'Fh0.92388_Fv-0.382683_R0.8.txt', 'Fh0.92388_Fv-0.382683_R0.8 copy.txt', 'Fh0.92388_Fv-0.382683_R0.8 copy 2.txt', 'Fh0.92388_Fv-0.382683_R0.8 copy 3.txt', 'Fh0.707107_Fv-0.707107_R0.8.txt', 'Fh0.707107_Fv-0.707107_R0.8 copy.txt', 'Fh0.707107_Fv-0.707107_R0.8 copy 2.txt', 'Fh0.707107_Fv-0.707107_R0.8 copy 3.txt', 'Fh0.707107_Fv0.707107_R0.75.txt', 'Fh0.707107_Fv0.707107_R0.75 copy.txt', 'Fh0.707107_Fv0.707107_R0.75 copy 2.txt', 'Fh0.707107_Fv0.707107_R0.75 copy 3.txt', 'Fh0.92388_Fv0.382683_R0.75.txt', 'Fh0.92388_Fv0.382683_R0.75 copy.txt', 'Fh0.92388_Fv0.382683_R0.75 copy 2.txt', 'Fh0.92388_Fv0.382683_R0.75 copy 3.txt', 'Fh1_Fv0_R0.75.txt', 'Fh1_Fv0_R0.75 copy.txt', 'Fh1_Fv0_R0.75 copy 2.txt', 'Fh1_Fv0_R0.75 copy 3.txt', 'Fh0.92388_Fv-0.382683_R0.75.txt', 'Fh0.92388_Fv-0.382683_R0.75 copy.txt', 'Fh0.92388_Fv-0.382683_R0.75 copy 2.txt', 'Fh0.92388_Fv-0.382683_R0.75 copy 3.txt', 'Fh0.707107_Fv-0.707107_R0.75.txt', 'Fh0.707107_Fv-0.707107_R0.75 copy.txt', 'Fh0.707107_Fv-0.707107_R0.75 copy 2.txt', 'Fh0.707107_Fv-0.707107_R0.75 copy 3.txt']
    data = Data(dataset, testData=extrap9ang5pos, verbose=True, saveInfo=True)
    data.load()
    # data.thresholdFilter(tol=0)
    data.rehsapeDataToArray()

    fcnn = FCNN(data,verbose=True, saveInfo=True)
    fcnn.build(codeSize=25, nNeurons=200, nHidLayers=4, regularisation=1e-4)

    model = Model(fcnn)
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.train(epochs=500, nBatch=16, earlyStopPatience=50, earlyStopTol=1e-4)
    model.predict()

    nDisplay = 5
    if plotPredictions:
        plottingPrediction(data, model, nDisplay)

