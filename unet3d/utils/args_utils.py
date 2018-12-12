import argparse
from .utils import str2bool



def train():
    parser = argparse.ArgumentParser(description='Data preparation')
    parser.add_argument('-c', '--competition', type=str,
                        choices=["brats"], default="brats",
                        help="competition")
    parser.add_argument('-y', '--year', type=str,
                        default=2018,
                        help="year of challenge")
    parser.add_argument('-o', '--overwrite', type=str2bool,
                        default="True")
    parser.add_argument('-i', '--inputshape', type=str,
                        default="240-240-155",
                        help="input shape to read")
    parser.add_argument('-b', '--isbiascorrection', type=str2bool,
                        default="True",
                        help="perform bias field removal?")
    parser.add_argument('-n', '--normalization', type=str,
                        choices=["Z", "01", "False"], default="Z",
                        help="what type of normalization")
    parser.add_argument('-a', '--clahe', type=str2bool,
                        default="False",
                        help="perform Contrast Limited Adaptive Histogram Equalization")
    parser.add_argument('-hm', '--histmatch', type=str2bool,
                        default="False",
                        help="perform histogram matching")
    args = parser.parse_args()
    return args

def prepare_data():
    parser = argparse.ArgumentParser(description='Data preparation')
    parser.add_argument('-c', '--challenge', type=str,
                        choices=[2018], default=2018,
                        help="year of brats challenge")
    parser.add_argument('-d', '--dataset', type=str,
                        default="test",
                        help="dataset type")
    parser.add_argument('-i', '--inms', type=str2bool,
                        default="False",
                        help="is normalize mean, standard deviation")
    parser.add_argument('-o', '--overwrite', type=str2bool,
                        default="True")                        
    args = parser.parse_args()
    return args    