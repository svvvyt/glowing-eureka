import torch.nn as nn
from .modules.transformation import TPS_SpatialTransformerNetwork
from .modules.feature_extraction import VGG_FeatureExtractor
from .modules.sequence_modeling import BidirectionalLSTM
from .modules.prediction import Attention

class Model(nn.Module):

    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        
        """ Transformation """
        self.Transformation = TPS_SpatialTransformerNetwork(
            F=opt.num_fiducial, I_size=(opt.imgH, opt.imgW), I_r_size=(opt.imgH, opt.imgW), I_channel_num=opt.input_channel)

        """ FeatureExtraction """
        self.FeatureExtraction = VGG_FeatureExtractor(opt.input_channel, opt.output_channel)
        self.FeatureExtraction_output = opt.output_channel  # int(imgH/16-1) * 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1

        """ Sequence modeling"""
        self.SequenceModeling = nn.Sequential(
            BidirectionalLSTM(self.FeatureExtraction_output, opt.hidden_size, opt.hidden_size),
            BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size))
        self.SequenceModeling_output = opt.hidden_size
        
        """ Prediction """
        self.Prediction = Attention(self.SequenceModeling_output, opt.hidden_size, opt.num_class)
        

    def forward(self, input, text, is_train=True):
        """ Transformation stage """
        input = self.Transformation(input)

        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)

        """ Sequence modeling stage """
        contextual_feature = self.SequenceModeling(visual_feature)
        
        """ Prediction stage """
        prediction = self.Prediction(contextual_feature.contiguous(), text, is_train, batch_max_length=self.opt.batch_max_length)

        return prediction
