import torch
import torch.nn as nn
import models.pytorch_prototyping as pytorch_prototyping

class Refinement(nn.Module):
    def __init__(self):
        super(Refinement, self).__init__()

        self.RefinementNetwork = pytorch_prototyping.Unet(in_channels=2,
                                                          out_channels=1,
                                                          nf0=4,
                                                          num_down=4,
                                                          max_channels=512,
                                                          use_dropout=False,
                                                          added_channels=2)

    def forward(self, batch_data):
        bproxi = batch_data['bproxi']
        mdi = batch_data['mdi']

        depth_concat = torch.cat((mdi, bproxi), dim=1)
        return self.RefinementNetwork(depth_concat)
