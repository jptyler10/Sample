import torch
import torch.nn as nn
import numpy as np
import modules as m

class MODEL(nn.Module):

    def __init__(self, parameters):

        super(MODEL, self).__init__()

        self.global_network = m.GlobalNetwork(parameters)

        self.retrieve_roi_crops = m.RetrieveROIModule(parameters)

        self.local_network = m.LocalNetwork(parameters)

    def _retrieve_crop(self, x_original, crop_positions):

        batch_size, num_crops, _ = crop_positions.shape

        crop_h = 10
        crop_w = 50

        output = torch.ones((batch_size, num_crops, crop_h, crop_w))

        for i in range(batch_size):
            for j in range(num_crops):
                output[i,j,:,:] = x_original[i,0,crop_positions[i,j,0]:crop_positions[i,j,0]+crop_h, :]             

        return output

    def forward(self, x_original):

        y_global, self.saliency_map = self.global_network.forward(x_original)

        x_locations = self.retrieve_roi_crops.forward(x_original, self.saliency_map)

        crops_variable = self._retrieve_crop(x_original, x_locations)

        batch_size, num_crops, I, J = crops_variable.size()

#        crops_variable = crops_variable.view(batch_size * num_crops, I, J).unsqueeze(1)

        h_crops = self.local_network.forward(crops_variable)


        return y_global.view(y_global.shape[0],-1) , h_crops, x_locations
