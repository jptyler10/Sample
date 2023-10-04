import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

def get_max_window(input_image, window_shape):

    N, C, H, W = input_image.size()
    # use average pooling to locate the window sums
    pool_map = torch.nn.functional.avg_pool2d(input_image, window_shape, stride=1)
    _, _, _, W_map = pool_map.size()
    # transform to linear and get the index of the max val locations
    _, max_linear_idx = torch.max(pool_map.view(N, C, -1), -1)
    # convert back to 2d index
    max_idx_x = max_linear_idx // W_map
    max_idx_y = max_linear_idx - max_idx_x * W_map
    # put together the 2d index
    upper_left_points = torch.cat([max_idx_x.unsqueeze(-1), max_idx_y.unsqueeze(-1)], dim=-1)
    return upper_left_points

def generate_mask_uplft(input_image, window_shape, upper_left_points):

    N, C, H, W = input_image.size()
    window_h, window_w = window_shape
    # get the positions of masks
    mask_x_min = upper_left_points[:,:,0]
    mask_x_max = upper_left_points[:,:,0] + window_h
    mask_y_min = upper_left_points[:,:,1]
    mask_y_max = upper_left_points[:,:,1] + window_w
    # generate masks
    mask_x = Variable(torch.arange(0, H).view(-1, 1).repeat(N, C, 1, W))
    mask_y = Variable(torch.arange(0, W).view(1, -1).repeat(N, C, H, 1))
    x_gt_min = mask_x.float() >= mask_x_min.unsqueeze(-1).unsqueeze(-1).float()
    x_ls_max = mask_x.float() < mask_x_max.unsqueeze(-1).unsqueeze(-1).float()
    y_gt_min = mask_y.float() >= mask_y_min.unsqueeze(-1).unsqueeze(-1).float()
    y_ls_max = mask_y.float() < mask_y_max.unsqueeze(-1).unsqueeze(-1).float()

    # since logic operation is not supported for variable
    # I used * for logic ANd
    selected_x = x_gt_min * x_ls_max
    selected_y = y_gt_min * y_ls_max
    selected = selected_x * selected_y
    mask = 1 - selected.float()
    return mask

class AbstractMILUnit:

    def __init__(self, parameters, parent_module):
        self.parameters = parameters
        self.parent_module = parent_module

class TopTPercentAggregationFunction(nn.Module):

    def __init__(self, parameters):
        super(TopTPercentAggregationFunction, self).__init__(parameters, parent_module)
        self.percent_t = parameters['percent_t']
        self.parent_module = parent_module

    def forward(self, cam):
        batch_size, num_class, H, W = cam.size()
        cam_flatten = cam.view(batch_size, num_class, -1)
        top_t = int(round(W*H*self.percent_t))
        selected_area = cam_flatten.topk(top_t, dim=2)[0]
        return selected_area.mean(dim=2)
       

class RetrieveROIModule():

    def __init__(self, parameters):
        self.num_crops_per_class = parameters['K']

    def forward(self, x_original, h_small):

        # retrieve parameters

        current_images = h_small
        all_max_position = []

        max_vals = current_images.view(1,1,-1).max(dim=2, keepdim=True)[0].unsqueeze(-1)
        min_vals = current_images.view(1,1,-1).min(dim=2,keepdim=True)[0].unsqueeze(-1)
        range_vals = max_vals-min_vals
        normalize_images = current_images-min_vals
        normalize_images = normalize_images / range_vals

        current_images = normalize_images

        for _ in range(self.num_crops_per_class):
            max_pos = get_max_window(current_images, (10,50))
            all_max_position.append(max_pos)
            mask = generate_mask_uplft(current_images, (10,50), max_pos)
            current_images = current_images * mask

        return torch.cat(all_max_position, dim=1).data.cpu().numpy()


class GlobalNetwork(nn.Module):

    def __init__(self, parameters):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 1, 5, stride=1, padding=2, dilation=1)        
        self.conv2 = nn.Conv2d(1, 1, 2, stride=1, padding=1, dilation=2)
        self.conv1by1 = nn.Conv2d(1, 1, 1, stride=1,padding=0)
        self.sig = nn.Sigmoid()

        self.avg_pool = nn.AdaptiveMaxPool2d((500,50))

        self.linear1 = nn.Linear(25000, 512)
        self.linear2 = nn.Linear(512,1)
        self.linear3 = nn.Linear(128,1)

        self.relu = nn.ReLU()

    def forward(self, x):

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.sig(self.conv1by1(out))

        after_pool = self.avg_pool(out).view(out.shape[0],out.shape[1],-1)
        linear1 = self.linear1(after_pool)
        linear1 = self.relu(linear1)
        final = self.linear2(linear1)

        final = self.sig(final)

        return final,out

class LocalNetwork(nn.Module):

    def __init__(self, parameters):
        super().__init__()

        self.conv_local1 = nn.Conv2d(20,10,3)
        self.conv_local2 = nn.Conv2d(10,5,2)
        self.conv_local3 = nn.Conv2d(5,1,2)

        self.dense_local1 = nn.Linear(276,128)
        self.dense_local2 = nn.Linear(128,64)
        self.dense_local3 = nn.Linear(64,1)
        self.relu = nn.ReLU()

        self.lstm1 = nn.LSTM(50,25,2,bidirectional=True)
        self.dense = nn.Linear(250, 50)

        self.final = nn.Linear(50,1)

        self.sig = nn.Sigmoid()

    def forward(self,x):

        x = self.conv_local1(x)
        x = self.conv_local2(x)
        x = self.conv_local3(x)

        x = x.view((x.shape[0], 1, -1))

        x = self.dense_local1(x)
        x = self.relu(x)

        x = self.dense_local2(x)
        x = self.relu(x)
        

        x_final = self.sig(self.dense_local3(x))

        x_final = x_final.view(x_final.shape[0],-1)
        return x_final
        
        
