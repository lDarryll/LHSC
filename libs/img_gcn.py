import torch
from torch import nn
from torch.nn import functional as F
import numpy as np



class IM_GCN(nn.Module):

    def __init__(self, in_channels, inter_channels, bn_layer=True):
        super(IM_GCN, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        conv_nd = nn.Conv1d
        max_pool = nn.MaxPool1d
        bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)


        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant(self.W.weight, 0)
            nn.init.constant(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels/4,
                             kernel_size=1, stride=1, padding=0)
        self.theta1 = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels/4,
                             kernel_size=1, stride=1, padding=0)
        self.theta2 = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels / 4,
                              kernel_size=1, stride=1, padding=0)
        self.theta3 = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels / 4,
                              kernel_size=1, stride=1, padding=0)

        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels/4,
                           kernel_size=1, stride=1, padding=0)
        self.phi1 = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels/4,
                           kernel_size=1, stride=1, padding=0)
        self.phi2 = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels / 4,
                            kernel_size=1, stride=1, padding=0)
        self.phi3 = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels / 4,
                            kernel_size=1, stride=1, padding=0)

        self.fc1 = nn.Linear(self.in_channels, self.inter_channels)
        self.fc2 = nn.Linear(self.in_channels, self.inter_channels)

    def init_weights(self):
        """Xavier initialization for the fully connected layer"""
        r = np.sqrt(6.) / np.sqrt(self.fc1.in_features +
                                  self.fc1.out_features)

        r1 = np.sqrt(6.) / np.sqrt(self.fc2.in_features +
                                  self.fc2.out_features)

        self.fc1.weight.data.uniform_(-r, r)
        self.fc1.bias.data.fill_(0)

        self.fc2.weight.data.uniform_(-r1, r1)
        self.fc2.bias.data.fill_(0)



    def forward(self, v):
        '''
        :param v: (B, D, N)
        :return:
        '''
        batch_size = v.size(0)

        g_v = self.g(v).view(batch_size, self.inter_channels, -1)
        g_v = g_v.permute(0, 2, 1)

        theta_v = self.theta(v).view(batch_size, self.inter_channels/4, -1)
        theta_v1 = self.theta1(v).view(batch_size, self.inter_channels/4, -1)
        theta_v2 = self.theta2(v).view(batch_size, self.inter_channels/4, -1)
        theta_v3 = self.theta3(v).view(batch_size, self.inter_channels/4, -1)
        
        phi_v = self.phi(v).view(batch_size, self.inter_channels/4, -1)
        phi_v1 = self.phi1(v).view(batch_size, self.inter_channels/4, -1)
        phi_v2 = self.phi2(v).view(batch_size, self.inter_channels/4, -1)
        phi_v3 = self.phi3(v).view(batch_size, self.inter_channels/4, -1)
        a = torch.cat((theta_v,theta_v1,theta_v2,theta_v3),1)
        a = a.permute(0,2,1)
        a = self.fc1(a)

        b = torch.cat((phi_v,phi_v1,phi_v2,phi_v3),1)
        b = b.permute(0, 2, 1)
        b = self.fc2(b)
        b = b.permute(0, 2, 1)
        R = torch.matmul(a,b)
        N = R.size(-1)
        R_div_C = R / N

        y = torch.matmul(R_div_C, g_v)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *v.size()[2:])
        W_y = self.W(y)
        v_star = W_y + v

        return v_star








