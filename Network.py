import torch.nn.functional as F
import torch.nn as nn

class NormConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(NormConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class NormUpConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(NormUpConv2d, self).__init__()
        self.transconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size = 3, stride = 2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.activation1 = nn.ReLU()

        self.conv1 = NormConv2d(out_channels, out_channels, kernel_size = 2)
        self.conv2 = NormConv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
        
    def forward(self, x):
        out = self.transconv(x)
        out = self.bn1(out)
        out = self.activation1(out)

        out = self.conv1(out)
        out = self.conv2(out)

        return out

class Attention_block(nn.Module):
    def __init__(self, in_channels):
        super(Attention_block, self).__init__()
        
        self.block = nn.Sequential(
            NormConv2d(in_channels, in_channels, kernel_size = 3, padding = 1),
            NormConv2d(in_channels, in_channels, kernel_size = 3, padding = 1),
            NormConv2d(in_channels, in_channels, kernel_size = 3, padding = 1),
            NormConv2d(in_channels, in_channels, kernel_size = 3, padding = 1),
            nn.Conv2d(in_channels, 1, kernel_size = 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
            )
        
    def forward(self, x):
        x = self.block(x)
        return x


class Encoder(nn.Module):
    
    def __init__(self):
        super(Encoder, self).__init__()

        self.attention1 = Attention_block(32)
        self.attention2 = Attention_block(32)
        
        self.block_01 = nn.Sequential(
            NormConv2d(3, 32, kernel_size = 3, padding=1),
            NormConv2d(32, 32, kernel_size = 3, padding=1),
            NormConv2d(32, 32, kernel_size = 3, padding=2, dilation=2),
            NormConv2d(32, 32, kernel_size = 3, padding=4, dilation=4)
            )
        
        self.block_02_2 = nn.Sequential(
            NormConv2d(32, 32, kernel_size = 3, padding=1),
            NormConv2d(32, 32, kernel_size = 3, padding=1),
            NormConv2d(32, 32, kernel_size = 3, padding=2, dilation=2),
            NormConv2d(32, 32, kernel_size = 3, padding=4, dilation=4)
            )

        self.block_02 = nn.Sequential(
            NormConv2d(32, 32, kernel_size = 3, padding=1),
            NormConv2d(32, 32, kernel_size = 3, padding=1),
            NormConv2d(32, 32, kernel_size = 3, padding=2, dilation=2),
            NormConv2d(32, 32, kernel_size = 3, padding=4, dilation=4)
            )
        
        
        self.block_03 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            NormConv2d(32, 64, kernel_size = 3, padding=1),
            NormConv2d(64, 64, kernel_size = 3, padding=1),
            NormConv2d(64, 64, kernel_size = 3, padding=1),
            NormConv2d(64, 64, kernel_size = 3, padding=1),
            )
        
        
        self.block_04 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            NormConv2d(64, 128, kernel_size = 3, padding=1),
            NormConv2d(128, 128, kernel_size = 3, padding=1),
            NormConv2d(128, 128, kernel_size = 3, padding=1),
            )
        
        
        self.block_05 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            NormConv2d(128, 256, kernel_size = 3, padding=1),
            NormConv2d(256, 256, kernel_size = 3, padding=1),
            NormConv2d(256, 256, kernel_size = 3, padding=1),
            )
        
        self.transpose_05 = NormUpConv2d(256, 128)
        self.transpose_04 = NormUpConv2d(128, 64)
        self.transpose_03 = NormUpConv2d(64, 32)
        

        self.reduction = NormConv2d(192, 32, kernel_size = 1)
        self.dropout = nn.Dropout(p=0.5, inplace=False)
        
    def forward(self, x):
        spacial_out = self.block_01(x)
        spacial_out1 = self.block_02_2(spacial_out)
        spacial_out2 = self.block_02(spacial_out1)
        
        receptive_out_3 = self.block_03(spacial_out2)
        receptive_out_4 = self.block_04(receptive_out_3)
        receptive_out_5 = self.block_05(receptive_out_4)
        
        t_out = self.transpose_05(receptive_out_5)
        t_out = t_out + receptive_out_4
        t_out = self.transpose_04(t_out)
        t_out = t_out + receptive_out_3
        t_out = self.transpose_03(t_out)

        spacial_x = self.attention1(spacial_out2)
        spacial_out = spacial_out * spacial_x

        receptive_x = self.attention2(t_out)
        t_out = t_out * receptive_x

        final_out = spacial_out + t_out

        return final_out

class segmentation_block(nn.Module):
    def __init__(self, in_channels, classes):
        super(segmentation_block, self).__init__()
        
        self.conv1 = NormConv2d(in_channels, in_channels, kernel_size = 3, padding = 1)
        self.conv1_1 = NormConv2d(in_channels, in_channels, kernel_size = 3, padding = 1)
        self.mp_01 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.conv2 = NormConv2d(in_channels, int(2*in_channels), kernel_size = 3, padding = 1)
        self.conv22 = NormConv2d(int(2*in_channels), int(2*in_channels), kernel_size = 3, padding = 1)
        self.conv33 = NormConv2d(int(2*in_channels), int(2*in_channels), kernel_size = 3, padding = 1)
        self.conv3 = NormConv2d(int(2*in_channels), int(2*in_channels), kernel_size = 3, padding = 1)

        self.down_chan = nn.Conv2d(int(2*in_channels), in_channels, kernel_size = 1)

        self.up_conv = NormUpConv2d(int(2*in_channels), in_channels)
        self.seg_conv = nn.Conv2d(in_channels, 2, kernel_size = 1)
        self.final_act = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.3, inplace = False)

        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv1_1(x)
        x = self.conv1_1(x)
        dx = self.mp_01(x)

        x1 = self.conv2(dx)
        x = self.conv22(x1)
        dropx = self.dropout(x)
        x = self.conv33(dropx)
        x = self.conv3(x+x1)

        x = self.up_conv(x)
        x = self.seg_conv(x)
        x = self.final_act(x)

        return x

class edge_block(nn.Module):
    def __init__(self, in_channels, classes):
        super(edge_block, self).__init__()
        
        self.pre_conv = NormConv2d(in_channels, classes, kernel_size = 3, padding = 1)
        
        self.conv1 = NormConv2d(classes, classes, kernel_size = 3, padding = 1)
        self.mp_01 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.conv2 = NormConv2d(classes, int(2*classes), kernel_size = 3, padding = 1)
        self.conv3 = NormConv2d(int(2*classes), int(2*classes), kernel_size = 3, padding = 1)

        self.up_conv = NormUpConv2d(int(2*classes), classes)
        self.seg_conv = nn.Conv2d(classes, classes, kernel_size = 1)
        
    def forward(self, x, seg):
        x = self.pre_conv(x)
        x = x + seg

        x = self.conv1(x)
        dx = self.mp_01(x)

        x = self.conv2(dx)
        x = self.conv3(x)

        x = self.up_conv(x)
        x = self.seg_conv(x)

        return x


class Network(nn.Module):
    def __init__(self, classes):
        super(Network, self).__init__()
        self.classes = classes
        
        self.encoder = Encoder()
        self.segmentation = segmentation_block(32, self.classes)
        self.Edge_detection = edge_block(32, self.classes)

    
    def forward(self, x):
        x = self.encoder(x)

        seg_map = self.segmentation(x)
        edge_map = self.Edge_detection(x, seg_map)


        return seg_map, edge_map