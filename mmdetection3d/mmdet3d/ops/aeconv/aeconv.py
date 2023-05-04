import torch
import torch.nn as nn
from torchvision.ops import deform_conv2d
from mmcv.cnn import CONV_LAYERS


@CONV_LAYERS.register_module()
class AeConv(nn.Conv2d):
    def __init__(self, *args, input_size=None, **kwargs):
        super(AeConv, self).__init__(*args, **kwargs)
        if input_size is None:
            self.ae_offset = None
        else:
            self.ae_offset = self.get_offset(input_size)

    def get_offset(self, size):
        # compute the rotation matrix of AeConv
        h, w = size
        out_h, out_w = h // self.stride[0], w // self.stride[1]
        cart_x = torch.arange(out_w) - out_w / 2.0 + 0.5
        cart_y = -(torch.arange(out_h) - out_w / 2.0 + 0.5)
        cart_x = cart_x.view(1, len(cart_x)).repeat(len(cart_y), 1)
        cart_y = cart_y.view(len(cart_y), 1).repeat(1, len(cart_x))
        azimuth = torch.atan2(cart_x, cart_y)
        rot_matrix = torch.stack([torch.cos(azimuth), torch.sin(azimuth), -torch.sin(azimuth), torch.cos(azimuth)], -1)
        rot_matrix = rot_matrix.view(-1, 2, 2)

        # sampling grid of type convolution
        kh, kw = self.weight.shape[-2:]
        kernel_num = kh * kw
        grid_x = torch.arange(-((kw - 1) // 2), kw // 2 + 1)
        grid_y = torch.arange(-((kh - 1) // 2), kh // 2 + 1)
        grid_x = grid_x.view(1, kw).repeat(kh, 1)
        grid_y = grid_y.view(kh, 1).repeat(1, kw)
        conv_offset = torch.stack([grid_y, grid_x]).permute(1, 2, 0).contiguous().view(-1)

        # compute the offset of AeConv
        conv_offset = conv_offset.view(1, kernel_num, 2).repeat(len(rot_matrix), 1, 1).type(rot_matrix.type())
        ae_offset = torch.bmm(rot_matrix, conv_offset.transpose(1, 2)).transpose(1, 2) - conv_offset

        # align the sampled grid with the feature
        shift_h = (h - self.weight.shape[2]) % self.stride[0]
        shift_w = (w - self.weight.shape[3]) % self.stride[1]
        ae_offset[:, :, 0] += shift_w / 2.0
        ae_offset[:, :, 1] += shift_h / 2.0

        # reshape the offset of AeConv
        ae_offset = ae_offset.contiguous().view(1, azimuth.shape[0], azimuth.shape[1], 2 * kernel_num).permute(0, 3, 1, 2)
        return ae_offset

    def forward(self, input):
        if self.ae_offset is None:
            self.ae_offset = self.get_offset(input.shape[2:])
        ae_offset = self.ae_offset.to(input.device).repeat(len(input), 1, 1, 1)
        out = deform_conv2d(input, ae_offset, self.weight, bias=self.bias, stride=self.stride, padding=self.padding)
        return out
