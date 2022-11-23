import torch
import torch.nn as nn
from loss_function import numParams
from fuse_module import RABlock, InteractionModule


class dual_two_branch(nn.Module):
    def __init__(self):
        super(dual_two_branch, self).__init__()
        self.en_shang = encoder_mag()
        self.en_xia = encoder()

        self.ra_block_shang_1 = RABlock(64)
        self.ra_block_xia_1 = RABlock(64)
        self.fuse_shang_1 = InteractionModule(64)
        self.fuse_xia_1 = InteractionModule(64)

        self.ra_block_shang_2 = RABlock(64)
        self.ra_block_xia_2 = RABlock(64)
        self.fuse_shang_2 = InteractionModule(64)
        self.fuse_xia_2 = InteractionModule(64)

        self.ra_block_shang_3 = RABlock(64)
        self.ra_block_xia_3 = RABlock(64)
        self.fuse_shang_3 = InteractionModule(64)
        self.fuse_xia_3 = InteractionModule(64)

        self.ra_block_shang_4 = RABlock(64)
        self.ra_block_xia_4 = RABlock(64)
        self.fuse_shang_4 = InteractionModule(64)
        self.fuse_xia_4 = InteractionModule(64)

        self.de_shang = decoder_mag()
        self.de_xia_real = decoder()
        self.de_xia_imag = decoder()

    def forward(self, x):
        x_shang_mag, x_shang_phase = torch.norm(x, dim=1), torch.atan2(x[:, -1, :, :], x[:, 0, :, :])
        x_mag = x_shang_mag.unsqueeze(dim=1)
        x_mag_en = self.en_shang(x_mag)
        x_ri_en = self.en_xia(x)

        x_shang_1 = self.ra_block_shang_1(x_mag_en)
        x_xia_1 = self.ra_block_xia_1(x_ri_en)
        x_shang_fuse1 = self.fuse_shang_1(x_shang_1, x_xia_1)
        x_xia_fuse1 = self.fuse_xia_1(x_xia_1, x_shang_1)

        x_shang_2 = self.ra_block_shang_2(x_shang_fuse1)
        x_xia_2 = self.ra_block_xia_2(x_xia_fuse1)
        x_shang_fuse2 = self.fuse_shang_2(x_shang_2, x_xia_2)
        x_xia_fuse2 = self.fuse_xia_2(x_xia_2, x_shang_2)

        x_shang_3 = self.ra_block_shang_3(x_shang_fuse2)
        x_xia_3 = self.ra_block_xia_3(x_xia_fuse2)
        x_shang_fuse3 = self.fuse_shang_3(x_shang_3, x_xia_3)
        x_xia_fuse3 = self.fuse_xia_3(x_xia_3, x_shang_3)

        x_shang_4 = self.ra_block_shang_4(x_shang_fuse3)
        x_xia_4 = self.ra_block_xia_4(x_xia_fuse3)
        x_shang_out = self.fuse_shang_4(x_shang_4, x_xia_4)
        x_xia_out = self.fuse_xia_4(x_xia_4, x_shang_4)

        x_mag_mask = self.de_shang(x_shang_out)
        x_mag_mask = x_mag_mask.squeeze(dim=1)
        x_mag_out = x_shang_mag * x_mag_mask

        x_real_out = self.de_xia_real(x_xia_out)
        x_imag_out = self.de_xia_imag(x_xia_out)
        x_real_out = x_real_out.squeeze(dim=1)
        x_imag_out = x_imag_out.squeeze(dim=1)

        x_phase = torch.atan2(x_imag_out, x_real_out)
        real_out, imag_out = (x_mag_out * torch.cos(x_phase) + x_real_out), (x_mag_out * torch.sin(x_phase) + x_imag_out)

        com_out = torch.stack((real_out, imag_out), dim=1)
        return com_out


class encoder_mag(nn.Module):
    def __init__(self, width=64):
        super(encoder_mag, self).__init__()
        self.in_channels = 1
        self.out_channels = 1
        self.width = width
        self.inp_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.width, kernel_size=(1, 1))
        self.inp_norm = nn.LayerNorm(161)
        self.inp_prelu = nn.PReLU(self.width)
        self.enc_dense1 = DenseBlock(161, 4, self.width)
        self.enc_conv1 = nn.Conv2d(in_channels=self.width, out_channels=self.width, kernel_size=(1, 3), stride=(1, 2))
        self.enc_norm1 = nn.LayerNorm(80)
        self.enc_prelu1 = nn.PReLU(self.width)

    def forward(self, x):
        out = self.inp_prelu(self.inp_norm(self.inp_conv(x)))
        out = self.enc_dense1(out)
        x = self.enc_prelu1(self.enc_norm1(self.enc_conv1(out)))
        return x


class encoder(nn.Module):
    def __init__(self, width=64):
        super(encoder, self).__init__()
        self.in_channels = 2
        self.out_channels = 1
        self.width = width
        self.inp_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.width, kernel_size=(1, 1))
        self.inp_norm = nn.LayerNorm(161)
        self.inp_prelu = nn.PReLU(self.width)
        self.enc_dense1 = DenseBlock(161, 4, self.width)
        self.enc_conv1 = nn.Conv2d(in_channels=self.width, out_channels=self.width, kernel_size=(1, 3), stride=(1, 2))
        self.enc_norm1 = nn.LayerNorm(80)
        self.enc_prelu1 = nn.PReLU(self.width)

    def forward(self, x):
        out = self.inp_prelu(self.inp_norm(self.inp_conv(x)))
        out = self.enc_dense1(out)
        x = self.enc_prelu1(self.enc_norm1(self.enc_conv1(out)))
        return x


class decoder_mag(nn.Module):
    def __init__(self, width=64):
        super(decoder_mag, self).__init__()
        self.in_channels = 1
        self.out_channels = 1
        self.width = width
        self.dec_dense1 = DenseBlock(80, 4, self.width)
        self.pad = nn.ConstantPad2d((1, 1, 0, 0), value=0.)
        self.dec_conv1 = SPConvTranspose2d(in_channels=self.width, out_channels=self.width, kernel_size=(1, 3), r=2)
        self.pad1 = nn.ConstantPad2d((1, 0, 0, 0), value=0.)
        self.dec_norm1 = nn.LayerNorm(161)
        self.dec_prelu1 = nn.PReLU(self.width)
        self.out_conv = nn.Conv2d(in_channels=self.width, out_channels=self.out_channels, kernel_size=(1, 1))
        self.mask1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1)),
            nn.Sigmoid()
        )
        self.mask2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1)),
            nn.Tanh()
        )
        self.maskconv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1))
        self.maskrelu = nn.Sigmoid()

    def forward(self, x):
        out = self.dec_dense1(x)
        out = self.dec_prelu1(self.dec_norm1(self.pad1(self.dec_conv1(self.pad(out)))))
        out = self.out_conv(out)
        out = self.mask1(out) * self.mask2(out)
        out = self.maskrelu(self.maskconv(out))
        return out


class decoder(nn.Module):
    def __init__(self, width=64):
        super(decoder, self).__init__()
        self.in_channels = 1
        self.out_channels = 1
        self.width = width
        self.dec_dense1 = DenseBlock(80, 4, self.width)
        self.pad = nn.ConstantPad2d((1, 1, 0, 0), value=0.)
        self.dec_conv1 = SPConvTranspose2d(in_channels=self.width, out_channels=self.width, kernel_size=(1, 3), r=2)
        self.pad1 = nn.ConstantPad2d((1, 0, 0, 0), value=0.)
        self.dec_norm1 = nn.LayerNorm(161)
        self.dec_prelu1 = nn.PReLU(self.width)
        self.out_conv = nn.Conv2d(in_channels=self.width, out_channels=self.out_channels, kernel_size=(1, 1))

    def forward(self, x):
        out = self.dec_dense1(x)
        out = self.dec_prelu1(self.dec_norm1(self.pad1(self.dec_conv1(self.pad(out)))))
        out = self.out_conv(out)
        return out


class SPConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, r=1):
        super(SPConvTranspose2d, self).__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels * r, kernel_size=kernel_size, stride=(1, 1))
        self.r = r

    def forward(self, x):
        out = self.conv(x)
        batch_size, nchannels, H, W = out.shape
        out = out.view((batch_size, self.r, nchannels // self.r, H, W))
        out = out.permute(0, 2, 3, 4, 1)
        out = out.contiguous().view((batch_size, nchannels // self.r, H, -1))
        return out


class DenseBlock(nn.Module):
    def __init__(self, input_size, depth=5, in_channels=64):
        super(DenseBlock, self).__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.pad = nn.ConstantPad2d((1, 1, 1, 0), value=0.)
        self.twidth = 2
        self.kernel_size = (self.twidth, 3)
        for i in range(self.depth):
            dil = 2 ** i
            pad_length = self.twidth + (dil - 1) * (self.twidth - 1) - 1
            setattr(self, 'pad{}'.format(i + 1), nn.ConstantPad2d((1, 1, pad_length, 0), value=0.))
            setattr(self, 'conv{}'.format(i + 1),
                    nn.Conv2d(self.in_channels * (i + 1), self.in_channels, kernel_size=self.kernel_size, dilation=(dil, 1)))
            setattr(self, 'norm{}'.format(i + 1), nn.LayerNorm(input_size))
            setattr(self, 'prelu{}'.format(i + 1), nn.PReLU(self.in_channels))

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            out = getattr(self, 'pad{}'.format(i + 1))(skip)
            out = getattr(self, 'conv{}'.format(i + 1))(out)
            out = getattr(self, 'norm{}'.format(i + 1))(out)
            out = getattr(self, 'prelu{}'.format(i + 1))(out)
            skip = torch.cat([out, skip], dim=1)
        return out


if __name__ == '__main__':
    model = dual_two_branch()
    model.eval()
    x = torch.FloatTensor(1, 2, 288, 161)
    x = model(x)
