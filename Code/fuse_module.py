from typing import Tuple
from typing import Optional
from typeguard import check_argument_types
import math
import torch
import torch.nn as nn
class ResidualBlock(nn.Module):
    def __init__(
            self,
            num_channels: int = 64,
    ):
        check_argument_types()
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(num_channels),
            nn.PReLU(),
            nn.Conv2d(num_channels, num_channels, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(num_channels),
            nn.PReLU(),
        )

    def forward(
            self,
            x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        output = self.conv(x) + x

        return output


class TemporalSelfAttention(nn.Module):
    def __init__(
            self,
            num_channels: int = 64,
    ):
        check_argument_types()
        super().__init__()
        self.conv_q = nn.Sequential(
            nn.Conv2d(num_channels, num_channels // 2, (1, 1), (1, 1)),
            nn.BatchNorm2d(num_channels // 2),
            nn.PReLU(),
        )
        self.conv_k = nn.Sequential(
            nn.Conv2d(num_channels, num_channels // 2, (1, 1), (1, 1)),
            nn.BatchNorm2d(num_channels // 2),
            nn.PReLU(),
        )
        self.conv_v = nn.Sequential(
            nn.Conv2d(num_channels, num_channels // 2, (1, 1), (1, 1)),
            nn.BatchNorm2d(num_channels // 2),
            nn.PReLU(),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(num_channels // 2, num_channels, (1, 1), (1, 1)),
            nn.BatchNorm2d(num_channels),
            nn.PReLU(),
        )

    def forward(
            self,
            x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, C, T, F = x.shape
        q = self.conv_q(x)
        k = self.conv_k(x).transpose(1, 2).contiguous()
        v = self.conv_v(x)
        qk = torch.softmax(torch.matmul(q, k) / math.sqrt(C * F // 2), dim=-1)
        logits = torch.matmul(qk, v).view(B, T, C // 2, F).permute(0, 2, 1, 3).contiguous()
        output = self.conv(logits) + x
        return output


class FrequencySelfAttention(nn.Module):
    def __init__(
            self,
            num_channels: int = 64,
    ):
        check_argument_types()
        super().__init__()
        self.conv_q = nn.Sequential(
            nn.Conv2d(num_channels, num_channels // 2, (1, 1), (1, 1)),
            nn.BatchNorm2d(num_channels // 2),
            nn.PReLU(),
        )
        self.conv_k = nn.Sequential(
            nn.Conv2d(num_channels, num_channels // 2, (1, 1), (1, 1)),
            nn.BatchNorm2d(num_channels // 2),
            nn.PReLU(),
        )
        self.conv_v = nn.Sequential(
            nn.Conv2d(num_channels, num_channels // 2, (1, 1), (1, 1)),
            nn.BatchNorm2d(num_channels // 2),
            nn.PReLU(),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(num_channels // 2, num_channels, (1, 1), (1, 1)),
            nn.BatchNorm2d(num_channels),
            nn.PReLU(),
        )

    def forward(
            self,
            x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, C, T, F = x.shape
        q = self.conv_q(x)
        k = self.conv_k(x).transpose(1, 2).contiguous()
        v = self.conv_v(x)
        qk = torch.softmax(torch.matmul(q, k) / math.sqrt(C * T // 2), dim=-1)
        logits = torch.matmul(qk, v).view(B, F, C // 2, T).permute(0, 2, 3, 1).contiguous()
        output = self.conv(logits) + x
        return output


class RABlock(nn.Module):
    def __init__(
            self,
            num_channels: int = 64,
    ):
        check_argument_types()
        super().__init__()
        self.residual_blocks = nn.Sequential(
            ResidualBlock(num_channels),
            ResidualBlock(num_channels),
        )
        self.temp_self_att = TemporalSelfAttention(num_channels)
        self.freq_self_att = FrequencySelfAttention(num_channels)
        self.conv = nn.Sequential(
            nn.Conv2d(num_channels * 2, num_channels, (1, 1), (1, 1)),
            nn.BatchNorm2d(num_channels),
            nn.PReLU(),
        )

    def forward(
            self,
            x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        f_res = self.residual_blocks(x)
        f_temp = self.temp_self_att(f_res)
        f_freq = self.freq_self_att(f_temp)
        f_comb = torch.cat((f_res, f_freq), dim=1)
        f_ra = self.conv(f_comb)
        return f_ra


class InteractionModule(nn.Module):
    def __init__(
            self,
            num_channels: int = 64,
    ):
        check_argument_types()
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(num_channels * 2, num_channels, (1, 1), (1, 1)),
            nn.Sigmoid(),
        )

    def forward(
            self,
            x1: torch.Tensor,
            x2: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        input = torch.cat((x1, x2), dim=1)
        mask = self.conv(input)
        output = x1 + x2 * mask
        return output