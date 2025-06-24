import torch
import torch.nn as nn
from torchvision.transforms.functional import center_crop

class OriginalUNet(nn.Module):
    def __init__(self):
        super().__init__()
        num_classes = 1

        self.pool = nn.MaxPool2d(2)

        self.pad14 = nn.ZeroPad2d(14)
        
        # Encoder
        # self.enc1_1 = nn.Sequential(nn.Conv2d(3, 64, 3), nn.ReLU())
        # self.enc1_2 = nn.Sequential(nn.Conv2d(64, 64, 3), nn.ReLU())
        
        # self.enc2_1 = nn.Sequential(nn.Conv2d(64, 128, 3), nn.ReLU())
        self.enc2_1 = nn.Sequential(nn.Conv2d(3, 128, 3), nn.ReLU())
        self.enc2_2 = nn.Sequential(nn.Conv2d(128, 128, 3), nn.ReLU())
        self.enc2_3 = nn.Sequential(nn.Conv2d(128, 128, 3), nn.ReLU())

        self.enc3_1 = nn.Sequential(nn.Conv2d(128, 256, 3), nn.ReLU())
        self.enc3_2 = nn.Sequential(nn.Conv2d(256, 256, 3), nn.ReLU())

        self.enc4_1 = nn.Sequential(nn.Conv2d(256, 512, 3), nn.ReLU())
        self.enc4_2 = nn.Sequential(nn.Conv2d(512, 512, 3), nn.ReLU())

        # Bottleneck
        self.bottleneck1 = nn.Sequential(nn.Conv2d(512, 1024, 3), nn.ReLU())
        self.bottleneck2 = nn.Sequential(nn.Conv2d(1024, 1024, 3), nn.ReLU())

        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4_1 = nn.Sequential(nn.Conv2d(1024, 512, 3), nn.ReLU())
        self.dec4_2 = nn.Sequential(nn.Conv2d(512, 512, 3), nn.ReLU())

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3_1 = nn.Sequential(nn.Conv2d(512, 256, 3), nn.ReLU())
        self.dec3_2 = nn.Sequential(nn.Conv2d(256, 256, 3), nn.ReLU())

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2_1 = nn.Sequential(nn.Conv2d(256, 128, 3), nn.ReLU())
        self.dec2_2 = nn.Sequential(nn.Conv2d(128, 128, 3), nn.ReLU())

        # Final
        self.final = nn.Conv2d(128, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        # e1_1 = self.enc1_1(x)
        # e1_2 = self.enc1_2(e1_1)
        # p1 = self.pool(e1_2)  # Skipping first encoder layer due to smaller input size

        # e2_1 = self.enc2_1(p1)
        x = self.pad14(x)
        e2_1 = self.enc2_1(x)
        e2_2 = self.enc2_2(e2_1)
        # e2_3 = self.enc2_3(e2_2)  # In order to get 120x120 instead of 122x122 which would result in 61x61 after 2x2 max pool
        # p2 = self.pool(e2_3)
        p2 = self.pool(e2_2)

        e3_1 = self.enc3_1(p2)
        e3_2 = self.enc3_2(e3_1)
        p3 = self.pool(e3_2)

        e4_1 = self.enc4_1(p3)
        e4_2 = self.enc4_2(e4_1)
        p4 = self.pool(e4_2)

        # Bottleneck
        b = self.bottleneck1(p4)
        b = self.bottleneck2(b)

        # Decoder
        u4 = self.up4(b)
        e4_crop = center_crop(e4_2, [u4.size(2), u4.size(3)])
        d4 = torch.cat([e4_crop, u4], dim=1)
        d4 = self.dec4_1(d4)
        d4 = self.dec4_2(d4)

        u3 = self.up3(d4)
        e3_crop = center_crop(e3_2, [u3.size(2), u3.size(3)])
        d3 = torch.cat([e3_crop, u3], dim=1)
        d3 = self.dec3_1(d3)
        d3 = self.dec3_2(d3)

        u2 = self.up2(d3)
        e2_crop = center_crop(e2_2, [u2.size(2), u2.size(3)])
        d2 = torch.cat([e2_crop, u2], dim=1)
        d2 = self.dec2_1(d2)
        d2 = self.dec2_2(d2)

        out_center = self.final(d2)

        # Embed into full 256x256 output
        B, C, Hc, Wc = out_center.shape
        H, W = 256, 256
        out_full = out_center.new_zeros((B, C, H, W))
        top = (H - Hc) // 2
        left = (W - Wc) // 2
        out_full[:, :, top:top + Hc, left:left + Wc] = out_center
        return out_full