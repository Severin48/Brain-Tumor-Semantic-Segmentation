import torch
import torch.nn as nn

class BaselineUNet(nn.Module):
    def __init__(self, num_layers=2, num_filters=16):
        super().__init__()
        self.num_layers = num_layers
        self.num_filters = num_filters
        num_classes = 1

        # Encoder layers
        self.encoders = nn.ModuleList()
        for i in range(num_layers):
            in_channels = 3 if i == 0 else num_filters * (2**(i-1))
            out_channels = num_filters * (2**i)
            self.encoders.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.ReLU()
            ))
        self.pools = nn.ModuleList([nn.MaxPool2d(2) for _ in range(num_layers)])

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(num_filters * (2**(num_layers-1)), num_filters * (2**num_layers), 3, padding=1),
            nn.ReLU()
        )

        # Decoder layers
        self.decoders = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        for i in range(num_layers):
            in_channels = num_filters * (2**(num_layers-i)) if i == 0 else num_filters * (2**(num_layers-i-1)) * 2
            out_channels = num_filters * (2**(num_layers-i-1))
            self.up_samples.append(nn.ConvTranspose2d(num_filters * (2**(num_layers-i)), num_filters * (2**(num_layers-i-1)), 2, stride=2))
            self.decoders.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.ReLU()
            ))

        # Final layer
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc_out = []
        pool_out = []
        for i in range(self.num_layers):
            enc_out_i = self.encoders[i](x if i == 0 else pool_out[i-1])
            enc_out.append(enc_out_i)
            pool_out_i = self.pools[i](enc_out_i)
            pool_out.append(pool_out_i)

        # Bottleneck
        bottleneck = self.bottleneck(pool_out[-1])

        # Decoder
        dec = bottleneck
        for i in range(self.num_layers):
            upsample = self.up_samples[i](dec)
            dec = self.decoders[i](torch.cat([upsample, enc_out[self.num_layers-i-1]], dim=1))

        # Final
        return self.final(dec)