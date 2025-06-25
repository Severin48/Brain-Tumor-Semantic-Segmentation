import torch
import torch.nn as nn

class ImprovedUNet(nn.Module):
    """
        num_layers: Number of layers in the encoder and decoder layers each
        num_filters: Number of filters for the first conv layer
    """
    def __init__(self, num_layers=4, num_filters=32):
        super().__init__()
        self.num_layers = num_layers
        self.num_filters = num_filters
        in_channels = 3 
        num_classes = 1  
        
        # Encoder layers
        self.encoders = nn.ModuleList()
        for i in range(num_layers):
            input_ch = in_channels if i == 0 else num_filters * (2**(i-1))
            output_ch = num_filters * (2**i)    # double the number of filters at each layer (starts with 3 -> 32, 64, etc.) (if num_filters=32)
            self.encoders.append(nn.Sequential(
                nn.Conv2d(input_ch, output_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(output_ch),
                nn.ReLU(),
                nn.Conv2d(output_ch, output_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(output_ch),
                nn.ReLU()
            ))
        
        self.pools = nn.ModuleList([nn.MaxPool2d(kernel_size=2, stride=2) for _ in range(num_layers)])

        # Bottleneck
        bottleneck_in_ch = num_filters * (2**(num_layers-1))
        bottleneck_out_ch = num_filters * (2**num_layers)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(bottleneck_in_ch, bottleneck_out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(bottleneck_out_ch),
            nn.ReLU(),
            nn.Conv2d(bottleneck_out_ch, bottleneck_out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(bottleneck_out_ch),
            nn.ReLU()
        )

        # Decoder layers
        self.decoders = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        for i in range(num_layers):
            up_in_ch = num_filters * (2**(num_layers - i))
            up_out_ch = num_filters * (2**(num_layers - i - 1))   # halve the number of filters at each layer
            
            self.up_samples.append(
                nn.ConvTranspose2d(up_in_ch, up_out_ch, kernel_size=2, stride=2)
            )
            
            dec_in_ch = up_out_ch * 2 # After concatenating with skip connection
            dec_out_ch = up_out_ch
            
            self.decoders.append(nn.Sequential(
                nn.Conv2d(dec_in_ch, dec_out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(dec_out_ch),
                nn.ReLU(),
                nn.Conv2d(dec_out_ch, dec_out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(dec_out_ch),
                nn.ReLU()
            ))

        # Final layer
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        skip_conns = []
        for i in range(self.num_layers):
            x = self.encoders[i](x)
            skip_conns.append(x)
            x = self.pools[i](x)    # Skip connections are stored before pooling -> cant put pooling in sequential layers

        # Bottleneck
        x = self.bottleneck(x)  # = Last Encoder layer (without pooling)
        
        # Reverse skip connections for easier access
        skip_conns = skip_conns[::-1]

        # Decoder
        for i in range(self.num_layers):
            x = self.up_samples[i](x)
            skip = skip_conns[i]

            concat_skip = torch.cat((x, skip), dim=1)   # Concat skip from encoder with upsampled output
            x = self.decoders[i](concat_skip)   # Pass concat through decoder layer

        # Final
        return self.final(x)