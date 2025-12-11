import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.relu = nn.LeakyReLU(0.02)

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.relu(x + residual)
        return x


class MultiScaleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleConv, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels // 4, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels, out_channels // 4, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels, out_channels // 4, kernel_size=5, padding=2)
        self.conv10 = nn.Conv1d(in_channels, out_channels // 4, kernel_size=10, padding=5)

        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.LeakyReLU(0.02)

    def forward(self, x):
        out1 = self.conv1(x)
        out3 = self.conv3(x)
        out5 = self.conv5(x)
        out10 = self.conv10(x)
        out10 = out10[:, :, :-1]

        out = torch.cat([out1, out3, out5, out10], dim=1)
        out = self.relu(self.bn(out))
        return out


class FeatureInteractionBlock(nn.Module):
    def __init__(self, channels):
        super(FeatureInteractionBlock, self).__init__()
        self.channel_interaction = nn.Sequential(
            nn.Conv1d(channels, channels // 4, 1),
            nn.LeakyReLU(0.02),
            nn.Conv1d(channels // 4, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        attention = self.channel_interaction(x)
        return x * attention


class PacketEncoder(nn.Module):
    def __init__(self, input_dim=256, output_dim=256):
        super(PacketEncoder, self).__init__()

        self.input_proj = nn.Sequential(
            nn.Conv1d(input_dim, 256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.02)
        )

        self.multiscale1 = MultiScaleConv(256, 512)
        self.feature_interact1 = FeatureInteractionBlock(512)

        self.multiscale2 = MultiScaleConv(512, 512)
        self.feature_interact2 = FeatureInteractionBlock(512)

        self.res_blocks = nn.ModuleList([
            ResidualBlock(512, 512, 3),
            ResidualBlock(512, 384, 3),
            ResidualBlock(384, 256, 3),
        ])

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)

        self.feature_fusion = nn.Sequential(
            nn.Linear(512, 768),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.2),
            nn.Linear(768, 768),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.15),
            nn.Linear(768, output_dim)
        )

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = x.float()
        x = self.input_proj(x)

        x = self.multiscale1(x)
        x = self.feature_interact1(x)
        x = self.dropout(x)

        x = self.multiscale2(x)
        x = self.feature_interact2(x)
        x = self.dropout(x)

        for res_block in self.res_blocks:
            x = res_block(x)
            x = self.dropout(x)

        avg_pooled = self.global_avg_pool(x).squeeze(-1)
        max_pooled = self.global_max_pool(x).squeeze(-1)

        pooled_features = torch.cat([avg_pooled, max_pooled], dim=1)
        output = self.feature_fusion(pooled_features)

        return output


class PacketDecoder(nn.Module):
    def __init__(self,
                 input_dim: int = 128,
                 num_packets: int = 10,
                 packet_length: int = 256,
                 hidden_dim: int = 1024):
        super().__init__()
        self.input_dim = input_dim
        self.num_packets = num_packets
        self.packet_length = packet_length

        self.decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.02),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.02),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_packets * packet_length)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.decoder(x)
        x = x.view(-1, self.num_packets, self.packet_length)
        return x


if __name__ == "__main__":
    batch_size = 32
    num_packets = 10
    packet_length = 256
    latent_dim = 256

    input_data = torch.randn(batch_size, num_packets, packet_length)
    encoder = PacketEncoder()
    decoder = PacketDecoder(
        input_dim=latent_dim, num_packets=num_packets, packet_length=packet_length
    )

    print("--- BALANCED Autoencoder Forward Pass ---")
    print(f"Input Shape:          {input_data.shape}")
    encoded_data = encoder(input_data)
    print(f"Encoded (Latent) Shape: {encoded_data.shape}")
    decoded_data = decoder(encoded_data)
    print(f"Decoded Shape:        {decoded_data.shape}\n")

    encoder_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    decoder_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print("--- BALANCED Model Size ---")
    print(f"Strengthened Encoder Parameters: {encoder_params:,}")
    print(f"Simplified Decoder Parameters:   {decoder_params:,}")
    print(f"Total Autoencoder Parameters:    {encoder_params + decoder_params:,}")
