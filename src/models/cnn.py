from torch import nn

class BaselineCNN(nn.Module):
    def __init__(self, base_width: int = 32, depth: int = 3):
        super().__init__()

        stages = []
        input_channels = 3

        for stage in range(depth):
            output_channels = base_width * (2 ** stage)
            stages.extend([
                nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ])
            input_channels = output_channels

        self.features = nn.Sequential(*stages)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(input_channels, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 10),
        )

    def forward(self, images):
        features = self.features(images)
        features = nn.functional.adaptive_avg_pool2d(features, (1, 1))
        features = features.flatten(start_dim=1)
        return self.classifier(features)

