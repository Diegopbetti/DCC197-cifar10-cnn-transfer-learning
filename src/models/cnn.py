from torch import nn

class BaselineCNN(nn.Module):
    def __init__(
        self,
        base_width: int = 32,
        depth: int = 3,
        dropout: float = 0.5,
        use_batch_norm: bool = True,
    ):
        super().__init__()

        if depth < 1 or depth > 5:
            raise ValueError("depth deve estar entre 1 e 5.")

        if base_width < 1:
            raise ValueError("base_width deve ser positivo.")

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout deve estar entre 0 e 1.")

        stages = []
        input_channels = 3

        for stage in range(depth):
            output_channels = base_width * (2 ** stage)
            for conv_input_channels in (input_channels, output_channels):
                stages.append(
                    nn.Conv2d(
                        conv_input_channels,
                        output_channels,
                        3,
                        padding=1,
                        bias=not use_batch_norm,
                    )
                )
                if use_batch_norm:
                    stages.append(nn.BatchNorm2d(output_channels))
                stages.append(nn.ReLU(inplace=True))
            stages.append(nn.MaxPool2d(2))
            input_channels = output_channels

        self.features = nn.Sequential(*stages)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_channels, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 10),
        )

    def forward(self, images):
        features = self.features(images)
        features = nn.functional.adaptive_avg_pool2d(features, (1, 1))
        features = features.flatten(start_dim=1)
        return self.classifier(features)

