import torch
import torch.nn as nn

class CNN3(nn.Module):#3 layers cnn
    def __init__(self, num_classes):
        super(CNN3, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # Conv1
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 224 → 112

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Conv2
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 112 → 56

            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Conv3
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 56 → 28
        )

        # calculate the size of the flattened layer
        self._to_linear = None
        self._get_flatten_size()

        self.classifier = nn.Sequential(
            nn.Linear(self._to_linear, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def _get_flatten_size(self):
        with torch.no_grad():
            x = torch.zeros(1, 3, 224, 224)
            x = self.features(x)
            self._to_linear = x.view(1, -1).shape[1]

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x

class CNN4(nn.Module):  # 4 layers CNN
    def __init__(self, num_classes):
        super(CNN4, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # Conv1
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 224 → 112

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Conv2
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 112 → 56

            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Conv3
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 56 → 28

            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # Conv4
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)   # 28 → 14
        )

        self._to_linear = None
        self._get_flatten_size()

        self.classifier = nn.Sequential(
            nn.Linear(self._to_linear, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def _get_flatten_size(self):
        with torch.no_grad():
            x = torch.zeros(1, 3, 224, 224)
            x = self.features(x)
            self._to_linear = x.view(1, -1).shape[1]

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class CNN6(nn.Module):
    def __init__(self, num_classes):
        super(CNN6, self).__init__()

        def conv_block(in_channels, out_channels, dropout=0.0):
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ]
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            conv_block(3, 32, dropout=0.1),
            conv_block(32, 32),
            nn.MaxPool2d(2),  # 224 → 112

            conv_block(32, 64, dropout=0.1),
            conv_block(64, 64),
            nn.MaxPool2d(2),  # 112 → 56

            conv_block(64, 128, dropout=0.1),
            conv_block(128, 128),
            nn.MaxPool2d(2)   # 56 → 28
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # → [batch, 128, 1, 1]

        self.classifier = nn.Sequential(
            nn.Flatten(),            # → [batch, 128]
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class CNN8(nn.Module):
    def __init__(self, num_classes):
        super(CNN8, self).__init__()

        def conv_block(in_channels, out_channels, dropout=0.0):
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ]
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            conv_block(3, 32, dropout=0.1),
            conv_block(32, 32),
            nn.MaxPool2d(2),  # 224 → 112

            conv_block(32, 64, dropout=0.1),
            conv_block(64, 64),
            nn.MaxPool2d(2),  # 112 → 56

            conv_block(64, 128, dropout=0.1),
            conv_block(128, 128),
            nn.MaxPool2d(2),  # 56 → 28

            conv_block(128, 256, dropout=0.1),
            conv_block(256, 256),
            nn.MaxPool2d(2)   # 28 → 14
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # → [batch, 256, 1, 1]

        self.classifier = nn.Sequential(
            nn.Flatten(),            # → [batch, 256]
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
