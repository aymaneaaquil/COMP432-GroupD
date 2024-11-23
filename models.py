import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()

        # convolutional layers
        self.convolution_layer_1 = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=stride, padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.convolution_layer_2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
        )

        self.downsample = downsample

        # activation function
        self.relu = nn.ReLU()

        # numebr of output layers
        self.out_channels = out_channels

    def forward(self, x):
        residual = x

        output = self.convolution_layer_1(x)
        output = self.convolution_layer_2(output)

        if self.downsample:
            residual = self.downsample(x)

        output += residual
        output = self.relu(output)

        return output


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.inplanes = 64

        self.convolutional_layer_1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.max_pooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer0 = self._generate_layer(block, 64, layers[0], stride=1)
        self.layer1 = self._generate_layer(block, 128, layers[1], stride=2)
        self.layer2 = self._generate_layer(block, 256, layers[2], stride=2)
        self.layer3 = self._generate_layer(block, 512, layers[3], stride=2)

        self.average_pooling = nn.AvgPool2d(7, stride=1)

        self.fully_connected = nn.Linear(512, num_classes)

    def _generate_layer(self, block, planes, blocks, stride=1):
        downsample = None

        # downsample when the width/height of the output is smaller than the input or the stride is > 1
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.convolutional_layer_1(x)
        x = self.max_pooling(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.average_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.fully_connected(x)

        return x
