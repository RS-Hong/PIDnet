import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.pidnet_training import weights_init

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class DAPPM(nn.Module):
    def __init__(self, inplanes, branch_planes, outplanes):
        super(DAPPM, self).__init__()
        self.scale0 = nn.Sequential(
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False)
        )
        self.scale1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False)
        )
        self.scale2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False)
        )
        self.scale3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False)
        )
        self.scale4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False)
        )
        self.process0 = nn.Sequential(
            nn.BatchNorm2d(branch_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False)
        )
        self.process1 = nn.Sequential(
            nn.BatchNorm2d(branch_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False)
        )
        self.process2 = nn.Sequential(
            nn.BatchNorm2d(branch_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False)
        )
        self.process3 = nn.Sequential(
            nn.BatchNorm2d(branch_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False)
        )
        self.process4 = nn.Sequential(
            nn.BatchNorm2d(branch_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False)
        )
        self.compression = nn.Sequential(
            nn.BatchNorm2d(branch_planes * 5),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes * 5, outplanes, kernel_size=1, bias=False)
        )
        self.shortcut = nn.Sequential(
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False)
        )

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        s0 = self.process0(self.scale0(x))
        s1 = self.process1(self.scale1(x))
        s2 = self.process2(self.scale2(x))
        s3 = self.process3(self.scale3(x))
        s4 = self.process4(self.scale4(x))
        s4 = F.interpolate(s4, size=(h, w), mode='bilinear', align_corners=True)
        cat = torch.cat([s0, s1, s2, s3, s4], dim=1)
        out = self.compression(cat) + self.shortcut(x)
        return out

class SegmentHead(nn.Module):
    def __init__(self, inplanes, interplanes, num_classes):
        super(SegmentHead, self).__init__()
        self.block1 = nn.Sequential(
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, interplanes, kernel_size=3, padding=1, bias=False)
        )
        self.block2 = nn.Sequential(
            nn.BatchNorm2d(interplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(interplanes, interplanes, kernel_size=3, padding=1, bias=False)
        )
        self.block3 = nn.Sequential(
            nn.BatchNorm2d(interplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(interplanes, interplanes, kernel_size=3, padding=1, bias=False)
        )
        self.final = nn.Conv2d(interplanes, num_classes, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.final(x)

class BoundaryHead(nn.Module):
    def __init__(self, inplanes, interplanes):
        super(BoundaryHead, self).__init__()
        self.block1 = nn.Sequential(
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, interplanes, kernel_size=3, padding=1, bias=False)
        )
        self.block2 = nn.Sequential(
            nn.BatchNorm2d(interplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(interplanes, interplanes, kernel_size=3, padding=2, dilation=2, bias=False)
        )
        self.block3 = nn.Sequential(
            nn.BatchNorm2d(interplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(interplanes, interplanes, kernel_size=3, padding=4, dilation=4, bias=False)
        )
        self.final = nn.Conv2d(interplanes, 1, kernel_size=1, bias=True)  # 输出单通道

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.final(x)

class PIDNet(nn.Module):
    def __init__(self, num_classes=2, planes=32, spp_branch_planes=96, head_planes=128, aux=True):
        super(PIDNet, self).__init__()
        self.aux = aux
        self.num_classes = num_classes
        self.align_corners = True

        self.stem = nn.Sequential(
            nn.Conv2d(3, planes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
        )

        self.layer1 = self._make_layer_res(Bottleneck, planes, planes, 2, stride=1)
        self.layer2 = self._make_layer_res(Bottleneck, planes*4, 64, 3, stride=2)
        self.layer3_p = self._make_layer_res(BasicBlock, 256, 64, 3, stride=1)
        self.layer3_i = self._make_layer_res(BasicBlock, 256, 128, 3, stride=2)
        self.layer4_i = self._make_layer_res(BasicBlock, 128, 256, 1, stride=2)

        self.compr = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=1, bias=False)
        )
        self.spp = DAPPM(128, spp_branch_planes // 5, 128)

        self.lap_p = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        )
        self.lap_d = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=1, bias=False)
        )
        self.fusion_block = nn.Sequential(
            nn.BatchNorm2d(128 + 128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128 + 128, 128, kernel_size=3, padding=1, bias=False)
        )

        self.seghead_final = SegmentHead(128, head_planes, num_classes)
        self.seghead_p = SegmentHead(64, head_planes, num_classes)
        self.boundary_head = BoundaryHead(64, head_planes // 2)

        weights_init(self, init_type='kaiming_normal')

    def _make_layer_res(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(inplanes, planes, stride, downsample)]
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]

        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)

        x_p = self.layer3_p(x)
        x_i = self.layer3_i(x)
        x_d = self.layer4_i(x_i)

        x_d = self.compr(x_d)
        x_spp = self.spp(x_d)

        x_spp_up = F.interpolate(x_spp, size=x_p.shape[2:], mode='bilinear', align_corners=self.align_corners)
        x_spp_up = self.lap_d(x_spp_up)
        x_p_lap = self.lap_p(x_p)
        fusion = self.fusion_block(torch.cat([x_p_lap, x_spp_up], dim=1))

        main_out = self.seghead_final(fusion)
        main_out = F.interpolate(main_out, size=(h, w), mode='bilinear', align_corners=self.align_corners)

        aux_out = self.seghead_p(x_p)
        aux_out = F.interpolate(aux_out, size=(h, w), mode='bilinear', align_corners=self.align_corners)

        edge_out = self.boundary_head(x_p)
        edge_out = F.interpolate(edge_out, size=(h, w), mode='bilinear', align_corners=self.align_corners)

        if self.aux:
            return main_out, aux_out, edge_out
        return main_out