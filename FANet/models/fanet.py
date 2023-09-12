# full assembly of the sub-parts to form the complete net

from .fanet_parts import *

class FANet(nn.Module):
    def __init__(self, num_classes):
        super(FANet, self).__init__()
        self.inc = inconv(3, 64)
        self.down1 = down_mhsa(64, 128, size=160)
        self.down2 = down_mhsa(128, 256, size=80)
        self.down3 = down_mhsa(256, 512, size=40)
        self.down4 = down_mhsa(512, 1024, size=20)
        self.up1 = up(1024, 512)
        self.up2 = up(512, 256)
        self.up3 = up(256, 128)
        self.up4 = up(128, 64)
        self.outc = outconv(64, num_classes)
        self.ga = globle_attention()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # x = self.ga(x5)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x
