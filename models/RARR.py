import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules import *

class Model(nn.Module):
    def __init__(self, in_c=3, out_c=3, n_feat=40, scale_unetfeats=20, num_cab=8, kernel_size=3, reduction=4, bias=False):
        super(Model, self).__init__()

        act = nn.PReLU()
        self.shallow_feat1 = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias), CAB(n_feat, kernel_size, reduction, bias=bias, act=act))
        self.shallow_feat2 = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias), CAB(n_feat, kernel_size, reduction, bias=bias, act=act))
        self.shallow_feat3 = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias), CAB(n_feat, kernel_size, reduction, bias=bias, act=act))

        # Cross Stage Feature Fusion (CSFF)
        self.stage1_encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=False)
        self.stage1_decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)

        self.stage2_encoder_1 = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=True)
        self.stage2_decoder_1 = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)
        self.stage2_encoder_2 = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=True)
        self.stage2_decoder_2 = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)

        # self.stage3_orsnet = ORSNet(n_feat, scale_orsnetfeats, kernel_size, reduction, act, bias, scale_unetfeats, num_cab)
        self.stage3_encoder_1 = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=True)
        self.stage3_decoder_1 = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)
        self.stage3_encoder_2 = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=True)
        self.stage3_decoder_2 = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)
        self.stage3_encoder_3 = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=True)
        self.stage3_decoder_3 = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)

        self.CGAM12 = CGAM(n_feat, kernel_size=1, bias=bias)
        self.CGAM23 = CGAM(n_feat, kernel_size=1, bias=bias)

        self.concat12 = conv(n_feat * 2, n_feat, kernel_size, bias=bias)
        # self.concat23 = conv(n_feat * 2, n_feat + scale_orsnetfeats, kernel_size, bias=bias)
        self.concat23 = conv(n_feat * 2, n_feat, kernel_size, bias=bias)
        # self.tail = conv(n_feat + scale_orsnetfeats, out_c, kernel_size, bias=bias)
        self.tail = conv(n_feat, out_c, kernel_size, bias=bias)

    def forward(self, x3_img):
        # Stage 1: Use 1/4 resolution image (downsampled)
        x1_img = F.interpolate(x3_img, scale_factor=0.25, mode='bilinear', align_corners=False)
        x1 = self.shallow_feat1(x1_img)
        feat1 = self.stage1_encoder(x1)
        res1 = self.stage1_decoder(feat1)

        # Apply CGAM at Stage 1
        # x2_CGAMfeats, stage1_img = self.CGAM12(res1[0], F.interpolate(x3_img, scale_factor=0.5, mode='bilinear', align_corners=False))
        x2_CGAMfeats, stage1_img = self.CGAM12(res1[0], x1_img)

        feat1 = [F.interpolate(f, scale_factor=2, mode='bilinear', align_corners=False) for f in feat1]
        res1 = [F.interpolate(f, scale_factor=2, mode='bilinear', align_corners=False) for f in res1]
        x2_CGAMfeats = F.interpolate(x2_CGAMfeats, scale_factor=2, mode='bilinear', align_corners=False)

        # Stage 2: Use 1/2 resolution image (downsampled)
        x2_img = F.interpolate(x3_img, scale_factor=0.5, mode='bilinear', align_corners=False)
        x2 = self.shallow_feat2(x2_img)
        
        # Concatenate CGAM features with shallow features of Stage 2
        x2_cat = self.concat12(torch.cat([x2, x2_CGAMfeats], 1))

        # Weighted U-Net, Unequal U-Net
        feat2_1 = self.stage2_encoder_1(x2_cat, feat1, res1)
        res2 = self.stage2_decoder_1(feat2_1)
        feat2_2 = self.stage2_encoder_2(res2[0], feat1, res1)
        res2 = self.stage2_decoder_2(feat2_2)

        # Apply CGAM at Stage 2
        # x3_CGAMfeats, stage2_img = self.CGAM23(res2[0], x3_img)
        x3_CGAMfeats, stage2_img = self.CGAM23(res2[0], x2_img)


        feat2 = [F.interpolate(f, scale_factor=2, mode='bilinear', align_corners=False) for f in feat2_2]
        res2 = [F.interpolate(f, scale_factor=2, mode='bilinear', align_corners=False) for f in res2]
        x3_CGAMfeats = F.interpolate(x3_CGAMfeats, scale_factor=2, mode='bilinear', align_corners=False)

        # Stage 3: Full resolution image
        x3 = self.shallow_feat3(x3_img)

        # Concatenate CGAM features with shallow features of Stage 3
        x3_cat = self.concat23(torch.cat([x3, x3_CGAMfeats], 1))
        # x3_cat = self.stage3_orsnet(x3_cat, feat2, res2)

        # Weighted U-Net, Unequal U-Net
        feat3_1 = self.stage3_encoder_1(x3_cat, feat2, res2)
        res3 = self.stage3_decoder_1(feat3_1)
        feat3_2 = self.stage3_encoder_2(res3[0], feat2, res2)
        res3 = self.stage3_decoder_2(feat3_2)
        feat3_3 = self.stage3_encoder_3(res3[0], feat2, res2)
        res3 = self.stage3_decoder_3(feat3_3)
    
        stage3_img = self.tail(res3[0])

        return [stage3_img + x3_img, stage2_img, stage1_img]
    
