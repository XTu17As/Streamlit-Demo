import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms

# FCOS configs
FCOS_STRIDES = [8, 16, 32, 64, 128]
NMS_IOU = 0.6

# ---- TinyViT backbone blocks ----
class Conv2d_BN(nn.Module):
    def __init__(self, in_ch, out_ch, k=1, s=1, p=0, g=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class WindowAttention(nn.Module):
    def __init__(self, dim, key_dim, num_heads=4, attn_ratio=4., window_size=7):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = key_dim * num_heads
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.q = nn.Linear(dim, self.nh_kd)
        self.k = nn.Linear(dim, self.nh_kd)
        self.v = nn.Linear(dim, self.dh)
        self.proj = nn.Linear(self.dh, dim)
    def forward(self, x):
        B,H,W,C = x.shape
        q = self.q(x).reshape(B,H*W,self.num_heads,self.key_dim).permute(0,2,1,3)
        k = self.k(x).reshape(B,H*W,self.num_heads,self.key_dim).permute(0,2,1,3)
        v = self.v(x).reshape(B,H*W,self.num_heads,self.dh//self.num_heads).permute(0,2,1,3)
        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).permute(0,2,1,3).reshape(B,H,W,self.dh)
        return self.proj(out)

class TinyViTBlock(nn.Module):
    def __init__(self, dim, num_heads=4, key_dim=16):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, key_dim, num_heads)
        self.conv = Conv2d_BN(dim, dim, 3, 1, 1, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Linear(dim*4, dim))
    def forward(self, x):
        B,C,H,W = x.shape
        y = self.norm1(x.permute(0,2,3,1))
        y = self.attn(y).permute(0,3,1,2)
        x = x + y + self.conv(x)
        y = self.mlp(self.norm2(x.permute(0,2,3,1))).permute(0,3,1,2)
        return x + y

class TinyViTBackbone(nn.Module):
    def __init__(self, in_ch=3, dims=(64,96,128,256)):
        super().__init__()
        self.stem = nn.Sequential(
            Conv2d_BN(in_ch, 32, 3, 2, 1),
            Conv2d_BN(32, dims[0], 3, 2, 1)
        )
        self.stage1 = TinyViTBlock(dims[0])
        self.down12 = Conv2d_BN(dims[0], dims[1], 3, 2, 1)
        self.stage2 = TinyViTBlock(dims[1])
        self.down23 = Conv2d_BN(dims[1], dims[2], 3, 2, 1)
        self.stage3 = TinyViTBlock(dims[2])
        self.down34 = Conv2d_BN(dims[2], dims[3], 3, 2, 1)
        self.stage4 = TinyViTBlock(dims[3])
    def forward(self, x):
        x = self.stem(x)
        c2 = self.stage1(x)
        c3 = self.stage2(self.down12(c2))
        c4 = self.stage3(self.down23(c3))
        c5 = self.stage4(self.down34(c4))
        return [c3,c4,c5]

class FPN(nn.Module):
    def __init__(self, in_channels, out_ch=96):
        super().__init__()
        self.lateral = nn.ModuleList([nn.Conv2d(c,out_ch,1) for c in in_channels])
        self.smooth = nn.ModuleList([nn.Conv2d(out_ch,out_ch,3,padding=1) for _ in in_channels])
        self.p6 = nn.Conv2d(out_ch,out_ch,3,2,1)
        self.p7 = nn.Conv2d(out_ch,out_ch,3,2,1)
    def forward(self, inputs):
        feats = [l(f) for l,f in zip(self.lateral,inputs)]
        for i in range(len(feats)-1,0,-1):
            feats[i-1] += F.interpolate(feats[i], size=feats[i-1].shape[-2:], mode="nearest")
        outs = [s(f) for s,f in zip(self.smooth,feats)]
        p6 = self.p6(outs[-1])
        p7 = self.p7(F.relu(p6))
        return outs + [p6,p7]

class FCOSHead(nn.Module):
    def __init__(self, in_ch=96, num_classes=1):
        super().__init__()
        def tower(): return nn.Sequential(
            *[nn.Sequential(nn.Conv2d(in_ch,in_ch,3,padding=1,bias=False),
                            nn.GroupNorm(32,in_ch),
                            nn.ReLU(True)) for _ in range(3)]
        )
        self.cls_tower, self.reg_tower = tower(), tower()
        self.cls_logits = nn.Conv2d(in_ch, num_classes, 3, padding=1)
        self.bbox_pred = nn.Conv2d(in_ch, 4, 3, padding=1)
        self.centerness = nn.Conv2d(in_ch, 1, 3, padding=1)
    def forward(self, feats):
        cls, reg, ctr = [], [], []
        for f in feats:
            c, r = self.cls_tower(f), self.reg_tower(f)
            cls.append(self.cls_logits(c))
            reg.append(F.relu(self.bbox_pred(r)))
            ctr.append(self.centerness(r))
        return cls, reg, ctr

class TinyViT_FCOS(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.backbone = TinyViTBackbone()
        self.fpn = FPN([96,128,256])
        self.head = FCOSHead(num_classes=num_classes)
    def forward(self, x):
        feats = self.backbone(x)
        feats = self.fpn(feats)
        return self.head(feats)

# ---- postprocessing ----
def generate_points(shapes, strides, device):
    pts=[]
    for (h,w),s in zip(shapes,strides):
        yy,xx=torch.meshgrid(torch.arange(h),torch.arange(w),indexing="ij")
        pts.append(torch.stack([(xx+0.5)*s,(yy+0.5)*s],dim=-1).view(-1,2))
    return pts

def decode_boxes_from_ltrb(points,ltrb):
    x0 = points[:,0]-ltrb[:,0]
    y0 = points[:,1]-ltrb[:,1]
    x1 = points[:,0]+ltrb[:,2]
    y1 = points[:,1]+ltrb[:,3]
    return torch.stack([x0,y0,x1,y1],dim=-1)

def postprocess_predictions(cls_logits, reg_preds, ctrness, image_size, device, score_thr=0.05, max_det=200):
    shapes=[(x.shape[2],x.shape[3]) for x in cls_logits]
    pts=torch.cat(generate_points(shapes,FCOS_STRIDES[:len(shapes)],device),dim=0)
    all_results=[]
    B=cls_logits[0].shape[0]
    for b in range(B):
        scores,boxes=[],[]
        for c,r,ct in zip(cls_logits,reg_preds,ctrness):
            s=torch.sigmoid(c[b])*torch.sigmoid(ct[b])
            s=s.permute(1,2,0).reshape(-1,s.shape[0])
            r=r[b].permute(1,2,0).reshape(-1,4)
            boxes.append(r); scores.append(s)
        scores=torch.cat(scores,dim=0)
        boxes=torch.cat(boxes,dim=0)
        boxes=decode_boxes_from_ltrb(pts,boxes)
        H,W=image_size
        boxes[:,[0,2]]=boxes[:,[0,2]].clamp(0,W-1)
        boxes[:,[1,3]]=boxes[:,[1,3]].clamp(0,H-1)
        results=[]
        for c in range(scores.shape[1]):
            sc=scores[:,c]; mask=sc>score_thr
            if not mask.any(): continue
            bx,sc=sc[mask],boxes[mask]
            keep=nms(bx,sc,NMS_IOU)
            for k in keep[:max_det]:
                results.append((bx[k].cpu().numpy(),float(sc[k]),c+1))
        all_results.append(results)
    return all_results
