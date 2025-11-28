# dihe_pytorch.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import List, Iterable, Optional, Sequence

# -------------------------
# Utility: MAC pooling + L2 norm
# -------------------------
def mac_pool(feature_map: torch.Tensor) -> torch.Tensor:
    """
    feature_map: (B, C, H, W)
    returns (B, C) max-pooled per-channel
    """
    return torch.amax(feature_map, dim=(-2, -1))  # shape (B, C)

def l2_normalize(x: torch.Tensor, eps=1e-10):
    return x / (x.norm(p=2, dim=1, keepdim=True).clamp(min=eps))

# -------------------------
# Encoder: VGG16 backbone -> MAC from conv4 and conv5 -> concat -> L2 normalize
# -------------------------
class VGG16_MAC_Encoder(nn.Module):
    def __init__(self,
                 pretrained: bool = False,
                 conv4_idx: int = 22,
                 conv5_idx: int = 29,
                 device: Optional[torch.device] = None):
        """
        conv4_idx, conv5_idx: indices of layers in vgg.features at which to capture activations
          (tune if torchvision's version differs).
        """
        super().__init__()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        vgg = models.vgg16(pretrained=pretrained)
        self.features = vgg.features.to(self.device)
        self.conv4_idx = conv4_idx
        self.conv5_idx = conv5_idx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,3,H,W)
        returns: embeddings (B, D) L2-normalized, where D = C_conv4 + C_conv5
        """
        # pass through vgg.features and capture activations
        act_conv4 = None
        act_conv5 = None
        cur = x
        for i, layer in enumerate(self.features):
            cur = layer(cur)
            if i == self.conv4_idx:
                act_conv4 = cur
            if i == self.conv5_idx:
                act_conv5 = cur
            # small optimization: break if both captured
            if (act_conv4 is not None) and (act_conv5 is not None):
                break

        if act_conv4 is None or act_conv5 is None:
            raise RuntimeError(
                f"Failed to capture conv4/conv5 activations at indices {self.conv4_idx},{self.conv5_idx}. "
                "Adjust the indices to match your torchvision version.")

        mac4 = mac_pool(act_conv4)  # (B, C4)
        mac5 = mac_pool(act_conv5)  # (B, C5)
        concat = torch.cat([mac4, mac5], dim=1)  # (B, C4+C5)
        normed = l2_normalize(concat)
        return normed

# -------------------------
# Generator: U-Net style (simple, configurable)
# -------------------------
class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_batchnorm=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=not use_batchnorm)]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class UNetUpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_dropout=False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        ]
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, ngf=64):
        super().__init__()
        # Encoder (downsampling)
        self.enc1 = UNetBlock(in_channels, ngf, use_batchnorm=False)   # 128
        self.enc2 = UNetBlock(ngf, ngf * 2)  # 64
        self.enc3 = UNetBlock(ngf * 2, ngf * 4)  # 32
        self.enc4 = UNetBlock(ngf * 4, ngf * 8)  # 16
        self.enc5 = UNetBlock(ngf * 8, ngf * 8)  # 8
        self.enc6 = UNetBlock(ngf * 8, ngf * 8)  # 4
        self.enc7 = UNetBlock(ngf * 8, ngf * 8)  # 2
        self.enc8 = UNetBlock(ngf * 8, ngf * 8)  # 1

        # Decoder (upsampling)
        self.up1 = UNetUpBlock(ngf * 8, ngf * 8, use_dropout=True)
        self.up2 = UNetUpBlock(ngf * 8 * 2, ngf * 8, use_dropout=True)
        self.up3 = UNetUpBlock(ngf * 8 * 2, ngf * 8, use_dropout=True)
        self.up4 = UNetUpBlock(ngf * 8 * 2, ngf * 8)
        self.up5 = UNetUpBlock(ngf * 8 * 2, ngf * 4)
        self.up6 = UNetUpBlock(ngf * 4 * 2, ngf * 2)
        self.up7 = UNetUpBlock(ngf * 2 * 2, ngf)
        self.up8 = nn.ConvTranspose2d(ngf * 2, out_channels, kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # Encoder forward and store features for skip-connections
        e1 = self.enc1(x)  # down1
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)
        e8 = self.enc8(e7)

        # Decoder with concatenation of skip connections
        d1 = self.up1(e8)                         # connect with e7
        d1 = torch.cat([d1, e7], dim=1)
        d2 = self.up2(d1)
        d2 = torch.cat([d2, e6], dim=1)
        d3 = self.up3(d2)
        d3 = torch.cat([d3, e5], dim=1)
        d4 = self.up4(d3)
        d4 = torch.cat([d4, e4], dim=1)
        d5 = self.up5(d4)
        d5 = torch.cat([d5, e3], dim=1)
        d6 = self.up6(d5)
        d6 = torch.cat([d6, e2], dim=1)
        d7 = self.up7(d6)
        d7 = torch.cat([d7, e1], dim=1)
        out = self.up8(d7)
        return self.tanh(out)  # output in [-1, 1]

# -------------------------
# Discriminator: PatchGAN (final logits map)
# -------------------------
class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=3, ndf=64):
        super().__init__()
        # following common PatchGAN pattern: no final sigmoid, produce patch logits
        self.net = nn.Sequential(
            # input: in_channels x 256 x 256 -> 128x128
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # final conv -> 1 channel logits
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1)  # output shape B x 1 x Hpatch x Wpatch
        )

    def forward(self, x):
        return self.net(x)

# -------------------------
# ZNCC (Zero-mean Normalized Cross Correlation) - differentiable
# -------------------------
def zncc_per_image(x: torch.Tensor, y: torch.Tensor, eps=1e-8) -> torch.Tensor:
    """
    Compute ZNCC for two images x,y with same shape (B, C, H, W).
    Returns a vector of shape (B,) with ZNCC per sample averaged across elements/channels.
    """
    B = x.shape[0]
    # Flatten each sample into vector
    x_flat = x.view(B, -1)
    y_flat = y.view(B, -1)
    x_mean = x_flat.mean(dim=1, keepdim=True)
    y_mean = y_flat.mean(dim=1, keepdim=True)
    xz = x_flat - x_mean
    yz = y_flat - y_mean
    num = (xz * yz).sum(dim=1)
    denom = torch.sqrt((xz ** 2).sum(dim=1) * (yz ** 2).sum(dim=1)).clamp(min=eps)
    zncc = num / denom
    # clip for numeric safety
    return zncc.clamp(-1.0, 1.0)

# -------------------------
# Hierarchical Triplet Loss
# -------------------------
class HierarchicalTripletLoss(nn.Module):
    def __init__(self, alpha_min: float = 0.05, alpha_max: float = 0.5):
        super().__init__()
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

    def forward(self,
                emb_anchor: torch.Tensor,
                emb_pos: torch.Tensor,
                emb_neg: torch.Tensor,
                H_anchor: Sequence[Iterable],
                H_neg: Sequence[Iterable]) -> torch.Tensor:
        """
        emb_* : (B, D) L2-normalized embeddings
        H_anchor, H_neg : sequence (length B) of iterables (e.g., sets/lists) containing parent nodes for the associated classes
            for example H_anchor[i] = set([...parent ids...]) for anchor's original class
        Returns average triplet loss over batch:
            loss_i = ReLU( d(a,p) - d(a,n) + alpha_i )
        with d(x,y) = 1 - <x,y> and alpha_i computed by Eq. 2:
            alpha = alpha_min + (1 - overlap) * (alpha_max - alpha_min)
            overlap = |H(ia) ∩ H(in)| / |H(ia)|  (if |H(ia)| == 0 -> overlap = 0)
        """
        B = emb_anchor.shape[0]
        # cosine similarity (embeddings already L2-normalized): dot product
        dot_ap = (emb_anchor * emb_pos).sum(dim=1)  # (B,)
        dot_an = (emb_anchor * emb_neg).sum(dim=1)  # (B,)
        d_ap = 1.0 - dot_ap
        d_an = 1.0 - dot_an

        # compute per-sample alpha
        alphas = []
        for Ha, Hn in zip(H_anchor, H_neg):
            # ensure sets
            Ha_set = set(Ha) if Ha is not None else set()
            Hn_set = set(Hn) if Hn is not None else set()
            if len(Ha_set) == 0:
                overlap = 0.0
            else:
                overlap = len(Ha_set.intersection(Hn_set)) / float(len(Ha_set))
            alpha = self.alpha_min + (1.0 - overlap) * (self.alpha_max - self.alpha_min)
            alphas.append(alpha)
        alphas_t = emb_anchor.new_tensor(alphas)  # (B,)

        loss_per = F.relu(d_ap - d_an + alphas_t)
        return loss_per.mean()

# -------------------------
# Training step
# -------------------------
def train_step(pos_A: torch.Tensor,
               neg_A: torch.Tensor,
               real_B: torch.Tensor,
               pos_parents: Sequence[Iterable],
               neg_parents: Sequence[Iterable],
               encoder: VGG16_MAC_Encoder,
               generator: UNetGenerator,
               discriminator: PatchGANDiscriminator,
               enc_optimizer: torch.optim.Optimizer,
               gen_optimizer: torch.optim.Optimizer,
               dis_optimizer: torch.optim.Optimizer,
               triplet_loss_fn: HierarchicalTripletLoss,
               lambda_reg: float = 1.0,
               lambda_emb: float = 0.1,
               device: Optional[torch.device] = None):
    """
    One training step that:
      - updates discriminator (real_B vs G(pos_A))
      - updates generator (adversarial + reg + embedding-adversarial)
      - updates encoder (hierarchical triplet loss using anchor = G(pos_A))
    Returns dict of scalar losses for logging.
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder.to(device); generator.to(device); discriminator.to(device)
    pos_A = pos_A.to(device); neg_A = neg_A.to(device); real_B = real_B.to(device)

    bsize = pos_A.shape[0]
    # -----------------
    # 1) Discriminator update
    # -----------------
    discriminator.train()
    dis_optimizer.zero_grad()
    with torch.no_grad():
        fake_B_detached = generator(pos_A).detach()
    logits_real = discriminator(real_B)  # shape B x 1 x h x w
    logits_fake = discriminator(fake_B_detached)

    # BCE with logits: real target 1, fake target 0
    # produce same-shaped targets
    target_real = torch.ones_like(logits_real, device=device)
    target_fake = torch.zeros_like(logits_fake, device=device)
    bce_loss = nn.BCEWithLogitsLoss()
    loss_d_real = bce_loss(logits_real, target_real)
    loss_d_fake = bce_loss(logits_fake, target_fake)
    loss_d = 0.5 * (loss_d_real + loss_d_fake)
    loss_d.backward()
    dis_optimizer.step()

    # -----------------
    # 2) Generator update
    # -----------------
    generator.train()
    gen_optimizer.zero_grad()
    fake_B = generator(pos_A)  # not detached here (we need grads for generator)
    logits_fake_for_gen = discriminator(fake_B)

    # adversarial loss for generator: encourage discriminator to output 1 for fake
    target_gen = torch.ones_like(logits_fake_for_gen, device=device)
    loss_g_adv = bce_loss(logits_fake_for_gen, target_gen)

    # ZNCC regularization: we want to minimize (1 - ZNCC) so generated image remains similar to input
    zncc_vals = zncc_per_image(pos_A, fake_B)  # (B,)
    loss_g_reg = (1.0 - zncc_vals).mean()

    # Embedding adversarial term: Lemb = -d(E(pos), E(G(pos))) -> generator wants to maximize d => Lemb negative
    emb_pos = encoder(pos_A)      # (B, D)
    emb_fake_for_gen = encoder(fake_B)  # (B, D)
    # d = 1 - dot
    d_pos_fake = 1.0 - (emb_pos * emb_fake_for_gen).sum(dim=1)
    loss_g_emb = -d_pos_fake.mean()

    loss_g = loss_g_adv + lambda_reg * loss_g_reg + lambda_emb * loss_g_emb
    loss_g.backward()
    gen_optimizer.step()

    # -----------------
    # 3) Encoder update (triplet; anchor = generated image)
    # -----------------
    encoder.train()
    enc_optimizer.zero_grad()
    with torch.no_grad():
        # use generator's latest weights but detach so encoder update doesn't change generator
        anchor_img = generator(pos_A).detach()

    emb_anchor = encoder(anchor_img)
    emb_pos2 = encoder(pos_A)
    emb_neg = encoder(neg_A.to(device))

    loss_enc = triplet_loss_fn(emb_anchor, emb_pos2, emb_neg, pos_parents, neg_parents)
    loss_enc.backward()
    enc_optimizer.step()

    # return scalars
    return {
        "loss_d": loss_d.item(),
        "loss_g": loss_g.item(),
        "loss_g_adv": loss_g_adv.item(),
        "loss_g_reg": loss_g_reg.item(),
        "loss_g_emb": loss_g_emb.item(),
        "loss_enc": loss_enc.item()
    }

# -------------------------
# Example of building the system + optimizers
# -------------------------
def build_dihe_system(device=None,
                      vgg_pretrained=True,
                      conv4_idx=22,
                      conv5_idx=29,
                      lr_gen=1e-5,
                      lr_disc=1e-5,
                      lr_enc=1e-6,
                      lambda_reg=1.0,
                      lambda_emb=0.1,
                      alpha_min=0.05,
                      alpha_max=0.5):
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    encoder = VGG16_MAC_Encoder(pretrained=vgg_pretrained, conv4_idx=conv4_idx, conv5_idx=conv5_idx, device=device)
    generator = UNetGenerator(in_channels=3, out_channels=3, ngf=64).to(device)
    discriminator = PatchGANDiscriminator(in_channels=3, ndf=64).to(device)

    # optimizers
    gen_opt = torch.optim.Adam(generator.parameters(), lr=lr_gen, betas=(0.5, 0.999))
    dis_opt = torch.optim.Adam(discriminator.parameters(), lr=lr_disc, betas=(0.5, 0.999))
    enc_opt = torch.optim.Adam(encoder.parameters(), lr=lr_enc, betas=(0.5, 0.999))

    triplet_loss = HierarchicalTripletLoss(alpha_min=alpha_min, alpha_max=alpha_max)

    return {
        "encoder": encoder,
        "generator": generator,
        "discriminator": discriminator,
        "enc_optimizer": enc_opt,
        "gen_optimizer": gen_opt,
        "dis_optimizer": dis_opt,
        "triplet_loss": triplet_loss,
        "lambda_reg": lambda_reg,
        "lambda_emb": lambda_emb,
        "device": device
    }

# -------------------------
# Example usage pseudocode:
# -------------------------
# system = build_dihe_system()
# for each training iteration:
#   pos_A, neg_A, real_B, pos_parents, neg_parents = get_batch(...)
#   stats = train_step(pos_A, neg_A, real_B,
#                      pos_parents, neg_parents,
#                      encoder=system['encoder'],
#                      generator=system['generator'],
#                      discriminator=system['discriminator'],
#                      enc_optimizer=system['enc_optimizer'],
#                      gen_optimizer=system['gen_optimizer'],
#                      dis_optimizer=system['dis_optimizer'],
#                      triplet_loss_fn=system['triplet_loss'],
#                      lambda_reg=system['lambda_reg'],
#                      lambda_emb=system['lambda_emb'],
#                      device=system['device'])
#   log(stats)
