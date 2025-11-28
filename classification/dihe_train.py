import os
import random
import math
import glob
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models

# ==========================================
# 1. Dataset & Hierarchy Logic
# ==========================================

class GroceryHierarchyDataset(Dataset):
    """
    Dataset loader that assumes a directory structure:
    Root/MacroCategory/ProductClass/Image.jpg
    
    It returns a triplet: 
    - Positive (Domain A - Studio)
    - Negative (Domain A - Studio)
    - Domain B Sample (Real In-Store Image for GAN discrimination)
    
    And the hierarchy overlap score for the adaptive margin.
    """
    def __init__(self, studio_root, instore_root, transform=None):
        self.studio_root = studio_root
        self.instore_root = instore_root
        self.transform = transform
        
        # Parse Hierarchy
        # Structure: self.classes = ['Cereal/Kelloggs', 'Dairy/Milk', ...]
        # We map index -> path parts
        self.image_paths = []
        self.labels = []
        self.hierarchy_map = {} # class_idx -> list of parent categories
        
        # Walk through Studio directory
        class_idx = 0
        class_to_idx = {}
        
        for macro in os.listdir(studio_root):
            macro_path = os.path.join(studio_root, macro)
            if not os.path.isdir(macro_path): continue
            
            for product in os.listdir(macro_path):
                prod_path = os.path.join(macro_path, product)
                if not os.path.isdir(prod_path): continue
                
                if prod_path not in class_to_idx:
                    class_to_idx[prod_path] = class_idx
                    # Store hierarchy: [Root, Macro, Product]
                    self.hierarchy_map[class_idx] = set([studio_root, macro, product])
                    class_idx += 1
                
                current_label = class_to_idx[prod_path]
                
                for img_name in os.listdir(prod_path):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(os.path.join(prod_path, img_name))
                        self.labels.append(current_label)

        self.instore_images = glob.glob(os.path.join(instore_root, "**/*.jpg"), recursive=True)
        if len(self.instore_images) == 0:
            print("Warning: No domain B (in-store) images found. GAN will train on placeholders.")
            self.instore_images = self.image_paths # Fallback

        # Create Label to Indices map for fast triplet sampling
        self.label_to_indices = {label: np.where(np.array(self.labels) == label)[0]
                                 for label in set(self.labels)}
        self.all_labels = set(self.labels)

    def __len__(self):
        return len(self.image_paths)

    def get_hierarchy_overlap(self, label_a, label_b):
        """
        Calculates |H(ia) n H(in)| / |H(ia)|
        Equation (2) logic from paper.
        """
        set_a = self.hierarchy_map[label_a]
        set_b = self.hierarchy_map[label_b]
        
        intersection = len(set_a.intersection(set_b))
        total_a = len(set_a)
        
        return intersection / total_a if total_a > 0 else 0

    def __getitem__(self, idx):
        # 1. Select Positive (Anchor P is taken from index)
        pos_path = self.image_paths[idx]
        pos_label = self.labels[idx]
        
        # 2. Select Negative
        # Simple strategy: random negative from different class
        neg_label = np.random.choice(list(self.all_labels - {pos_label}))
        neg_idx = np.random.choice(self.label_to_indices[neg_label])
        neg_path = self.image_paths[neg_idx]
        
        # 3. Select Domain B image (Random real in-store image)
        domain_b_path = np.random.choice(self.instore_images)
        
        # Load Images
        pos_img = Image.open(pos_path).convert('RGB')
        neg_img = Image.open(neg_path).convert('RGB')
        dom_b_img = Image.open(domain_b_path).convert('RGB')
        
        if self.transform:
            pos_img = self.transform(pos_img)
            neg_img = self.transform(neg_img)
            dom_b_img = self.transform(dom_b_img)
            
        # Calculate Hierarchy Factor
        # Note: In the paper, 'Anchor' is generated from 'Positive'. 
        # The hierarchy comparison is between Positive(Source) and Negative.
        hierarchy_overlap = self.get_hierarchy_overlap(pos_label, neg_label)
        
        return pos_img, neg_img, dom_b_img, hierarchy_overlap

# ==========================================
# 2. Models (Encoder, Generator, Discriminator)
# ==========================================

class MACLayer(nn.Module):
    """ Maximum Activation of Convolutions (MAC) Layer """
    def __init__(self):
        super(MACLayer, self).__init__()

    def forward(self, x):
        # Global Max Pooling: [Batch, Channel, H, W] -> [Batch, Channel, 1, 1]
        x = F.max_pool2d(x, (x.size(2), x.size(3)))
        # Flatten: [Batch, Channel]
        return x.view(x.size(0), -1)

class Encoder(nn.Module):
    """ 
    VGG16 backbone with MAC descriptor as described in Sec 4.1 
    Paper recommends concatenating conv4_3 and conv5_3 MAC features.
    """
    def __init__(self):
        super(Encoder, self).__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        features = list(vgg.features.children())
        
        # Split VGG to access intermediate layers
        # VGG16: conv4_3 is around layer 22, conv5_3 is last
        self.features_conv4 = nn.Sequential(*features[:23]) 
        self.features_conv5 = nn.Sequential(*features[23:])
        
        self.mac = MACLayer()
        
    def forward(self, x):
        f4 = self.features_conv4(x)
        desc4 = self.mac(f4)
        
        f5 = self.features_conv5(f4)
        desc5 = self.mac(f5)
        
        # Concatenate descriptors
        embedding = torch.cat([desc4, desc5], dim=1)
        
        # L2 Normalize (Crucial for Cosine Similarity)
        return F.normalize(embedding, p=2, dim=1)

class UNetGenerator(nn.Module):
    """ Simple U-Net for Image-to-Image Translation """
    def __init__(self, in_channels=3, out_channels=3):
        super(UNetGenerator, self).__init__()
        
        def down_block(in_feat, out_feat, normalize=True):
            layers = [nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False)]
            if normalize: layers.append(nn.BatchNorm2d(out_feat))
            layers.append(nn.LeakyReLU(0.2))
            return nn.Sequential(*layers)

        def up_block(in_feat, out_feat, dropout=0.0):
            layers = [nn.ConvTranspose2d(in_feat, out_feat, 4, 2, 1, bias=False),
                      nn.BatchNorm2d(out_feat),
                      nn.ReLU(inplace=True)]
            if dropout: layers.append(nn.Dropout(dropout))
            return nn.Sequential(*layers)

        # Downsampling
        self.down1 = down_block(in_channels, 64, normalize=False)
        self.down2 = down_block(64, 128)
        self.down3 = down_block(128, 256)
        self.down4 = down_block(256, 512)
        
        # Upsampling
        self.up1 = up_block(512, 256)
        self.up2 = up_block(512, 128) # concat dim * 2
        self.up3 = up_block(256, 64)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        
        u1 = self.up1(d4)
        u2 = self.up2(torch.cat([u1, d3], 1))
        u3 = self.up3(torch.cat([u2, d2], 1))
        out = self.final(torch.cat([u3, d1], 1))
        return out

class PatchDiscriminator(nn.Module):
    """ PatchGAN Discriminator """
    def __init__(self, in_channels=3):
        super(PatchDiscriminator, self).__init__()
        
        def discriminator_block(in_filters, out_filters, normalization=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization: layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            nn.Conv2d(256, 1, 4, padding=1) # Output 1 channel prediction map
        )

    def forward(self, x):
        return self.model(x)

# ==========================================
# 3. Custom Losses
# ==========================================

def zncc_loss(img1, img2):
    """ Zero Mean Normalized Cross Correlation for Regularization """
    # Flatten: [B, C, H, W] -> [B, -1]
    img1_flat = img1.view(img1.size(0), -1)
    img2_flat = img2.view(img2.size(0), -1)
    
    mean1 = torch.mean(img1_flat, 1, keepdim=True)
    mean2 = torch.mean(img2_flat, 1, keepdim=True)
    
    std1 = torch.std(img1_flat, 1, keepdim=True)
    std2 = torch.std(img2_flat, 1, keepdim=True)
    
    # Normalize
    img1_norm = (img1_flat - mean1) / (std1 + 1e-8)
    img2_norm = (img2_flat - mean2) / (std2 + 1e-8)
    
    # Cross Correlation
    correlation = torch.mean(img1_norm * img2_norm, 1)
    return 1 - torch.mean(correlation) # Minimize 1 - correlation

class HierarchicalTripletLoss(nn.Module):
    """ 
    Equation (1) and (2) from the paper.
    alpha = alpha_min + (1 - overlap) * (alpha_max - alpha_min)
    L = max(0, d(a, p) - d(a, n) + alpha)
    """
    def __init__(self, alpha_min=0.1, alpha_max=0.5):
        super(HierarchicalTripletLoss, self).__init__()
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

    def forward(self, anchor_emb, pos_emb, neg_emb, hierarchy_overlap):
        # Calculate cosine distance: d(x, y) = 1 - cos(x, y)
        # Since embeddings are already L2 normalized, cos(x,y) = x . y
        
        d_ap = 1 - torch.sum(anchor_emb * pos_emb, dim=1)
        d_an = 1 - torch.sum(anchor_emb * neg_emb, dim=1)
        
        # Dynamic margin calculation (Eq. 2)
        # hierarchy_overlap is [Batch_Size] vector
        dynamic_margin = self.alpha_min + \
                         (1 - hierarchy_overlap) * (self.alpha_max - self.alpha_min)
        
        loss = torch.clamp(d_ap - d_an + dynamic_margin, min=0.0)
        return torch.mean(loss)

# ==========================================
# 4. Training Logic
# ==========================================

def train_dihe():
    # Configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 8
    LR_G = 1e-5
    LR_D = 1e-5
    LR_E = 1e-6
    LAMBDA_REG = 1.0
    LAMBDA_EMB = 0.1
    
    # Data Transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Paths (Update these)
    dataset = GroceryHierarchyDataset(
        studio_root='./data/studio', 
        instore_root='./data/instore', 
        transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize Models
    encoder = Encoder().to(DEVICE)
    generator = UNetGenerator().to(DEVICE)
    discriminator = PatchDiscriminator().to(DEVICE)
    
    # Optimizers
    opt_E = optim.Adam(encoder.parameters(), lr=LR_E)
    opt_G = optim.Adam(generator.parameters(), lr=LR_G)
    opt_D = optim.Adam(discriminator.parameters(), lr=LR_D)
    
    # Loss Functions
    criterion_triplet = HierarchicalTripletLoss(alpha_min=0.1, alpha_max=0.5).to(DEVICE)
    criterion_gan = nn.MSELoss() # Least Squares GAN (often more stable) or BCE
    
    print("Starting Training...")
    
    for epoch in range(10):
        for i, (pos_img, neg_img, real_B_img, hierarchy_overlap) in enumerate(dataloader):
            
            pos_img = pos_img.to(DEVICE)       # Studio (Source)
            neg_img = neg_img.to(DEVICE)       # Studio (Source)
            real_B = real_B_img.to(DEVICE)     # In-Store (Target Real)
            overlap = hierarchy_overlap.float().to(DEVICE)
            
            # =================================
            # 1. Train Generator (Domain Adaptation + Hard Negative)
            # =================================
            opt_G.zero_grad()
            
            # Generate "Synthetic Query" (Anchor) from Positive Studio Image
            # i_a^B = G(i_p^A)
            fake_anchor = generator(pos_img)
            
            # A. Adversarial Loss (Fool the discriminator)
            pred_fake = discriminator(fake_anchor)
            valid = torch.ones_like(pred_fake).to(DEVICE)
            loss_adv = criterion_gan(pred_fake, valid)
            
            # B. Regularization Loss (ZNCC - Keep content structure)
            loss_reg = zncc_loss(pos_img, fake_anchor)
            
            # C. Embedding Adversarial Loss (Make it hard for Encoder)
            # We want to MAXIMIZE distance between Enc(FakeAnchor) and Enc(Positive)
            # i.e., MINIMIZE -distance or maximize cosine similarity?
            # Paper says: Lemb = - d(E(ip), E(G(ip))) 
            # d is distance. If we minimize -distance, we maximize distance.
            # This forces G to make the image "hard" to recognize.
            
            # Temporarily freeze encoder for G update (optional but cleaner)
            with torch.no_grad():
                enc_pos = encoder(pos_img)
                enc_fake = encoder(fake_anchor)
                
            # Distance = 1 - Cosine
            dist_emb = 1 - torch.sum(enc_pos * enc_fake, dim=1).mean()
            loss_emb_adv = -dist_emb 
            
            # Total Generator Loss
            loss_G = loss_adv + (LAMBDA_REG * loss_reg) + (LAMBDA_EMB * loss_emb_adv)
            loss_G.backward()
            opt_G.step()
            
            # =================================
            # 2. Train Discriminator
            # =================================
            opt_D.zero_grad()
            
            pred_real = discriminator(real_B)
            pred_fake = discriminator(fake_anchor.detach()) # Detach to stop grad to G
            
            valid = torch.ones_like(pred_real).to(DEVICE)
            fake = torch.zeros_like(pred_fake).to(DEVICE)
            
            loss_real = criterion_gan(pred_real, valid)
            loss_fake = criterion_gan(pred_fake, fake)
            loss_D = 0.5 * (loss_real + loss_fake)
            
            loss_D.backward()
            opt_D.step()
            
            # =================================
            # 3. Train Encoder (Metric Learning)
            # =================================
            opt_E.zero_grad()
            
            # Encoder sees: 
            # 1. Generated Anchor (Synthetic In-store) - The query
            # 2. Positive (Studio) - The database match
            # 3. Negative (Studio) - The database mismatch
            
            # Re-generate fake_anchor with gradients flowing? 
            # No, usually we train E on the output of fixed G for this step, 
            # or we let gradients flow through G if we want G to help E (but G is adversarial).
            # Usually, we treat G's output as input data for E here.
            
            emb_anchor = encoder(fake_anchor.detach())
            emb_pos = encoder(pos_img)
            emb_neg = encoder(neg_img)
            
            loss_E = criterion_triplet(emb_anchor, emb_pos, emb_neg, overlap)
            
            loss_E.backward()
            opt_E.step()
            
            if i % 10 == 0:
                print(f"[Epoch {epoch}/{i}] Loss G: {loss_G.item():.4f} | Loss D: {loss_D.item():.4f} | Loss E: {loss_E.item():.4f}")

    # Save Model
    torch.save(encoder.state_dict(), "grocery_encoder.pth")
    print("Training Complete. Encoder saved.")

if __name__ == "__main__":
    # Create dummy dirs for demonstration if they don't exist
    if not os.path.exists('./data'):
        os.makedirs('./data/studio/Cereal/CornFlakes', exist_ok=True)
        os.makedirs('./data/instore', exist_ok=True)
        print("Created dummy directory structure in ./data. Please populate with images before running.")
    
    try:
        train_dihe()
    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure you have actual images in ./data/studio and ./data/instore")