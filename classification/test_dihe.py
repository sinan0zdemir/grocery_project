# test_dihe.py
import torch
from dihe_pytorch import (
    VGG16_MAC_Encoder,
    UNetGenerator,
    PatchGANDiscriminator,
    zncc_per_image,
    HierarchicalTripletLoss,
    train_step,
    build_dihe_system
)

def test_encoder_forward_backward():
    print("Testing Encoder...")
    encoder = VGG16_MAC_Encoder(pretrained=False)
    x = torch.randn(2, 3, 256, 256)
    emb = encoder(x)
    assert emb.shape[0] == 2
    assert len(emb.shape) == 2
    loss = emb.sum()
    loss.backward()
    print("Encoder OK\n")

def test_generator_forward_backward():
    print("Testing Generator...")
    G = UNetGenerator()
    x = torch.randn(2, 3, 256, 256)
    out = G(x)
    assert out.shape == x.shape
    out.mean().backward()
    print("Generator OK\n")

def test_discriminator_forward_backward():
    print("Testing Discriminator...")
    D = PatchGANDiscriminator()
    x = torch.randn(2, 3, 256, 256)
    out = D(x)
    assert out.ndim == 4
    out.mean().backward()
    print("Discriminator OK\n")

def test_zncc():
    print("Testing ZNCC...")
    x = torch.randn(2, 3, 64, 64)
    y = torch.randn(2, 3, 64, 64)
    z = zncc_per_image(x, y)
    assert z.shape == (2,)
    print("ZNCC OK\n")

def test_triplet_loss():
    print("Testing Hierarchical Triplet Loss...")
    loss_fn = HierarchicalTripletLoss()

    a = torch.randn(2, 512, requires_grad=True)
    p = torch.randn(2, 512, requires_grad=True)
    n = torch.randn(2, 512, requires_grad=True)

    Ha = [{"1","2"}, {"1"}]
    Hn = [{"2","3"}, set()]

    loss = loss_fn(a, p, n, Ha, Hn)
    loss.backward()   # now works
    print("Triplet loss OK\n")


def test_train_step():
    print("Testing full train_step...")

    system = build_dihe_system(vgg_pretrained=False)
    B = 2
    pos = torch.randn(B, 3, 256, 256)
    neg = torch.randn(B, 3, 256, 256)
    real = torch.randn(B, 3, 256, 256)
    pos_parents = [{"1","2"}, {"A"}]
    neg_parents = [{"C"}, {"A"}]

    stats = train_step(
        pos, neg, real,
        pos_parents, neg_parents,
        encoder=system["encoder"],
        generator=system["generator"],
        discriminator=system["discriminator"],
        enc_optimizer=system["enc_optimizer"],
        gen_optimizer=system["gen_optimizer"],
        dis_optimizer=system["dis_optimizer"],
        triplet_loss_fn=system["triplet_loss"],
        lambda_reg=system["lambda_reg"],
        lambda_emb=system["lambda_emb"],
        device=system["device"]
    )
    print("Stats:", stats)
    print("train_step OK\n")

if __name__ == "__main__":
    test_encoder_forward_backward()
    test_generator_forward_backward()
    test_discriminator_forward_backward()
    test_zncc()
    test_triplet_loss()
    test_train_step()
    print("All tests passed.")
