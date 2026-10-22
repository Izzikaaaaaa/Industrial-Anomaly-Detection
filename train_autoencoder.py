import argparse
from .autoencoder import train

def main():
    p = argparse.ArgumentParser(description="Train CNN Autoencoder on normal metal-nut images.")
    p.add_argument("--data", default="data/metal_nut")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-images", type=int, default=None)
    a = p.parse_args()
    train(a.data, a.epochs, a.batch_size, a.lr, a.max_images)

if __name__ == "__main__":
    main()
