import argparse
from .patchcore_model import train

def main():
    p = argparse.ArgumentParser(description="Train PatchCore-style detector on normal metal-nut images.")
    p.add_argument("--data", default="data/metal_nut")
    p.add_argument("--max-bank", type=int, default=5000)
    p.add_argument("--max-images", type=int, default=None)
    a = p.parse_args()
    train(a.data, a.max_bank, a.max_images)

if __name__ == "__main__":
    main()
