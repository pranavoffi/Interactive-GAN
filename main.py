# main.py
import argparse
import torch
import os

from models.gan_model import GANModel
from ui.simple_ui import build_app  # <<< ADD THIS

def parse_args():
    parser = argparse.ArgumentParser(description="Interactive GAN Editing")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use: cuda or cpu")
    parser.add_argument("--network", type=str, default="./weights/stylegan2-car-config-f.pkl", help="Path to pretrained GAN model (.pkl)")
    return parser.parse_args()

def setup_environment(device):
    print(f"Setting device to: {device}")
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU.")
        device = "cpu"
    return device

def load_gan_model(device, network_pkl):
    gan = GANModel(device)
    gan.load_pretrained(network_pkl)
    return gan

def launch_ui(gan_model, device):
    print("Launching Gradio UI...")
    app = build_app(gan_model)
    app.launch(share=True)

def main():
    args = parse_args()
    device = setup_environment(args.device)

    # Step 1: Load GAN
    gan_model = load_gan_model(device, args.network)

    # Step 2: Launch the UI
    launch_ui(gan_model, device)

if __name__ == "__main__":
    main()
