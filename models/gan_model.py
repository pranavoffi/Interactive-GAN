# /models/gan_model.py
import torch
import os
import lpips
import torchvision.transforms as T
from PIL import Image

# Since legacy.py and dnnlib are in the same folder, we can import directly
from models.legacy import load_network_pkl

class GANModel:
    def __init__(self, device="cuda"):
        self.device = device
        self.generator = None

    def load_pretrained(self, network_pkl):
        """
        Load a pretrained StyleGAN2-ADA model properly.
        """
        if not os.path.exists(network_pkl):
            raise FileNotFoundError(f"Model checkpoint not found: {network_pkl}")

        print(f"Loading GAN model from {network_pkl}...")
        with open(network_pkl, 'rb') as f:
            G = load_network_pkl(f)['G_ema'].to(self.device)
            self.generator = G.eval()
        print("GAN model loaded successfully!")

    def sample_latent(self, batch_size=1):
        if self.generator is None:
            raise ValueError("Generator not loaded!")
        z_dim = self.generator.z_dim
        return torch.randn(batch_size, z_dim, device=self.device)

    def generate(self, z):
        if self.generator is None:
            raise ValueError("Generator not loaded!")
        img = self.generator(z, None)  # (N, C, H, W)
        img = (img + 1) / 2  # Normalize to [0, 1]
        return img

    def optimize_latent_for_sketch(self, sketch_img_pil, steps=100, lr=0.05):
        """
        Optimize a latent vector z so that generated image matches the sketch using LPIPS loss.
        sketch_img_pil: PIL.Image from Gradio Sketchpad
        """
        if self.generator is None:
            raise ValueError("Generator not loaded!")

        # Init LPIPS
        percept = lpips.LPIPS(net='alex').to(self.device).eval()

        # Preprocess sketch image to tensor
        preprocess = T.Compose([
            T.Resize((self.generator.img_resolution, self.generator.img_resolution)),
            T.ToTensor(),
        ])
        target_tensor = preprocess(sketch_img_pil).unsqueeze(0).to(self.device)

        if target_tensor.shape[1] == 1:
            # Grayscale to 3-channel
            target_tensor = target_tensor.repeat(1, 3, 1, 1)

        # Start with random latent
        z = torch.randn(1, self.generator.z_dim, device=self.device, requires_grad=True)

        optimizer = torch.optim.Adam([z], lr=lr)

        best_loss = float('inf')
        best_img = None

        for i in range(steps):
            optimizer.zero_grad()
            gen_img = self.generator(z, None)
            gen_norm = (gen_img + 1) / 2  # Normalize to [0,1]
            loss = percept(gen_norm, target_tensor)
            loss.backward()
            optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_img = gen_norm.detach()

        return best_img
