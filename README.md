# 🎨 Interactive Sketch-to-Image Generation using GANs

A real-time AI system that transforms rough sketches into photo-realistic images using StyleGAN2. Draw on a canvas, and the app optimizes a latent vector through a pretrained GAN until the generated image matches your sketch — with a Gradio UI for a fast, interactive editing experience.

---

## 🧠 Overview

The project loads a pretrained **StyleGAN2-ADA** generator and exposes a simple sketch canvas in the browser. When you draw and hit generate, the system searches the GAN's latent space (via gradient descent on an **LPIPS** perceptual loss) for the latent vector that produces an image most similar to your sketch, then renders that image back to you.

## ⚙️ Core Components

| Component | File | Description |
|---|---|---|
| GAN backbone | `models/gan_model.py`, `models/legacy.py` | Loads a pretrained StyleGAN2-ADA `.pkl` checkpoint and wraps it for sampling and generation. |
| Latent optimization | `models/gan_model.py` → `optimize_latent_for_sketch()` | Initializes a random latent `z`, then runs Adam gradient descent against an LPIPS perceptual loss between the generated image and the user's sketch. |
| Interactive UI | `ui/simple_ui.py` | A Gradio `Blocks` app with a sketchpad, brush size/color controls, and a "Generate from Sketch" button. |
| Entry point | `main.py` | CLI that loads the GAN onto CPU/GPU and launches the Gradio app. |
| StyleGAN2 internals | `models/dnnlib/`, `models/torch_utils/`, `models/training/` | Vendored utility and network-definition modules from NVIDIA's official StyleGAN2-ADA-PyTorch implementation, required to unpickle and run the generator. |
| Reference implementation | `stylegan2-ada-pytorch/` | A full copy of NVIDIA's official [StyleGAN2-ADA-PyTorch](https://github.com/NVlabs/stylegan2-ada-pytorch) repo, kept for reference and for its training/projection scripts (`train.py`, `projector.py`, `generate.py`, etc.). |

## 🚀 How It Works

1. `main.py` loads a pretrained StyleGAN2-ADA generator (`GANModel.load_pretrained`) onto CPU or GPU.
2. `ui/simple_ui.py` launches a Gradio interface with a sketch canvas.
3. When you draw a sketch and click **Generate from Sketch**:
   - The sketch is converted to a tensor and resized to the generator's native resolution.
   - A random latent vector `z` is initialized and optimized for a fixed number of steps using Adam, minimizing the LPIPS perceptual distance between the generated image and your sketch.
   - The best image found during optimization is returned and displayed in the UI.

---

## 🚧 Project Status

This repo currently implements the **GAN backbone + LPIPS-based latent optimization + Gradio sketch UI** pipeline described above end-to-end. The broader vision for this project also includes the features below, which are **planned but not yet implemented in code**:

- CLIP-guided loss for improved semantic alignment during projection.
- RAFT optical-flow-based transfer of sketch edits onto real photographs.
- Guided upsampling / lightweight CNN post-processing for sharpening and refinement.
- Text-driven editing via CLIP guidance and super-resolution.
- A Streamlit-based alternative front end.

See [Roadmap](#-roadmap) for details.

---

## 🛠️ Tech Stack

- **Language:** Python
- **GAN model:** StyleGAN2-ADA (PyTorch) — supports 256×256 and 512×512 outputs depending on the checkpoint used
- **Perceptual loss:** LPIPS
- **Deep learning framework:** PyTorch / TorchVision
- **UI:** Gradio
- **Numerical computing:** NumPy

*(CLIP, RAFT, and Streamlit are part of the project's planned scope — see [Roadmap](#-roadmap).)*

---

## 📁 Project Structure

```
Interactive-GAN/
├── main.py                       # CLI entry point: loads GAN + launches Gradio UI
├── models/
│   ├── gan_model.py               # GANModel: load checkpoint, sample latents, generate, optimize-for-sketch
│   ├── legacy.py                  # Pickle loader for legacy StyleGAN2 network formats
│   ├── dnnlib/                    # NVIDIA utility library (vendored)
│   ├── torch_utils/                # Custom ops / persistence helpers used by the StyleGAN2 network code (vendored)
│   └── training/                  # StyleGAN2 network/training definitions (vendored, used for architecture defs)
├── ui/
│   └── simple_ui.py                # Gradio Blocks app: sketchpad, brush controls, generate button
├── weights/                        # (not included) place pretrained .pkl checkpoints here
└── stylegan2-ada-pytorch/          # Official NVIDIA StyleGAN2-ADA-PyTorch repo (reference + training/projection scripts)
```

---

## 🧰 Prerequisites

- Python 3.7+ (3.9 recommended, matching the vendored StyleGAN2-ADA codebase)
- PyTorch with matching TorchVision (CUDA build strongly recommended — see [pytorch.org](https://pytorch.org/) for the right command for your system)
- An NVIDIA GPU with CUDA is recommended for reasonable generation speed; CPU fallback is supported but slow.
- A pretrained StyleGAN2 / StyleGAN2-ADA `.pkl` checkpoint (see [Pretrained Models](#-pretrained-models) below).

## 📦 Installation

```bash
git clone https://github.com/pranavoffi/Interactive-GAN.git
cd Interactive-GAN

python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate

# Install PyTorch separately first, matching your CUDA version, e.g.:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

pip install gradio lpips numpy pillow click requests tqdm pyspng ninja imageio-ffmpeg==0.4.3
```

> The `click`, `requests`, `tqdm`, `pyspng`, `ninja`, and `imageio-ffmpeg` packages are required by the vendored StyleGAN2-ADA network code for compiling custom CUDA ops and loading checkpoints — they're listed in `stylegan2-ada-pytorch`'s own documentation.

## 🖼️ Pretrained Models

This project does not ship model weights. Download a pretrained StyleGAN2-ADA checkpoint (`.pkl`) from [NVIDIA's model zoo](https://github.com/NVlabs/stylegan2-ada-pytorch#data-repository) — for example `ffhq.pkl`, `afhqcat.pkl`, or a car/face checkpoint of your choice — and place it under `weights/`.

## ▶️ Usage

```bash
python main.py --network ./weights/your-model.pkl --device cuda
```

| Flag | Default | Description |
|---|---|---|
| `--network` | `./weights/stylegan2-car-config-f.pkl` | Path to the pretrained `.pkl` checkpoint to load. |
| `--device` | `cuda` | `cuda` or `cpu`. Falls back to CPU automatically if CUDA isn't available. |

This launches a Gradio app (with a public share link) where you can:
1. Draw a rough sketch on the canvas.
2. Adjust brush thickness and color.
3. Click **✨ Generate from Sketch** to run latent optimization and view the resulting image.

---

## 🗺️ Roadmap

- [ ] Integrate CLIP loss alongside LPIPS for better semantic alignment during projection.
- [ ] Add RAFT optical flow to transfer sketch edits onto real photographs while preserving structure.
- [ ] Add guided upsampling / lightweight CNN post-processing for sharper, refined output.
- [ ] Explore text-driven editing via CLIP guidance.
- [ ] Add a Streamlit-based alternative UI.
- [ ] Cache/checkpoint latent optimization for faster iterative edits.

---

## ⚠️ Notes & Limitations

- Latent optimization currently starts from a random vector rather than a proper GAN-inversion projection, so results depend on the number of optimization steps and may vary between runs.
- The `stylegan2-ada-pytorch/` subdirectory is NVIDIA's official repository, included for reference and to support training/projecting your own models — it is **not** automatically invoked by `main.py`.
- Generation quality and resolution depend entirely on the pretrained checkpoint you load (e.g. a face-trained model will not generate convincing cars, and vice versa).

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome — especially around the CLIP/RAFT/post-processing items in the roadmap. Feel free to open an issue or submit a pull request.

## 👤 Author

**Pranav** — [@pranavoffi](https://github.com/pranavoffi)

## 📄 License

The original code in this repository (`main.py`, `ui/`, `models/gan_model.py`) is provided as-is; add a `LICENSE` file (e.g. MIT) if you'd like to specify usage terms for it.

The vendored `stylegan2-ada-pytorch/` directory and the StyleGAN2-ADA architecture code under `models/dnnlib/`, `models/torch_utils/`, and `models/training/` are © NVIDIA Corporation and are distributed under the **NVIDIA Source Code License**, which permits non-commercial use only. See `stylegan2-ada-pytorch/LICENSE.txt` for the full terms before using this project commercially or redistributing pretrained checkpoints.
