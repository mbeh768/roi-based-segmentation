import torch
import matplotlib.pyplot as plt
import sys
from pathlib import Path

path = sys.argv[1] if len(sys.argv) > 1 else "/media/max/500GBSSD/machine_learning_projects/Mayo/att_vent_seg/data/ssl/images/25P48-2/global.pt"
t = torch.load(path, weights_only=True)

print(f"Path:  {path}")
print(f"Shape: {t.shape}")
print(f"Dtype: {t.dtype}")
print(f"Range: [{t.min().item():.4f}, {t.max().item():.4f}]")
print(f"Mean:  {t.mean().item():.4f}")

if t.ndim == 3:
    C = t.shape[0]
    if C == 1:
        plt.imshow(t[0].numpy(), cmap='gray')
    elif C == 3:
        img = t.permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        plt.imshow(img)
    else:
        fig, axes = plt.subplots(1, C, figsize=(4 * C, 4))
        for i in range(C):
            axes[i].imshow(t[i].numpy(), cmap='gray')
            axes[i].set_title(f"Ch {i}")
        plt.tight_layout()
        plt.show()
        sys.exit()
elif t.ndim == 2:
    plt.imshow(t.numpy(), cmap='gray')
else:
    print(f"Unexpected ndim={t.ndim}, showing flattened histogram")
    plt.hist(t.flatten().numpy(), bins=100)

plt.title(Path(path).name)
plt.colorbar()
plt.tight_layout()
plt.show()
