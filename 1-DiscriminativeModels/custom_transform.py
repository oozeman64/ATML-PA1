import torch.nn.functional as F
import torch
import random

# Need this since F.pad requires individual images. Make it a class for cleaner code and pass it to Transform
class TranslateTransform:
    def __init__(self, shift):
        self.shift = shift

    def __call__(self, img):
        C,H,W = img.shape
        dir = random.randint(0,3)
        if dir == 0:
            return F.pad(img, (self.shift,0,0,0))[:, :, :W]
        if dir == 1:
            return F.pad(img, (0,self.shift,0,0))[:, :, self.shift:]
        if dir == 2:
            return F.pad(img, (0,0,self.shift,0))[:, :H, :]
        if dir == 3:
            return F.pad(img, (0,0,0,self.shift))[:, self.shift:, :]

# Expects 224 x 224 img -> 28 x 28 patches -> 8 x 8 grid
class MaskPatchTransform:
    def __init__(self, num_patches):
        self.num_patches = num_patches

    def __call__(self, img):
        for i in range(self.num_patches):
            patch_index = random.randint(0, 63)
            patches = img.unfold(1,28,28).unfold(2,28,28)  # [C,8,8,28,28]
            patches = patches.permute(1,2,0,3,4).reshape(-1,3,28,28)  # [64,3,28,28]

            patches[patch_index] = torch.zeros_like(patches[patch_index])
            patches_grid = patches.reshape(8, 8, 3, 28, 28)
            patches_grid = patches_grid.permute(2,0,3,1,4)

            img = patches_grid.reshape(3, 8*28, 8*28)
        return img
    
class ShufflePatchTransform:
    def __init__(self):
        pass

    def __call__(self, img):
        patches = img.unfold(1,16,16).unfold(2,16,16)  # [C,14,14,16,16]
        patches = patches.permute(1,2,0,3,4).reshape(-1,3,14,14)  # [196,3,16,16]

        # shuffle
        idx = torch.randperm(patches.size(0))
        patches = patches[idx]

        patches_grid = patches.reshape(14, 14, 3, 16, 16)
        patches_grid = patches_grid.permute(2,0,3,1,4)

        img = patches_grid.reshape(3, 16*14, 16*14)
        return img