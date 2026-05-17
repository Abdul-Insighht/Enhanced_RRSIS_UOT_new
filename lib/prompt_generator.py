import torch
import torch.nn as nn
import torch.nn.functional as F


class GroundingAwarePromptGenerator(nn.Module):
    """
    Grounding-Aware Prompts Generation (GPG) module inspired by SGSRF.

    This module takes the OT transport plan (which acts as a structural-semantic
    alignment map) and generates explicit geometric prompts for the SAM3 Decoder.
    
    It extracts:
    1. Sparse Prompts (Points): Coordinates of the highest probability regions.
    2. Dense Prompts (Masks): Low-resolution coarse mask priors.
    """
    def __init__(self, num_points=1):
        super().__init__()
        self.num_points = num_points
        print(f"[GPG] Initialized Grounding-Aware Prompt Generator (num_points={num_points})")

    @torch.no_grad()
    def forward(self, P, text_mask, original_image_size, device):
        """
        Generate geometric prompts from the OT transport plan.

        Args:
            P: (B, HW, seq) transport plan from the deepest OT aligner.
            text_mask: (B, seq) boolean mask (True = padding).
            original_image_size: int, size of the input image (e.g., 504).
            device: torch.device.

        Returns:
            points: (B, num_points, 2) normalized coordinates [0, 1] or absolute?
                    SAM3 expects absolute coordinates if scaled, but for the FindStage 
                    we can provide absolute pixel coordinates.
            points_mask: (B, num_points) boolean mask.
            dense_mask: (B, 1, 256, 256) dense mask prior for SAM3.
        """
        B, HW, seq = P.shape
        H = W = int(HW ** 0.5)

        # 1. Mask out padding tokens in the transport plan
        if text_mask is not None:
            # text_mask is True for padding. We want to keep valid tokens (False)
            valid_mask = (~text_mask).unsqueeze(1)  # (B, 1, seq)
            P_valid = P * valid_mask.float()
        else:
            P_valid = P

        # 2. Aggregate over sequence to get a global spatial heatmap
        heatmap = P_valid.sum(dim=-1)  # (B, HW)
        heatmap = heatmap.view(B, H, W)

        # 3. Generate Sparse Prompts (Top-K Points)
        # Flatten and get topk
        flat_heatmap = heatmap.view(B, -1)
        _, topk_idx = torch.topk(flat_heatmap, self.num_points, dim=-1)

        # Convert 1D indices to 2D (y, x) coordinates in the feature map space
        y_feat = torch.div(topk_idx, W, rounding_mode='floor')
        x_feat = topk_idx % W

        # Scale coordinates to the original image size
        # Feature map size is H x W. Image size is original_image_size x original_image_size
        scale_y = original_image_size / H
        scale_x = original_image_size / W

        y_img = (y_feat.float() + 0.5) * scale_y
        x_img = (x_feat.float() + 0.5) * scale_x

        # SAM3 expects points as (x, y)
        points = torch.stack((x_img, y_img), dim=-1)  # (B, num_points, 2)
        points_mask = torch.ones((B, self.num_points), dtype=torch.bool, device=device)

        # 4. Generate Dense Prompts (Masks)
        # SAM3 prompt encoder expects dense masks to be typically 256x256 (1/4 of 1024)
        # For 504x504, 256x256 is fine, SAM3 will interpolate internally if needed.
        dense_mask = F.interpolate(
            heatmap.unsqueeze(1),  # (B, 1, H, W)
            size=(256, 256),
            mode='bilinear',
            align_corners=False
        )
        
        # Normalize dense mask to roughly [0, 1] or standard normal
        b_max = dense_mask.view(B, -1).max(dim=1)[0].view(B, 1, 1, 1) + 1e-6
        dense_mask = dense_mask / b_max

        # We don't return dense mask as it's not strictly necessary for find_stage unless we modify it heavily.
        # Providing points is the most critical part of Grounding-first.
        
        return points, points_mask, dense_mask
