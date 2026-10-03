import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.dataset import TemporalUltrasoundDataset
from src.model import VascularAttentionUNet


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        intersection = (probs_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            probs_flat.sum() + targets_flat.sum() + self.smooth
        )
        return 1.0 - dice

def calculate_metrics_components(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5):
    """Returns the raw intersection and union counts for epoch aggregation."""
    preds = (torch.sigmoid(logits) > threshold).float()
    intersection = (preds * targets).sum().item()
    pred_sum = preds.sum().item()
    target_sum = targets.sum().item()
    union = pred_sum + target_sum - intersection
    return intersection, union, pred_sum, target_sum

def train_model(epochs: int = 5, batch_size: int = 8, lr: float = 1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    dataset = TemporalUltrasoundDataset(num_samples=160, img_size=256)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = VascularAttentionUNet(pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_dice = DiceLoss()

    model.train()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        epoch_intersection = 0.0
        epoch_union = 0.0
        epoch_pred_sum = 0.0
        epoch_target_sum = 0.0
        for batch in loader:
            imgs = batch["image"].to(device)
            masks = batch["mask"].to(device)
            contacts = batch["contact_label"].to(device)
            vessels = batch["vessel_label"].to(device)

            optimizer.zero_grad()
            contact_logits, vessel_logits, seg_logits = model(imgs)

            l_contact = criterion_bce(contact_logits, contacts)
            l_vessel = criterion_bce(vessel_logits, vessels)
            l_seg = (2.0 * criterion_dice(seg_logits, masks)) + criterion_bce(seg_logits, masks)
            total_loss = (0.2 * l_contact) + (0.3 * l_vessel) + (1.0 * l_seg)
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()
            
            # Aggregate metrics components
            inter, uni, p_sum, t_sum = calculate_metrics_components(seg_logits, masks)
            epoch_intersection += inter
            epoch_union += uni
            epoch_pred_sum += p_sum
            epoch_target_sum += t_sum

        epoch_loss = running_loss / len(loader)
        epoch_iou = (epoch_intersection + 1e-6) / (epoch_union + 1e-6)
        epoch_dice = (2.0 * epoch_intersection + 1e-6) / (epoch_pred_sum + epoch_target_sum + 1e-6)
        
        print(f"Epoch [{epoch}/{epochs}] — Loss: {epoch_loss:.4f} | Val Dice: {epoch_dice:.4f} | Val IoU: {epoch_iou:.4f}")

    os.makedirs("models", exist_ok=True)
    save_path = "models/vascular_attention_unet.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Checkpoint successfully saved to: {save_path}")


if __name__ == "__main__":
    train_model()
