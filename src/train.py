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
    preds = (torch.sigmoid(logits) > threshold).float()
    intersection = (preds * targets).sum().item()
    pred_sum = preds.sum().item()
    target_sum = targets.sum().item()
    union = pred_sum + target_sum - intersection
    return intersection, union, pred_sum, target_sum


def train_model(epochs: int = 5, batch_size: int = 8, lr: float = 1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Training and Validation sets
    train_dataset = TemporalUltrasoundDataset(num_samples=160, img_size=256)
    val_dataset = TemporalUltrasoundDataset(num_samples=32, img_size=256)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = VascularAttentionUNet(pretrained=True).to(device)
    # Filter the optimizer to only update the unfrozen decoder and linear heads
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)

    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_dice = DiceLoss()

    for epoch in range(1, epochs + 1):
        # ------------------ TRAINING ------------------
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            imgs = batch["image"].to(device)
            masks = batch["mask"].to(device)
            contacts = batch["contact_label"].to(device)
            vessels = batch["vessel_label"].to(device)

            optimizer.zero_grad()
            contact_logits, vessel_logits, seg_logits = model(imgs)

            l_contact = criterion_bce(contact_logits, contacts)
            l_vessel = criterion_bce(vessel_logits, vessels)
            l_seg = (2.0 * criterion_dice(seg_logits, masks)) + criterion_bce(seg_logits, masks)

            total_loss = (1.0 * l_contact) + (1.0 * l_vessel) + (1.0 * l_seg)
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()

        train_loss = running_loss / len(train_loader)

        # ------------------ EVALUATION (model.eval()) ------------------
        model.eval()
        val_intersection = 0.0
        val_union = 0.0
        val_pred_sum = 0.0
        val_target_sum = 0.0
        val_correct_contact = 0
        val_correct_vessel = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device)
                masks = batch["mask"].to(device)
                contacts = batch["contact_label"].to(device)
                vessels = batch["vessel_label"].to(device)

                contact_logits, vessel_logits, seg_logits = model(imgs)

                pred_c = (torch.sigmoid(contact_logits) > 0.5).float()
                pred_v = (torch.sigmoid(vessel_logits) > 0.5).float()

                val_correct_contact += (pred_c == contacts).sum().item()
                val_correct_vessel += (pred_v == vessels).sum().item()
                val_total += contacts.size(0)

                inter, uni, p_sum, t_sum = calculate_metrics_components(seg_logits, masks)
                val_intersection += inter
                val_union += uni
                val_pred_sum += p_sum
                val_target_sum += t_sum

        val_dice = (2.0 * val_intersection + 1e-6) / (val_pred_sum + val_target_sum + 1e-6)
        val_iou = (val_intersection + 1e-6) / (val_union + 1e-6)
        val_contact_acc = (val_correct_contact / val_total) * 100
        val_vessel_acc = (val_correct_vessel / val_total) * 100

        print(
            f"Epoch [{epoch}/{epochs}] — Train Loss: {train_loss:.4f} | "
            f"Eval Contact: {val_contact_acc:.1f}% | Eval Vessel: {val_vessel_acc:.1f}% | "
            f"Eval Dice: {val_dice:.4f} | Eval IoU: {val_iou:.4f}"
        )

    os.makedirs("models", exist_ok=True)
    save_path = "models/vascular_attention_unet.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to: {save_path}")


if __name__ == "__main__":
    train_model()
