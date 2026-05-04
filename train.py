import os
import time
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms
from torchvision.transforms import InterpolationMode

SEED = 42              
np.random.seed(SEED)                                           
torch.manual_seed(SEED)                                      
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True                                                                               
torch.backends.cudnn.benchmark = False                

                         
SCRIPT_DIR = Path(__file__).resolve().parent          
PROJECT_ROOT = SCRIPT_DIR
ROOT = PROJECT_ROOT / "veriseti" / "refuge2"
                                          
IMG_SIZE = 256                                                            
BATCH_SIZE = 4
EPOCHS = 75             
LR = 5e-4                 
NUM_WORKERS = 0 
PIN_MEMORY = False

PRINT_EVERY = 20                                                                                                                                                                              
TEST_EVERY = 3                                                         
DEBUG_LIMIT = 0
                         
PATIENCE = 12                                                               

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"                                                         

USE_AMP = DEVICE == "cuda"
                                             
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
MASK_EXTS = {".bmp", ".png", ".jpg", ".jpeg"}
                                         
SAVE_DIR = PROJECT_ROOT / "model"                           
SAVE_DIR.mkdir(parents=True, exist_ok=True)

                                  
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

                                                                                                                          
def list_files_recursive(folder: Path, exts: set[str]) -> list[Path]:
    if not folder.exists():
        return []
    return sorted([p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in exts])

                                         
def stem_key(p: Path) -> str:
    return p.stem

                                                  
def find_existing_split_root(root: Path) -> Path:
    root = Path(root)
    expected = root / "train" / "images"
    if expected.exists():
        return root
    hits = list(root.rglob("train/images"))
    if hits:
        return hits[0].parents[1]
    return root

def random_crop(img, mask, crop_ratio=0.85):
    w, h = img.size                                      
    new_w = int(w * crop_ratio)                                      
    new_h = int(h * crop_ratio)
    
    left = np.random.randint(0, w - new_w + 1)                                   
    top = np.random.randint(0, h - new_h + 1)                  
    
    img_cropped = img.crop((left, top, left + new_w, top + new_h))                                          
    mask_cropped = mask.crop((left, top, left + new_w, top + new_h))                           
    
    img_resized = img_cropped.resize((w, h), Image.BILINEAR)                                                                 
    mask_resized = mask_cropped.resize((w, h), Image.NEAREST)
    
    return img_resized, mask_resized

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
                                                                            
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)                                  
        probs_flat = probs.view(-1)                               
        targets_flat = targets.view(-1)                           
        intersection = (probs_flat * targets_flat).sum()                                             
        dice = (2. * intersection + self.smooth) / (                
            probs_flat.sum() + targets_flat.sum() + self.smooth
        )
        return 1 - dice

class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()                                  
        self.dice = DiceLoss()                                  
        self.bce_weight = bce_weight                    
        self.dice_weight = dice_weight           

    def forward(self, logits, targets):                
        return self.bce_weight * self.bce(logits, targets) + \
               self.dice_weight * self.dice(logits, targets)

class Refuge2DiscDataset(Dataset):
    def __init__(self, split_dir: str | Path, img_size: int = 256, is_train: bool = False):                         
        self.split_dir = Path(split_dir)
        self.img_dir = self.split_dir / "images"                                                 
        self.mask_dir = self.split_dir / "mask"                                                
        self.is_train = is_train
        self.img_size = img_size

        t0 = time.time()
        img_paths = list_files_recursive(self.img_dir, IMG_EXTS)                                                            
        mask_paths = list_files_recursive(self.mask_dir, MASK_EXTS)                               
        scan_s = time.time() - t0

        mask_map = {stem_key(p): p for p in mask_paths}

        pairs = []                                                   
        missing = 0
                                                                                  
        for ip in img_paths:
            mp = mask_map.get(stem_key(ip))
            if mp is None:
                missing += 1
            else:   
                pairs.append((ip, mp))
        
        if DEBUG_LIMIT and len(pairs) > DEBUG_LIMIT:
            pairs = pairs[:DEBUG_LIMIT]

        print(
            f"[DATASET] split={self.split_dir.name} | is_train={is_train}\n"                                    
            f"pairs={len(pairs)} (img={len(img_paths)} mask={len(mask_paths)} missing={missing}) scan={scan_s:.1f}s"
        )
               
        if len(pairs) == 0:
            raise RuntimeError(
                "Eşleşen (image,mask) bulunamadı.\n"
                f"img_dir={self.img_dir}\nmask_dir={self.mask_dir}\n"
            )

        self.pairs = pairs
                                           
        if is_train:
            self.color_jitter = transforms.ColorJitter(
                brightness=0.2,             
                  contrast=0.2,            
                    saturation=0.2,            
                      hue=0.1            
            )
        else:
            self.color_jitter = None                                          
                             
        self.normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                                                                       
    def __len__(self):
        return len(self.pairs)

    @staticmethod                                  
    def mask_to_disc_binary(mask_pil: Image.Image) -> torch.Tensor:
       
        m = np.array(mask_pil, dtype=np.uint8)                       
        disc = (m < 255).astype(np.float32)                                                                 
        return torch.from_numpy(disc)                             

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]                                                            
                                
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
                                      
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)
                                                     
        if self.is_train:
           
            if np.random.random() < 0.3:
                img, mask = random_crop(img, mask, crop_ratio=0.85)
                                    
            if np.random.random() > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
                                 
            if np.random.random() > 0.5:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
                mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
                              
            rotation = np.random.choice([0, 90, 180, 270])
            if rotation > 0:
                img = img.rotate(rotation)
                mask = mask.rotate(rotation)
                                     
            if self.color_jitter:
                img = self.color_jitter(img)
                                
        img = transforms.ToTensor()(img)
        img = self.normalize(img)
        
        mask = self.mask_to_disc_binary(mask).unsqueeze(0)

        return img, mask

class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(                                
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),                                 
            nn.BatchNorm2d(out_c),                                 
            nn.ReLU(inplace=True),                            
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),                      
            nn.BatchNorm2d(out_c),                             
            nn.ReLU(inplace=True),                          
        )

    def forward(self, x):                                                    
        return self.net(x)

class UNetSmall(nn.Module):
    def __init__(self, base=16):
        super().__init__()
                  
                                          
        self.d1 = DoubleConv(3, base)
        self.d2 = DoubleConv(base, base * 2)
        self.d3 = DoubleConv(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
                                                      
        self.dropout = nn.Dropout2d(0.1)
                                                            
        self.bottleneck = DoubleConv(base * 4, base * 8)
                 
        self.u2 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2)
        self.c2 = DoubleConv(base * 8, base * 4)

        self.u1 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.c1 = DoubleConv(base * 4, base * 2)

        self.u0 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.c0 = DoubleConv(base * 2, base)
                                                     
        self.out = nn.Conv2d(base, 1, 1)
                                                              
        self.aux2 = nn.Conv2d(base * 4, 1, 1)
        self.aux1 = nn.Conv2d(base * 2, 1, 1)

    def forward(self, x):
                 
        x1 = self.d1(x)              
        x2 = self.d2(self.pool(x1))                        
        x3 = self.d3(self.pool(x2))                        
        x3 = self.dropout(x3)                
                                                              
        x = self.bottleneck(self.pool(x3))
        x = self.dropout(x)
                  
        x = self.u2(x)
        x = self.c2(torch.cat([x, x3], dim=1))
        aux2 = self.aux2(x)

        x = self.u1(x)
        x = self.c1(torch.cat([x, x2], dim=1))
        aux1 = self.aux1(x)

        x = self.u0(x)
        x = self.c0(torch.cat([x, x1], dim=1))

        out = self.out(x)
        return out, aux2, aux1

@torch.no_grad()                                                                         
def metrics_binary(logits, y, thr=0.5, eps=1e-7):
    p = torch.sigmoid(logits)
    pb = (p >= thr).float()

    acc = (pb == y).float().mean().item()

    inter = (pb * y).sum(dim=(1, 2, 3))                           
    union = (pb + y - pb * y).sum(dim=(1, 2, 3))                           
                              
    iou = ((inter + eps) / (union + eps)).mean().item()
                                   
    dice = ((2 * inter + eps) / (pb.sum(dim=(1, 2, 3)) + y.sum(dim=(1, 2, 3)) + eps)).mean().item()
    
                         
    tp = inter
    fp = pb.sum(dim=(1, 2, 3)) - inter
    fn = y.sum(dim=(1, 2, 3)) - inter
    
    precision = ((tp + eps) / (tp + fp + eps)).mean().item()
    recall = ((tp + eps) / (tp + fn + eps)).mean().item()
    
    return acc, iou, dice, precision, recall

@torch.no_grad()                                               
def predict_with_tta(model, imgs):
    model.eval()
    
    out1, _, _ = model(imgs)                            
    pred1 = torch.sigmoid(out1)
    
    imgs_hflip = torch.flip(imgs, dims=[-1])                                
    out2, _, _ = model(imgs_hflip)
    pred2 = torch.flip(torch.sigmoid(out2), dims=[-1])
    
    imgs_vflip = torch.flip(imgs, dims=[-2])               
    out3, _, _ = model(imgs_vflip)
    pred3 = torch.flip(torch.sigmoid(out3), dims=[-2])
    
    imgs_both = torch.flip(imgs, dims=[-1, -2])                      
    out4, _, _ = model(imgs_both)
    pred4 = torch.flip(torch.sigmoid(out4), dims=[-1, -2])
    
    return (pred1 + pred2 + pred3 + pred4) / 4.0

@torch.no_grad()
def evaluate_with_tta(model, loader, criterion):
    model.eval()
    
    total_loss = total_acc = total_iou = total_dice = 0.0
    total_precision = total_recall = 0.0
    n = 0
    eps = 1e-7                  
    
    for imgs, masks in loader:
        imgs = imgs.to(DEVICE)  
        masks = masks.to(DEVICE)
        
        pred_avg = predict_with_tta(model, imgs)
                      
        out, _, _ = model(imgs)
        loss = criterion(out, masks)
                      
        pb = (pred_avg >= 0.5).float()
        acc = (pb == masks).float().mean().item()
                                          
        inter = (pb * masks).sum(dim=(1, 2, 3))
        union = (pb + masks - pb * masks).sum(dim=(1, 2, 3))
        iou = ((inter + eps) / (union + eps)).mean().item()
        dice = ((2 * inter + eps) / (pb.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3)) + eps)).mean().item()
                                            
        tp = inter
        fp = pb.sum(dim=(1, 2, 3)) - inter
        fn = masks.sum(dim=(1, 2, 3)) - inter
        precision = ((tp + eps) / (tp + fp + eps)).mean().item()
        recall = ((tp + eps) / (tp + fn + eps)).mean().item()
                                           
        total_loss += loss.item()
        total_acc += acc
        total_iou += iou
        total_dice += dice
        total_precision += precision
        total_recall += recall
        n += 1
    
    return (total_loss / max(1, n), total_acc / max(1, n), total_iou / max(1, n), 
            total_dice / max(1, n), total_precision / max(1, n), total_recall / max(1, n))

def run_epoch(model, loader, criterion, optimizer=None, scaler=None, log_batches=False):
    train = optimizer is not None
    model.train(train)
                                                      
    total_loss = total_acc = total_iou = total_dice = 0.0
    total_precision = total_recall = 0.0
    n = 0

    ALPHA = 0.3                

    for bi, (imgs, masks) in enumerate(loader, start=1):
        imgs = imgs.to(DEVICE)           
        masks = masks.to(DEVICE)
                                
        if train:
            optimizer.zero_grad(set_to_none=True)
                          
        with autocast(enabled=USE_AMP):
            out, aux2, aux1 = model(imgs)
                                                              
            aux2_up = F.interpolate(aux2, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            aux1_up = F.interpolate(aux1, size=masks.shape[-2:], mode="bilinear", align_corners=False)

            loss_main = criterion(out, masks)           
            loss_aux = criterion(aux2_up, masks) + criterion(aux1_up, masks)         
            loss = loss_main + ALPHA * loss_aux

        if train:
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()                     
                optimizer.step()
                         
        acc, iou, dice, precision, recall = metrics_binary(out.detach().float(), masks)
                           
        total_loss += loss.item()
        total_acc += acc
        total_iou += iou
        total_dice += dice
        total_precision += precision
        total_recall += recall
        n += 1
                                               
        if log_batches and (bi % PRINT_EVERY == 0):
            print(f"  batch {bi}/{len(loader)} | loss {loss.item():.4f} | dice {dice:.4f}")

    n = max(1, n)                                                        
    return (total_loss / n, total_acc / n, total_iou / n, total_dice / n, 
            total_precision / n, total_recall / n)

def make_loader(split: str, root: Path):
    is_train = (split == "train")                   
                                          
    ds = Refuge2DiscDataset(root / split, img_size=IMG_SIZE, is_train=is_train)
    return DataLoader(                                           
        ds,
        batch_size=BATCH_SIZE,                                              
        shuffle=is_train,               
        num_workers=NUM_WORKERS,                      
        pin_memory=PIN_MEMORY,      
        drop_last=False,                   
    )

                                                                                                                                   
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):                                                      
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:                                                                                    
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:                       
            self.best_score = score
            self.counter = 0
        return self.early_stop

def plot_training_curves(history, save_dir):
    """Loss ve Dice eğrilerini çiz"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))                                                 
    
                    
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
                                                
    axes[1].plot(history['train_dice'], label='Train Dice', linewidth=2)
    axes[1].plot(history['val_dice'], label='Validation Dice', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Dice Coefficient', fontsize=12)
    axes[1].set_title('Training and Validation Dice Coefficient', fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {save_dir / 'training_curves.png'}")

def plot_segmentation_examples(model, test_loader, save_dir, num_examples=4):
    """Segmentasyon örneklerini görselleştir"""
    model.eval()
    
    fig, axes = plt.subplots(num_examples, 4, figsize=(16, 4*num_examples))
    
                          
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    
    with torch.no_grad():
        for i, (imgs, masks) in enumerate(test_loader):
            if i >= num_examples:
                break
            
            imgs = imgs.to(DEVICE)
            out, _, _ = model(imgs)
            pred = torch.sigmoid(out) > 0.5
            
                            
            pred_tta = predict_with_tta(model, imgs) > 0.5
            
                              
            img = imgs[0].cpu()
            mask = masks[0, 0].cpu().numpy()
            pred_mask = pred[0, 0].cpu().numpy()
            pred_tta_mask = pred_tta[0, 0].cpu().numpy()
            
                         
            img = img * std + mean
            img = img.permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            
                  
            axes[i, 0].imshow(img)
            axes[i, 0].set_title('Input Image', fontsize=12)
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(mask, cmap='gray')
            axes[i, 1].set_title('Ground Truth', fontsize=12)
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(pred_mask, cmap='gray')
            axes[i, 2].set_title('Prediction', fontsize=12)
            axes[i, 2].axis('off')
            
            axes[i, 3].imshow(pred_tta_mask, cmap='gray')
            axes[i, 3].set_title('Prediction (TTA)', fontsize=12)
            axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'segmentation_examples.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Segmentation examples saved to {save_dir / 'segmentation_examples.png'}")

def save_results_to_csv(history, test_results, test_tta_results, save_dir):
    """Sonuçları CSV olarak kaydet"""
    import csv
    
                   
    with open(save_dir / 'training_history.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train_Loss', 'Train_Dice', 'Val_Loss', 'Val_Dice'])
        for i in range(len(history['train_loss'])):
            writer.writerow([i+1, 
                           f"{history['train_loss'][i]:.4f}",
                           f"{history['train_dice'][i]:.4f}",
                           f"{history['val_loss'][i]:.4f}",
                           f"{history['val_dice'][i]:.4f}"])
    
                   
    with open(save_dir / 'final_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Without_TTA', 'With_TTA'])
        writer.writerow(['Loss', f"{test_results[0]:.4f}", f"{test_tta_results[0]:.4f}"])
        writer.writerow(['Accuracy', f"{test_results[1]*100:.2f}%", f"{test_tta_results[1]*100:.2f}%"])
        writer.writerow(['IoU', f"{test_results[2]:.4f}", f"{test_tta_results[2]:.4f}"])
        writer.writerow(['Dice', f"{test_results[3]:.4f}", f"{test_tta_results[3]:.4f}"])
        writer.writerow(['Precision', f"{test_results[4]:.4f}", f"{test_tta_results[4]:.4f}"])
        writer.writerow(['Recall', f"{test_results[5]:.4f}", f"{test_tta_results[5]:.4f}"])
    
    print(f"Results saved to {save_dir / 'training_history.csv'} and {save_dir / 'final_results.csv'}")

def main():
    root = find_existing_split_root(ROOT)
    print("=" * 60)
    print("REFUGE2 Optic Disc Segmentation - U-Net + Deep Supervision")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Mixed Precision: {USE_AMP}")
    print(f"Random Seed: {SEED}")
    print(f"ROOT: {root}")
    print(f"Image Size: {IMG_SIZE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LR}")
    print(f"Max Epochs: {EPOCHS}")
    print(f"Early Stopping Patience: {PATIENCE}")
    print("=" * 60)

                  
    train_loader = make_loader("train", root)
    val_loader = make_loader("val", root)
    test_loader = make_loader("test", root)
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

           
    model = UNetSmall(base=16).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("=" * 60)

                                
    criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    scaler = GradScaler() if USE_AMP else None
    early_stopping = EarlyStopping(patience=PATIENCE)

                     
    save_dir = SAVE_DIR / "runs"
    save_dir.mkdir(parents=True, exist_ok=True)
    best_path = save_dir / "best_unet_improved.pth"

    best_val_dice = -1.0
    best_epoch = 0
    history = {
        "train_loss": [], "train_dice": [], "train_iou": [],
        "val_loss": [], "val_dice": [], "val_iou": [],
        "test_dice": [], "lr": []
    }

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        current_lr = optimizer.param_groups[0]['lr']

                  
        tr = run_epoch(model, train_loader, criterion, optimizer, scaler, log_batches=True)
        
                    
        va = run_epoch(model, val_loader, criterion, optimizer=None, scaler=None, log_batches=False)

                          
        te = None
        if epoch % TEST_EVERY == 0:
            te = run_epoch(model, test_loader, criterion, optimizer=None, scaler=None, log_batches=False)

        dt_min = (time.time() - t0) / 60.0

        tr_loss, tr_acc, tr_iou, tr_dice, tr_prec, tr_rec = tr
        va_loss, va_acc, va_iou, va_dice, va_prec, va_rec = va

                 
        history["train_loss"].append(tr_loss)
        history["train_dice"].append(tr_dice)
        history["train_iou"].append(tr_iou)
        history["val_loss"].append(va_loss)
        history["val_dice"].append(va_dice)
        history["val_iou"].append(va_iou)
        history["lr"].append(current_lr)
        
        if te is not None:
            history["test_dice"].append((epoch, te[3]))

             
        if te is None:
            print(
                f"Epoch {epoch:02d}/{EPOCHS} | {dt_min:.2f} dk | LR {current_lr:.2e} | "
                f"TRAIN loss {tr_loss:.4f} Dice {tr_dice:.4f} | "
                f"VAL loss {va_loss:.4f} Dice {va_dice:.4f}"
            )
        else:
            te_loss, te_acc, te_iou, te_dice, te_prec, te_rec = te
            print(
                f"Epoch {epoch:02d}/{EPOCHS} | {dt_min:.2f} dk | LR {current_lr:.2e} | "
                f"TRAIN loss {tr_loss:.4f} Dice {tr_dice:.4f} | "
                f"VAL loss {va_loss:.4f} Dice {va_dice:.4f} | "
                f"TEST Dice {te_dice:.4f}"
            )

                           
        if va_dice > best_val_dice:
            best_val_dice = va_dice
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': va_dice,
                'history': history,
            }, str(best_path))
            print(f"   New best model saved! Val Dice: {va_dice:.4f}")

        scheduler.step(va_dice)

        if early_stopping(va_dice):
            print(f"\nEarly stopping at epoch {epoch}! No improvement for {PATIENCE} epochs.")
            break

                      
                 
    print("\n" + "=" * 60)
    print("FINAL TEST EVALUATION")
    print("=" * 60)
    
                      
    checkpoint = torch.load(str(best_path), map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
                       
    te = run_epoch(model, test_loader, criterion, optimizer=None, scaler=None, log_batches=False)
    te_loss, te_acc, te_iou, te_dice, te_prec, te_rec = te
    
    print(f"Best Model (Epoch {checkpoint['epoch']}):")
    print(f"  Test Loss: {te_loss:.4f}")
    print(f"  Test Accuracy: {te_acc*100:.2f}%")
    print(f"  Test IoU: {te_iou:.4f}")
    print(f"  Test Dice: {te_dice:.4f}")
    print(f"  Test Precision: {te_prec:.4f}")
    print(f"  Test Recall: {te_rec:.4f}")
    
                    
    print("\n" + "-" * 40)
    print("WITH TEST TIME AUGMENTATION (TTA):")
    print("-" * 40)
    
    te_tta = evaluate_with_tta(model, test_loader, criterion)
    te_loss_tta, te_acc_tta, te_iou_tta, te_dice_tta, te_prec_tta, te_rec_tta = te_tta
    
    print(f"  Test Loss: {te_loss_tta:.4f}")
    print(f"  Test Accuracy: {te_acc_tta*100:.2f}%")
    print(f"  Test IoU: {te_iou_tta:.4f}")
    print(f"  Test Dice: {te_dice_tta:.4f}")
    print(f"  Test Precision: {te_prec_tta:.4f}")
    print(f"  Test Recall: {te_rec_tta:.4f}")
    
    print(f"\n  TTA Improvement: {(te_dice_tta - te_dice)*100:.2f}%")
    
  
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS FOR PAPER")
    print("=" * 60)
    
                     
    plot_training_curves(history, save_dir)
    
                           
    plot_segmentation_examples(model, test_loader, save_dir, num_examples=4)
    
                         
    test_results = (te_loss, te_acc, te_iou, te_dice, te_prec, te_rec)
    test_tta_results = (te_loss_tta, te_acc_tta, te_iou_tta, te_dice_tta, te_prec_tta, te_rec_tta)
    save_results_to_csv(history, test_results, test_tta_results, save_dir)
    
    print("=" * 60)
    print(f"All outputs saved to: {save_dir}")
    print("=" * 60)
    
                             
    print("\n" + "=" * 60)
    print("PAPER-READY SUMMARY")
    print("=" * 60)
    print(f"""
╔══════════════════════════════════════════════════════════╗
║                    FINAL RESULTS                         ║
╠══════════════════════════════════════════════════════════╣
║  Metric          │  Without TTA  │  With TTA            ║
╠══════════════════════════════════════════════════════════╣
║  Dice Coefficient│  {te_dice:.4f}       │  {te_dice_tta:.4f}              ║
║  IoU (Jaccard)   │  {te_iou:.4f}       │  {te_iou_tta:.4f}              ║
║  Precision       │  {te_prec:.4f}       │  {te_prec_tta:.4f}              ║
║  Recall          │  {te_rec:.4f}       │  {te_rec_tta:.4f}              ║
║  Accuracy        │  {te_acc*100:.2f}%      │  {te_acc_tta*100:.2f}%             ║
╠══════════════════════════════════════════════════════════╣
║  Best Epoch: {best_epoch}                                        ║
║  Total Parameters: {total_params:,}                          ║
║  TTA Improvement: {(te_dice_tta - te_dice)*100:+.2f}%                              ║
╚══════════════════════════════════════════════════════════╝
""")

if __name__ == "__main__":
    main()
