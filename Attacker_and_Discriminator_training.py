
from sched import scheduler

import kagglehub
import numpy as np
import random
import torch
import torchaudio
import torchaudio.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
import matplotlib.pyplot as plt
import os

class AudioUtil:
    @staticmethod
    def open(audio_file):
        sig, sr = torchaudio.load(audio_file)
        return sig, sr

    @staticmethod
    def rechannel(aud, new_channel):
        sig, sr = aud
        if sig.shape[0] == new_channel:
            return aud
        if new_channel == 1:
            resig = sig[:1, :]
        else:
            resig = sig.repeat(new_channel, 1)
        return resig, sr

    @staticmethod
    def resample(aud, newsr):
        sig, sr = aud
        if sr == newsr:
            return aud
        num_channels = sig.shape[0]
        resig = transforms.Resample(sr, newsr)(sig[:1, :])
        if num_channels > 1:
            retwo = transforms.Resample(sr, newsr)(sig[1:, :])
            resig = torch.cat([resig, retwo], dim=0)
        return resig, newsr

    @staticmethod
    def pad_trunc(aud, max_ms):
        sig, sr = aud
        num_rows, sig_len = sig.shape
        max_len = sr // 1000 * max_ms
        if sig_len > max_len:
            sig = sig[:, :max_len]
        elif sig_len < max_len:
            pad_begin_len = random.randint(0, max_len - sig_len)
            pad_end_len = max_len - sig_len - pad_begin_len
            pad_begin = torch.zeros((num_rows, pad_begin_len))
            pad_end = torch.zeros((num_rows, pad_end_len))
            sig = torch.cat((pad_begin, sig, pad_end), dim=1)
        return sig, sr

    @staticmethod
    def time_shift(aud, shift_pct):
        sig, sr = aud
        _, sig_len = sig.shape
        shift_amt = int(random.random() * shift_pct * sig_len)
        return sig.roll(shift_amt), sr

    @staticmethod
    def add_noise(aud, noise_level=0.005):
        sig, sr = aud
        noise = torch.randn_like(sig) * noise_level
        return sig + noise, sr

    @staticmethod
    def change_gain(aud, gain_db_range=(-6, 6)):
        sig, sr = aud
        gain_db = random.uniform(*gain_db_range)
        gain = 10 ** (gain_db / 20)
        return sig * gain, sr

    @staticmethod
    def pitch_shift(aud, n_steps=2):
        sig, sr = aud
        return torchaudio.functional.pitch_shift(sig, sr, n_steps), sr

    @staticmethod
    def spectro_gram(aud, n_mels=64, n_fft=780, hop_len=195):
        sig, sr = aud
        spec = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)
        spec = transforms.AmplitudeToDB(top_db=80)(spec)
        return spec

    @staticmethod
    def spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
        _, n_mels, n_steps = spec.shape
        mask_value = spec.mean()
        aug_spec = spec
        freq_mask_param = int(max_mask_pct * n_mels)
        for _ in range(n_freq_masks):
            aug_spec = transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)
        time_mask_param = int(max_mask_pct * n_steps)
        for _ in range(n_time_masks):
            aug_spec = transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)
        return aug_spec


class SoundDS(Dataset):
    def __init__(self, data_path, label, mode="original"):
        self.label = label
        self.data_path = [str(p) for p in data_path]
        self.duration = 4000
        self.sr = 16000
        self.channel = 2
        self.shift_pct = 0.4
        self.mode = mode

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        audio_file = self.data_path[idx]
        class_id = self.label[idx]

        aud = AudioUtil.open(audio_file)
        aud = AudioUtil.resample(aud, self.sr)
        aud = AudioUtil.rechannel(aud, self.channel)
        aud = AudioUtil.pad_trunc(aud, self.duration)

        if self.mode == "time_shift":
            aud = AudioUtil.time_shift(aud, self.shift_pct)
        elif self.mode == "add_noise":
            aud = AudioUtil.add_noise(aud, noise_level=random.uniform(0.005, 0.05))
        elif self.mode == "pitch_shift":
            aud = AudioUtil.pitch_shift(aud, n_steps=random.randint(1,3))
        elif self.mode == "combined":
            aud = AudioUtil.time_shift(aud, self.shift_pct)
            aud = AudioUtil.add_noise(aud, noise_level=random.uniform(0.005, 0.05))

        sgram = AudioUtil.spectro_gram(aud, n_mels=64, n_fft=780, hop_len=195)

        if self.mode in ["spectro_augment", "combined"]:
            sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=random.randint(1,3), n_time_masks=random.randint(1,3))

        return sgram, class_id

from torch.utils.data import random_split, ConcatDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class AttentionPool(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.attn = nn.Linear(in_dim, 1)

    def forward(self, x):
        # x: [B, T, D]
        scores = self.attn(x)              # [B, T, 1]
        weights = F.softmax(scores, dim=1) # [B, T, 1]
        return (weights * x).sum(dim=1)    # [B, D]

class CRNNWithAttn(nn.Module):
    def __init__(self,  pretrained=True, hidden_size=128, num_layers=1, dropout=0.2):
        super().__init__()
        # 1. Pretrained ResNet18
        if pretrained:
          resnet = models.resnet18(weights='DEFAULT')
        else:
          resnet = models.resnet18()
        # Adapt first conv to accept 1-channel input
        w = resnet.conv1.weight.data.clone()
        resnet.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        resnet.conv1.weight.data[:, 0] = w[:, 0]
        # Remove final pooling & fc
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        # 2. Bi-GRU for temporal modeling
        self.gru = nn.GRU(
            input_size=512,          # ResNet last block outputs 512 channels
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers>1 else 0.0
        )

        # 3. Attention pooling
        self.attn_pool = AttentionPool(hidden_size*2)

        # 4. Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        # x: [B, 1, F, T]
        feat = self.backbone(x)            # [B, 512, F', T']
        feat = feat.mean(dim=2)            # collapse freq → [B,512,T']
        feat = feat.permute(0,2,1)         # → [B,T',512]

        out, _ = self.gru(feat)            # → [B,T',2*hidden_size]
        pooled = self.attn_pool(out)       # → [B,2*hidden_size]
        return self.classifier(pooled)     # → [B,1]




import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """Double conv block - extracts richer features at each scale"""
    def __init__(self, in_channels, out_channels, use_bn=True, activation='leaky'):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=not use_bn),
        ]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True) if activation == 'leaky' 
                      else nn.ReLU(inplace=True))
        # Second conv at same resolution - doubles feature extraction depth
        layers += [
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=not use_bn),
        ]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True) if activation == 'leaky' 
                      else nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class EncoderBlock(nn.Module):
    """ConvBlock + strided downsampling"""
    def __init__(self, in_channels, out_channels, use_bn=True):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels, use_bn=use_bn)
        self.down = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              stride=2, padding=1, bias=False)
        self.bn   = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.act  = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        feat = self.conv(x)       # full resolution features saved for skip
        down = self.act(self.bn(self.down(feat)))
        return down, feat         # return both downsampled and skip features


class DecoderBlock(nn.Module):
    """Upsample + concat skip + ConvBlock"""
    def __init__(self, in_channels, skip_channels, out_channels, dropout=0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels, 
                                     kernel_size=3, stride=2, padding=1)
        # Takes upsampled + skip connection channels
        self.conv = ConvBlock(in_channels + skip_channels, out_channels, 
                              use_bn=True, activation='relu')
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, skip):
        x = F.interpolate(self.up(x), size=skip.shape[2:])
        x = torch.cat([x, skip], dim=1)
        return self.dropout(self.conv(x))


class SoundAttacker(nn.Module):
    """
    Deep Pix2Pix UNet with:
    - Double conv blocks at each scale
    - Wide bottleneck (512 channels) with 3 conv layers
    - Dense skip connections (encoder features concatenated in decoder)
    - Dropout in first 3 decoder blocks
    Input/Output: [B, 2, 64, T]
    """
    def __init__(self, dropout=0.3):
        super().__init__()

        # --- ENCODER ---
        # No BN on first block (Pix2Pix convention)
        self.enc1 = EncoderBlock(2,   32,  use_bn=False)  # skip: [32,  32, T/2]
        self.enc2 = EncoderBlock(32,  64,  use_bn=True)   # skip: [64,  16, T/4]
        self.enc3 = EncoderBlock(64,  128, use_bn=True)   # skip: [128,  8, T/8]
        self.enc4 = EncoderBlock(128, 256, use_bn=True)   # skip: [256,  4, T/16]

        # --- BOTTLENECK ---
        # 3 conv layers at compressed representation, wide channels
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # [512, 2, T/32]
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),            # [512, 2, T/32]
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),            # [512, 2, T/32]
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # --- DECODER ---
        # in_channels = prev decoder out, skip_channels = matching encoder feat
        self.dec4 = DecoderBlock(512, 256, 256, dropout=dropout)   # up + enc4 skip
        self.dec3 = DecoderBlock(256, 128, 128, dropout=dropout)   # up + enc3 skip
        self.dec2 = DecoderBlock(128, 64,  64,  dropout=dropout)   # up + enc2 skip
        self.dec1 = DecoderBlock(64,  32,  32,  dropout=0.0)       # up + enc1 skip

        # --- OUTPUT ---
        # Final 1x1 conv to map to 2 channels, Tanh to bound output
        self.final = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, kernel_size=1),
            nn.Tanh()
        )

    def forward(self, x):
        # Encoder - save skip features at each scale
        d1, s1 = self.enc1(x)
        d2, s2 = self.enc2(d1)
        d3, s3 = self.enc3(d2)
        d4, s4 = self.enc4(d3)

        # Bottleneck
        b = self.bottleneck(d4)

        # Decoder - each block receives upsampled input + skip from encoder
        u4 = self.dec4(b,  s4)
        u3 = self.dec3(u4, s3)
        u2 = self.dec2(u3, s2)
        u1 = self.dec1(u2, s1)

        # Final output, resized back to input dimensions
        out = self.final(u1)
        return F.interpolate(out, size=x.shape[2:])

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

def training(model, full_dataset, batch_size=32, num_epochs=20,
             val_split=0.2, patience=3, log_dir="runs/exp1"):

    # ── Prepare data loaders ───────────────────────────────────────────────────
    dataset_size = len(full_dataset)
    val_size     = int(val_split * dataset_size)
    train_size   = dataset_size - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=10, persistent_workers=True, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=10, persistent_workers=True, pin_memory=True)

    # ── Loss, Optimizer, Scheduler ────────────────────────────────────────────
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3,
        steps_per_epoch=len(train_dl),
        epochs=num_epochs,
        anneal_strategy='linear'
    )

    # ── TensorBoard writer ─────────────────────────────────────────────────────
    writer = SummaryWriter(log_dir)

    # ── Early stopping vars ────────────────────────────────────────────────────
    best_val_acc = 0.0
    epochs_no_improve = 0

    device = next(model.parameters()).device

    # Precompute ImageNet stats tensors for normalization
    imagenet_mean = torch.tensor([0.485, 0.456], device=device).view(1, 2, 1, 1)
    imagenet_std  = torch.tensor([0.229, 0.224], device=device).view(1, 2, 1, 1)

    for epoch in range(1, num_epochs + 1):
        model.train()

        # Epoch-level progress bar
        epoch_bar = tqdm(train_dl, desc=f"Epoch {epoch}/{num_epochs}", unit="batch")

        running_loss = 0.0
        epoch_loss   = 0.0
        correct_preds = 0
        total_preds   = 0

        for inputs, labels in epoch_bar:
            inputs = inputs.to(device)
            labels = labels.to(device).unsqueeze(1).float()

            # normalize to ImageNet stats for pretrained ResNet backbone
            inputs = (inputs - imagenet_mean) / imagenet_std

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            epoch_loss   += loss.item()

            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct_preds += (preds == labels).sum().item()
            total_preds   += preds.size(0)

            # Update the bar’s postfix with live metrics (average over all seen samples)
            epoch_bar.set_postfix({
                "loss": f"{running_loss / total_preds:.4f}",
                "acc":  f"{correct_preds / total_preds:.4f}"
            })

        train_acc  = correct_preds / total_preds
        train_loss = epoch_loss / len(train_dl)

        # —— Validation ——
        model.eval()
        val_loss     = 0.0
        val_correct  = 0
        val_total    = 0

        with torch.no_grad():
            for inputs, labels in val_dl:
                inputs = inputs.to(device)
                labels = labels.to(device).unsqueeze(1).float()
                inputs = (inputs - imagenet_mean) / imagenet_std

                outputs = model(inputs)
                val_loss    += criterion(outputs, labels).item()
                preds        = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total   += preds.size(0)

        val_loss = val_loss / len(val_dl)
        val_acc  = val_correct / val_total

        # — Log to TensorBoard —
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Acc/train',  train_acc,  epoch)
        writer.add_scalar('Loss/val',   val_loss,   epoch)
        writer.add_scalar('Acc/val',    val_acc,    epoch)

        print(f"Epoch {epoch:02d}  Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}  "
              f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}")

        # —— Early Stopping & Checkpointing ——
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), f"best_model{epoch}.pth")
            print(f"→ New best model saved (Val Acc: {best_val_acc:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping after {epoch} epochs "
                      f"(no improvement in {patience} epochs).")
                break

    writer.close()
    print("Training complete. Best Val Acc: {:.4f}".format(best_val_acc))


def train_attacker_on_discriminator(attacker, discriminator, dataset,
                                     batch_size=32, num_epochs=50):
    device = next(discriminator.parameters()).device
    attacker.to(device)

    discriminator.train()
    for param in discriminator.parameters():
        param.requires_grad = False
    for m in discriminator.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

    train_dl = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                         num_workers=10, persistent_workers=True, pin_memory=True)

    optimizer_atk = torch.optim.Adam(
        attacker.parameters(), lr=2e-3, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer_atk, T_0=10, T_mult=2, eta_min=1e-5)

    imagenet_mean = torch.tensor([0.485, 0.456], device=device).view(1, 2, 1, 1)
    imagenet_std  = torch.tensor([0.229, 0.224], device=device).view(1, 2, 1, 1)

    best_loss = 99999.0

    for epoch in range(1, num_epochs + 1):
        attacker.train()
        epoch_loss = 0.0
        pbar = tqdm(train_dl, desc=f"Attacker Epoch {epoch}")

        for inputs, labels in pbar:
            inputs = inputs.to(device)

            optimizer_atk.zero_grad()

            # --- Progressive noise injection ---
            # Early epochs: add strong noise to discriminator input
            # making it easier to fool. Noise decays to 0 by epoch 20.
            # This is a form of curriculum learning.
            noise_std = max(0.0, 1.0 - epoch / 20.0)
            
            adv_inputs = attacker(inputs)
            adv_inputs_norm = (adv_inputs - imagenet_mean) / imagenet_std

            # Inject decaying noise into discriminator input only
            # Does not affect attacker output, only smooths discriminator
            if noise_std > 0:
                adv_inputs_norm = adv_inputs_norm + \
                    torch.randn_like(adv_inputs_norm) * noise_std

            outputs = discriminator(adv_inputs_norm)

            # Non-saturating GAN loss
            loss = F.softplus(-outputs).mean()

            loss.backward()

            # Log gradient norm
            total_norm = sum(
                p.grad.norm().item() ** 2 
                for p in attacker.parameters() 
                if p.grad is not None
            ) ** 0.5

            torch.nn.utils.clip_grad_norm_(attacker.parameters(), max_norm=1.0)
            optimizer_atk.step()

            epoch_loss += loss.item()

            with torch.no_grad():
                adv_prob = torch.sigmoid(outputs).mean().item()

            pbar.set_postfix({
                "loss":  f"{loss.item():.4f}",
                "adv_p": f"{adv_prob:.3f}",
                "grad":  f"{total_norm:.4f}",
                "noise": f"{noise_std:.3f}"
            })

        scheduler.step(epoch)
        avg_loss = epoch_loss / len(train_dl)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(attacker.state_dict(), f"best_att_model{epoch}.pth")
            print(f"→ New best model saved (Avg Loss: {best_loss:.4f})")

        print(f"Epoch {epoch} | Avg Loss: {avg_loss:.4f}")

    return attacker


if __name__ == '__main__':
    # Download latest version
    path = kagglehub.dataset_download("awsaf49/asvpoof-2019-dataset")

    print("Path to dataset files:", path)

    # Define paths and parameters
    DATASET_PATH_LA = os.path.join(path, "LA/LA/ASVspoof2019_LA_train/flac")
    LABEL_FILE_PATH_LA = os.path.join(path,"LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt")
    DATASET_PATH_PA = os.path.join(path,"PA/PA/ASVspoof2019_PA_train/flac")
    LABEL_FILE_PATH_PA = os.path.join(path, "PA/PA/ASVspoof2019_PA_cm_protocols/ASVspoof2019.PA.cm.train.trn.txt")


    datasets = [
        {'label_file': LABEL_FILE_PATH_LA, 'data_path': DATASET_PATH_LA},
        {'label_file': LABEL_FILE_PATH_PA, 'data_path': DATASET_PATH_PA}
    ]

    # Separate fake audio files into PA, LA and others
    fake_pa = []
    fake_la = []
    real = []

    def parse_labels_with_path(label_file_path, data_path):
        data = []
        with open(label_file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                file_name = parts[1]
                label = 1 if parts[-1] == "bonafide" else 0
                file_path = os.path.join(data_path, file_name + ".flac")
                data.append((file_path, label))
        return data

    for dataset in datasets:
        data = parse_labels_with_path(dataset['label_file'], dataset['data_path'])
        for file_path, label in data:
            if label == 1:
                real.append((file_path, label))
            else:
                if "PA" in dataset['label_file']:
                    fake_pa.append((file_path, label))
                elif "LA" in dataset['label_file']:
                    fake_la.append((file_path, label))

    # Shuffle the data
    np.random.shuffle(real)
    np.random.shuffle(fake_pa)
    np.random.shuffle(fake_la)

    np.random.seed(42)

    # Calculate how many fakes we need.(70% PA and 30% LA)
    n_real = len(real)
    n_fake_pa = int(n_real * 0.7)
    n_fake_la = n_real - n_fake_pa  # remaining 30%

    # Ensure we don't sample more than available. (Here is not the case)
    n_fake_pa = min(n_fake_pa, len(fake_pa))
    n_fake_la = min(n_fake_la, len(fake_la))

    # Random sampling.
    # Provides an array containing a specific number of samples from a given list of labelled audio files.
    fake_pa_sample = random.sample(fake_pa, k=n_fake_pa)
    fake_la_sample = random.sample(fake_la, k=n_fake_la)

    balanced_data = fake_pa_sample + fake_la_sample
    random.shuffle(balanced_data)

    # Separate paths and labels
    X_balanced = np.array([x[0] for x in balanced_data])
    y_balanced = np.array([x[1] for x in balanced_data])


    original_ds = SoundDS(X_balanced, y_balanced, mode="original")


    # Create the model and put it on the GPU if available
    myModel = CRNNWithAttn()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    myModel = myModel.to(device)
    # Check that it is on Cuda
    next(myModel.parameters()).device

    # training(myModel, original_ds, num_epochs = 100, batch_size=64, patience=10, log_dir="runs/crnn_attn_exp1")


    # 1. Load your best trained discriminator
    myModel.load_state_dict(torch.load("best_model10.pth"))
    myModel.eval() # Set the model to evaluation mode
    print(f"Model weights loaded from best_model1.pth")

    # 2. Initialize the U-Net attacker
    myAttacker = SoundAttacker().to(device)

    # 3. Train the attacker to fool your real model
    trained_attacker = train_attacker_on_discriminator(
        attacker=myAttacker,
        discriminator=myModel,
        dataset=original_ds,
        num_epochs=100)
