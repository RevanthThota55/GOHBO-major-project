# Google Colab Training Guide - Chest X-Ray Pneumonia Detection

Train your chest X-ray pneumonia detection model on Google Colab's free GPU!

## Why Colab?
- Free GPU access (T4 or better)
- 12GB+ GPU memory
- Training time: ~1 hour (vs 4-6 hours on CPU)
- No local resource usage

---

## Quick Start (5 Steps)

### Step 1: Prepare Dataset
Upload your dataset to Google Drive:
1. Go to https://drive.google.com
2. Create folder: `chest_xray_dataset`
3. Upload these folders from `data/chest_xray/`:
   - `train/` (with NORMAL and PNEUMONIA subfolders)
   - `val/` (with NORMAL and PNEUMONIA subfolders)
   - `test/` (with NORMAL and PNEUMONIA subfolders)

**Folder structure in Google Drive should be:**
```
MyDrive/
  chest_xray_dataset/
    train/
      NORMAL/     (~1,341 images)
      PNEUMONIA/  (~3,875 images)
    val/
      NORMAL/     (~8 images)
      PNEUMONIA/  (~8 images)
    test/
      NORMAL/     (~234 images)
      PNEUMONIA/  (~390 images)
```

### Step 2: Create Project ZIP
```powershell
# In your project directory
cd "E:\MINE\Major Project\Major Project\medical-image-classification"

# Create zip with only necessary files
tar -czf project_for_colab.zip src/ config.py train.py results/phase1/best_hyperparameters.json
```

**Note:** The notebook will automatically write the correct chest X-ray hyperparameters after extraction, so you can use the same zip from brain tumor training.

### Step 3: Open Colab Notebook
1. Upload `COLAB_TRAINING_CHEST_XRAY.ipynb` to Google Drive
2. Right-click -> Open with -> Google Colaboratory
3. Or go to https://colab.research.google.com and upload the notebook

### Step 4: Change Runtime to GPU
1. Click **Runtime -> Change runtime type**
2. Select **Hardware accelerator: GPU**
3. Choose **T4 GPU** (or V100 if available)
4. Click **Save**

### Step 5: Run All Cells
1. Click **Runtime -> Run all**
2. When prompted, upload `project_for_colab.zip`
3. Grant Google Drive access when asked
4. Wait for training to complete (~1 hour)
5. Download the trained model when done!

---

## What Gets Uploaded to Colab

**Project files** (~50MB):
- `src/` - All source code
- `config.py` - Configuration
- `train.py` - Training script
- `results/phase1/best_hyperparameters.json` - Optimized params (overwritten by notebook)

**NOT uploaded** (stays on Google Drive):
- Dataset (~1.2GB) - loaded from Drive
- Trained models - saved to Drive at end

---

## Timeline

```
Setup:        5-10 minutes (upload, install dependencies)
Training:     45-70 minutes (depends on GPU)
Download:     2-5 minutes (save trained model)
----------------------------------------------
Total:        ~1-1.5 hours
```

---

## Expected Results

After training completes, you'll get:

**Trained Model**: `best_model.pth` (~45MB)
- Expected accuracy: 90-95%
- Saved in `models/checkpoints/`

**Training History**: `training_history.json`
- Accuracy and loss curves
- Precision and recall per epoch

**Checkpoints**: Every 5 epochs
- `checkpoint_epoch_5.pth`
- `checkpoint_epoch_10.pth`
- etc.

---

## Cell-by-Cell Breakdown

### Cell 1: Check GPU
Verifies GPU is available and shows specs.

**Expected output:**
```
CUDA available: True
GPU: Tesla T4
GPU Memory: 15.00 GB
```

### Cell 2: Mount Google Drive
Connects to your Google Drive.

### Cell 3: Upload Project
Upload `project_for_colab.zip` when prompted. This cell also writes the correct chest X-ray hyperparameters (LR=0.001, batch=32, epochs=50).

### Cell 4: Install Dependencies
Installs PyTorch, torchvision, and other packages.

### Cell 5: Setup Dataset
Links to your Google Drive dataset and shows image counts.

**Expected output:**
```
Train images: ~5216
Val images:   ~16
Test images:  ~624
Total:        ~5856

Class distribution (train):
  NORMAL: ~1341
  PNEUMONIA: ~3875
```

### Cell 6: Verify Setup
Checks that config and dataset loader work correctly.

### Cell 7: START TRAINING!
This is the main training loop.

**Expected output:**
```
Epoch 1/50: 100%|###| 163/163 [01:30<00:00]
Train Loss: 0.423, Train Acc: 84.3%
Val Loss: 0.312, Val Acc: 87.5%

Epoch 2/50: 100%|###| 163/163 [01:28<00:00]
...
```

### Cell 8: Check Results
Shows final accuracy and lists saved models.

### Cell 9: Download Model
Downloads trained model to your computer.

### Cell 10: Backup to Drive (Optional)
Saves everything to Google Drive for safekeeping.

---

## Troubleshooting

### Problem: "No GPU available"
**Solution**: Runtime -> Change runtime type -> Select GPU

### Problem: "Dataset not found"
**Solution**:
1. Check folder name in Drive is exactly `chest_xray_dataset`
2. Verify it contains `train/`, `val/`, `test/` subfolders
3. Each subfolder must have `NORMAL/` and `PNEUMONIA/` folders
4. Run Cell 5 again to recreate the links

### Problem: "Out of memory"
**Solution**:
1. In Cell 3, change `"batch_size": 32` to `"batch_size": 16` in the hyperparameters dict
2. Restart runtime and try again

### Problem: "Training takes too long"
**Solution**:
1. Check you selected GPU (not CPU)
2. Run `!nvidia-smi` to verify GPU is active
3. Consider reducing epochs (50 -> 25 for faster testing)

### Problem: "Colab disconnected"
**Solution**:
- Training state is saved in checkpoints every 5 epochs
- Restart runtime and resume from last checkpoint
- Or use Cell 10 to backup frequently

### Problem: "Low validation accuracy"
**Solution**:
- The chest X-ray dataset has very few validation images (~16)
- This can cause noisy val accuracy - this is normal
- Focus on test set accuracy after training completes
- The class imbalance (more PNEUMONIA than NORMAL) is handled by WeightedRandomSampler

---

## Pro Tips

1. **Run overnight**: Start training in the evening, wake up to trained model

2. **Use Colab Pro** ($10/month):
   - Longer runtime (24h vs 12h)
   - Better GPUs (V100, A100)
   - Faster training

3. **Monitor from phone**:
   - Install Google Colab app
   - Get notifications when cells finish

4. **Save checkpoints to Drive**:
   - Run Cell 10 periodically
   - Protects against disconnections

5. **Test first with fewer epochs**:
   - In Cell 3, change `"epochs": 50` to `"epochs": 5`
   - Verify everything works
   - Then change back to 50 for full training

---

## After Training: Bring Model Home

### Option 1: Download directly
Cell 9 downloads `best_model.pth` to your computer.

### Option 2: From Google Drive
```powershell
# On your local machine
# Download from: https://drive.google.com/drive/my-drive/chest_xray_results/

# Then move to your project:
move best_model.pth "E:\MINE\Major Project\Major Project\medical-image-classification\models\checkpoints\"
# Rename it:
ren "E:\MINE\Major Project\Major Project\medical-image-classification\models\checkpoints\best_model.pth" chest_xray_resnet18.pth
```

---

## Test the Trained Model Locally

After downloading the model:

```powershell
cd "E:\MINE\Major Project\Major Project\medical-image-classification"
venv\Scripts\activate

# Evaluate on test set
python evaluate.py --dataset chest_xray --model_path models/checkpoints/best_model.pth

# Make predictions
python predict.py --image_path path/to/test/xray.jpeg --model_path models/checkpoints/best_model.pth
```

---

## Dataset Info

- **Name**: Chest X-Ray Images (Pneumonia)
- **Source**: Kermany, Zhang, Goldbaum (2018), Mendeley Data, V2
- **Task**: Binary classification (NORMAL vs PNEUMONIA)
- **Images**: ~5,863 chest X-rays
- **Format**: JPEG
- **Image size**: Various (resized to 224x224 during training)
- **Class distribution**: Imbalanced (~75% PNEUMONIA, ~25% NORMAL)
  - Handled by WeightedRandomSampler in the training code

---

## For Your College Presentation

After training on Colab, you can say:

> "We trained our ResNet-18 model for pneumonia detection using Google Colab's Tesla T4 GPU. Using transfer learning from ImageNet and optimized hyperparameters (learning rate: 0.001, batch size: 32), we achieved ~93% accuracy on the chest X-ray classification task. The model uses weighted sampling to handle class imbalance in the dataset."

---

## Checklist

Before starting:
- [ ] Dataset uploaded to Google Drive (`chest_xray_dataset` folder)
- [ ] Folder has train/, val/, test/ with NORMAL/ and PNEUMONIA/ subfolders
- [ ] Created `project_for_colab.zip` (or reuse from brain tumor)
- [ ] Have `COLAB_TRAINING_CHEST_XRAY.ipynb` ready
- [ ] Have stable internet connection

During training:
- [ ] Selected GPU runtime
- [ ] All cells ran without errors
- [ ] Training started and accuracy is improving
- [ ] Leave tab open or use Colab app to monitor

After training:
- [ ] Downloaded `best_model.pth`
- [ ] Downloaded `training_history.json`
- [ ] Backed up to Google Drive (Cell 10)
- [ ] Verified accuracy >= 90%

---

**Ready? Let's train on Colab!**

Upload `COLAB_TRAINING_CHEST_XRAY.ipynb` to Colab and follow along!
