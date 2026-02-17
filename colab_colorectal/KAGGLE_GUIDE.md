# Kaggle Training Guide - Colorectal Histology Classification

Train the colorectal tissue classification model (8 classes) on Kaggle's free GPU!

## Dataset Info
- **Task**: 8-class tissue type classification from histopathology images
- **Classes**: TUMOR, STROMA, COMPLEX, LYMPHO, DEBRIS, MUCOSA, ADIPOSE, EMPTY
- **Images**: ~5,000 histology images (150x150px, resized to 224x224 for training)
- **Format**: .tif files
- **Expected Accuracy**: 90-95%
- **Training Time**: ~1.5-2 hours on T4 GPU (70 epochs)

---

## Quick Start (6 Steps)

### Step 1: Create a Kaggle Account
1. Go to https://www.kaggle.com
2. Sign up (free) if you don't have an account

### Step 2: Upload Your Project Code as a Dataset
1. Go to https://www.kaggle.com/datasets
2. Click **+ New Dataset**
3. Name it: `medical-image-project`
4. Upload the file: `project_for_colab.zip` (from `colab_colorectal/` folder)
5. Click **Create** (keep it Private)

### Step 3: Create a New Notebook
1. Go to https://www.kaggle.com/code
2. Click **+ New Notebook**
3. Or: Upload `KAGGLE_TRAINING_COLORECTAL.ipynb` directly

### Step 4: Add the Datasets
In the notebook, click **+ Add Data** (right sidebar):

**Dataset 1 - Colorectal Histology (public):**
1. Search: `colorectal-histology-mnist`
2. Select the one by **Kevin Mader** (~300MB)
3. Click **Add**

**Dataset 2 - Your project code:**
1. Search: `medical-image-project` (your private dataset from Step 2)
2. Click **Add**

### Step 5: Enable GPU
1. Click **Settings** (right sidebar)
2. Under **Accelerator**, select **GPU T4 x2** or **GPU P100**
3. The session will restart with GPU enabled

### Step 6: Run All Cells
1. Click **Run All** (or Shift+Enter each cell)
2. Wait ~1.5-2 hours for training
3. Download results from the Output tab

---

## What the Notebook Does

The colorectal dataset on Kaggle does NOT come pre-split into train/val/test.
The notebook automatically:

1. **Finds** the dataset in `/kaggle/input/`
2. **Splits** it into 80% train / 10% val / 10% test (reproducible, seed=42)
3. **Renames** folders from `01_TUMOR` -> `TUMOR`, `02_STROMA` -> `STROMA`, etc.
4. **Trains** for 70 epochs with GOHBO-optimized hyperparameters
5. **Saves** the best model and training history for download

---

## Dataset Structure on Kaggle

When you add the `colorectal-histology-mnist` dataset, it appears at:
```
/kaggle/input/colorectal-histology-mnist/
  Kather_texture_2016_image_tiles_5000/
    Kather_texture_2016_image_tiles_5000/
      01_TUMOR/     (~625 images)
      02_STROMA/    (~625 images)
      03_COMPLEX/   (~625 images)
      04_LYMPHO/    (~625 images)
      05_DEBRIS/    (~625 images)
      06_MUCOSA/    (~625 images)
      07_ADIPOSE/   (~625 images)
      08_EMPTY/     (~625 images)
```

The notebook detects this automatically and splits the data.

---

## Expected Results

**Validation Accuracy**: 90-95%
**Training Time**: ~1.5-2 hours on T4 GPU
**Model Size**: ~45MB

---

## Downloading Results

After training completes:

### Option 1: Save & Run All (Recommended)
1. Click **Save Version** (top right)
2. Select **Save & Run All (Commit)**
3. Wait for the run to complete
4. Go to your notebook page -> **Output** tab
5. Download `best_model.pth`, `training_history.json`, and `training_results_colorectal.json`

### Option 2: Quick Save
1. After the last cell runs, files are in `/kaggle/working/`
2. Click the **Output** section in the right sidebar
3. Download files directly

---

## Troubleshooting

### "No GPU available"
- Check Settings -> Accelerator -> GPU is selected
- You may have exhausted your weekly 30-hour quota
- Wait for next week or use CPU (much slower)

### "Dataset not found"
- Make sure you added BOTH datasets:
  1. `colorectal-histology-mnist` (public, by Kevin Mader)
  2. `medical-image-project` (your private dataset with project code)
- Check the right sidebar under **Data** to see attached datasets

### "Out of memory"
- Change batch_size from 32 to 16 in Cell 2
- Restart the session and run again

### "Module not found"
- Make sure Cell 3 ran successfully (installs albumentations, scikit-image)
- Check that the project zip was extracted properly in Cell 2

---

## After Training: Bring Model Home

```powershell
# After downloading from Kaggle Output tab:
# Move to your project:
move best_model.pth "E:\MINE\Major Project\Major Project\medical-image-classification\models\checkpoints\"
# Rename it:
ren "E:\MINE\Major Project\Major Project\medical-image-classification\models\checkpoints\best_model.pth" colorectal_resnet18.pth
```

---

## Checklist

Before starting:
- [ ] Kaggle account created
- [ ] Project zip uploaded as Kaggle dataset
- [ ] Notebook created/uploaded on Kaggle
- [ ] Colorectal histology dataset added (search: colorectal-histology-mnist)
- [ ] Project code dataset added (your private upload)
- [ ] GPU enabled in Settings

During training:
- [ ] All cells ran without errors
- [ ] Training started and accuracy is improving
- [ ] GPU is being used (check Cell 1 output)

After training:
- [ ] Clicked "Save Version" -> "Save & Run All"
- [ ] Downloaded best_model.pth from Output tab
- [ ] Downloaded training_history.json
- [ ] Downloaded training_results_colorectal.json
- [ ] Verified accuracy >= 90%

---

**Ready? Go to Kaggle and start training!**
