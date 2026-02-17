# Kaggle Training Guide - Chest X-Ray Pneumonia Detection

Train your chest X-ray pneumonia detection model on Kaggle's free GPU!

## Why Kaggle?
- Free 30 hours/week of GPU (P100 or T4)
- Chest X-ray dataset already available on Kaggle (no upload needed!)
- Pre-installed ML libraries (PyTorch, scikit-learn, etc.)
- No credit card needed

---

## Quick Start (6 Steps)

### Step 1: Create a Kaggle Account
1. Go to https://www.kaggle.com
2. Sign up (free) if you don't have an account

### Step 2: Upload Your Project Code as a Dataset
1. Go to https://www.kaggle.com/datasets
2. Click **+ New Dataset**
3. Name it: `medical-image-project`
4. Upload the file: `project_for_colab.zip` (from `colab_chest_xray/` folder)
5. Click **Create** (keep it Private)

### Step 3: Create a New Notebook
1. Go to https://www.kaggle.com/code
2. Click **+ New Notebook**
3. Or: Upload `KAGGLE_TRAINING_CHEST_XRAY.ipynb` directly

### Step 4: Add the Datasets
In the notebook, click **+ Add Data** (right sidebar):

**Dataset 1 - Chest X-rays (public):**
1. Search: `chest-xray-pneumonia`
2. Select the one by **Paul Mooney** (~2GB)
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
2. Wait ~1 hour for training
3. Download results from the Output tab

---

## How Kaggle Differs from Colab

| Feature | Colab | Kaggle |
|---------|-------|--------|
| GPU credits | Limited free | 30 hrs/week free |
| Dataset location | Google Drive | /kaggle/input/ |
| File upload | Manual upload prompt | Add as Dataset |
| Output download | Auto-download | Output tab |
| Libraries | Need to install | Pre-installed |
| Session limit | ~12 hours | ~12 hours |

---

## Dataset Structure on Kaggle

When you add the `chest-xray-pneumonia` dataset, it appears at:
```
/kaggle/input/chest-xray-pneumonia/
  chest_xray/
    chest_xray/
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

The notebook automatically detects this path structure.

---

## Expected Results

**Validation Accuracy**: 90-95%
**Training Time**: ~45-70 minutes on P100, ~60-90 minutes on T4
**Model Size**: ~45MB

---

## Downloading Results

After training completes:

### Option 1: Save & Run All (Recommended)
1. Click **Save Version** (top right)
2. Select **Save & Run All (Commit)**
3. Wait for the run to complete
4. Go to your notebook page -> **Output** tab
5. Download `best_model.pth` and `training_history.json`

### Option 2: Quick Save
1. After Cell 9 runs, files are in `/kaggle/working/`
2. Click the **Output** section in the right sidebar
3. Download files directly

---

## Troubleshooting

### "No GPU available"
- Check Settings -> Accelerator -> GPU is selected
- You may have exhausted your weekly 30-hour quota
- Wait for next week or use CPU (slower but works)

### "Dataset not found"
- Make sure you added BOTH datasets:
  1. `chest-xray-pneumonia` (public, by Paul Mooney)
  2. `medical-image-project` (your private dataset with project code)
- Check the right sidebar under **Data** to see attached datasets

### "Out of memory"
- In Cell 3, change `"batch_size": 32` to `"batch_size": 16`
- Restart the session and run again

### "Module not found"
- Make sure Cell 4 ran successfully
- Check that the project zip was extracted properly in Cell 3

### "Permission denied" when writing files
- Kaggle input dirs are read-only
- The notebook copies files to `/kaggle/working/` which is writable

---

## Pro Tips

1. **Use "Save & Run All"** to create a committed version - this runs the entire notebook from scratch and makes the output downloadable

2. **Kaggle sessions last ~12 hours** - plenty of time for training

3. **Check GPU quota**: Go to Settings to see remaining GPU hours

4. **Internet access**: If you need to download packages, enable Internet in Settings

5. **Test with fewer epochs first**: In Cell 3, change `"epochs": 50` to `"epochs": 5` to verify everything works, then change back

---

## After Training: Bring Model Home

```powershell
# After downloading from Kaggle Output tab:
# Move to your project:
move best_model.pth "E:\MINE\Major Project\Major Project\medical-image-classification\models\checkpoints\"
# Rename it:
ren "E:\MINE\Major Project\Major Project\medical-image-classification\models\checkpoints\best_model.pth" chest_xray_resnet18.pth
```

---

## Checklist

Before starting:
- [ ] Kaggle account created
- [ ] Project zip uploaded as Kaggle dataset
- [ ] Notebook created/uploaded on Kaggle
- [ ] Chest X-ray dataset added (search: chest-xray-pneumonia)
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
- [ ] Verified accuracy >= 90%

---

**Ready? Go to Kaggle and start training!**
