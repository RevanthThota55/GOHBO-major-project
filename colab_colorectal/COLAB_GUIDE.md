# 🚀 Colorectal Tissue Classification on Google Colab

Train your histology tissue classification model on Google Colab's free GPU!

## Dataset Info
- **Task**: 8-class tissue classification  
- **Classes**: TUMOR, STROMA, COMPLEX, LYMPHO, DEBRIS, MUCOSA, ADIPOSE, EMPTY
- **Images**: ~5,000 histology images (150x150px)
- **Expected Accuracy**: 90-95%
- **Training Time**: ~1.5-2 hours on T4 GPU

---

## 📋 Quick Start (5 Steps)

### Step 1: Prepare Dataset in Google Drive
1. Go to https://drive.google.com
2. Create folder: `colorectal_dataset`
3. Upload the **inner** folder from `data/colorectal/Kather_texture_2016_image_tiles_5000/Kather_texture_2016_image_tiles_5000/`
   
   **Important**: Upload the folder that contains the 8 class folders:
   - 01_TUMOR/
   - 02_STROMA/
   - 03_COMPLEX/
   - 04_LYMPHO/
   - 05_DEBRIS/
   - 06_MUCOSA/
   - 07_ADIPOSE/
   - 08_EMPTY/

### Step 2: Upload Notebook to Google Drive
1. Upload `COLAB_TRAINING_COLORECTAL.ipynb` to Google Drive
2. Right-click → Open with → Google Colaboratory

### Step 3: Change Runtime to GPU
1. Click **Runtime → Change runtime type**
2. Select **Hardware accelerator: GPU**
3. Choose **T4 GPU**
4. Click **Save**

### Step 4: Run All Cells
1. Click **Runtime → Run all**
2. When Cell 3 prompts, upload `project_for_colab.zip`
3. Grant Google Drive access when asked
4. Wait for training (~1.5-2 hours)

### Step 5: Download Results
- `best_model.pth` will auto-download
- Also backed up to Google Drive: `MyDrive/colorectal_results/`

---

## 📊 Expected Results

**Validation Accuracy**: 90-95%
**Training Time**: ~80-120 minutes (8 classes)
**Model Size**: ~45MB

---

## 🔧 Troubleshooting

### "Dataset not found"
- Check folder structure in Drive:
  ```
  colorectal_dataset/
    Kather_texture_2016_image_tiles_5000/
      01_TUMOR/
      02_STROMA/
      ...
  ```
- The dataset has nested folders - make sure you uploaded the right level

### "Out of memory"
- Reduce batch size in hyperparameters (32 → 16)
- Restart runtime and try again

### Images not loading
- Colorectal images are `.tif` format (not `.jpg`)
- The dataset loader handles this automatically

---

## 📖 Dataset Citation

```
Kather, J. N., Weis, C.-A., Bianconi, F., Melchers, S. M., Schad, L. R., 
Gaiser, T., Marx, A., Zöllner, F. G. (2016). 
Multi-class texture analysis in colorectal cancer histology. 
Scientific Reports, 6, 27988.
```

**Ready? Upload the notebook and start training!** 🚀
