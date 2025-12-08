# Deployment Guide: Running on Server or Google Colab

## 🎯 Overview

This guide shows you how to run the DiffusionModel_NILM project on:
1. Remote servers (with GPU)
2. Google Colab (free GPU)

---

## 📦 Method 1: Using Git (Recommended)

### Step 1: Prepare Your Local Repository

```bash
# Navigate to project directory
cd C:\Users\Raymond Tie\Desktop\DiffusionModel_NILM

# Initialize Git (if not already done)
git init

# Create .gitignore
echo ".Checkpoints/" >> .gitignore
echo "OUTPUT/" >> .gitignore
echo "*.pyc" >> .gitignore
echo "__pycache__/" >> .gitignore
echo ".venv/" >> .gitignore

# Add all files
git add .
git commit -m "Initial commit: DiffusionModel_NILM project"
```

### Step 2: Push to GitHub

```bash
# Create a new repository on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/DiffusionModel_NILM.git
git branch -M main
git push -u origin main
```

### Step 3: Clone on Server/Colab

```bash
# On remote server or Colab
git clone https://github.com/YOUR_USERNAME/DiffusionModel_NILM.git
cd DiffusionModel_NILM

# Install dependencies
pip install -r requirements.txt

# Start training
python main.py --config Config/kettle.yaml --train
```

---

## 🌐 Method 2: Google Colab (Free GPU)

### Option A: Using the Notebook

1. Upload `setup_colab.ipynb` to Google Colab
2. Change runtime to GPU: `Runtime` → `Change runtime type` → `GPU` → `Save`
3. Run all cells

### Option B: Manual Setup in Colab

```python
# 1. Check GPU
!nvidia-smi

# 2. Clone your repo
!git clone https://github.com/YOUR_USERNAME/DiffusionModel_NILM.git
%cd DiffusionModel_NILM

# 3. Install dependencies
!pip install -r requirements.txt

# 4. Start training
!python main.py --config Config/kettle.yaml --train

# 5. Generate samples (after training)
!python main.py --config Config/kettle.yaml --sample 2000

# 6. Download results
from google.colab import files
!zip -r results.zip .Checkpoints OUTPUT
files.download('results.zip')
```

---

## 🖥️ Method 3: Remote Server (SSH)

### Step 1: Upload Project

```bash
# Option 1: Using Git (recommended)
ssh user@server
git clone https://github.com/YOUR_USERNAME/DiffusionModel_NILM.git

# Option 2: Using SCP
scp -r C:\Users\Raymond Tie\Desktop\DiffusionModel_NILM user@server:/path/to/destination/
```

### Step 2: Setup Environment

```bash
# SSH into server
ssh user@server
cd DiffusionModel_NILM

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate     # On Windows

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Run Training

```bash
# Run in background with nohup
nohup python main.py --config Config/kettle.yaml --train > training.log 2>&1 &

# Or use screen/tmux
screen -S diffusion_training
python main.py --config Config/kettle.yaml --train
# Press Ctrl+A then D to detach
```

### Step 4: Monitor Progress

```bash
# Check log
tail -f training.log

# Or reattach to screen
screen -r diffusion_training
```

---

## 📊 Best Practices

### 1. Version Control
- ✅ Use Git for code
- ✅ Add `.gitignore` for large files
- ✅ Commit frequently

### 2. Configuration Management
- ✅ Keep configs in separate files (`Config/*.yaml`)
- ✅ Don't hardcode paths
- ✅ Use relative paths

### 3. Data Management
- ✅ Upload data separately (don't commit to Git)
- ✅ Use cloud storage (Google Drive, S3) for large datasets
- ✅ Download data on server/Colab when needed

### 4. Checkpoint Management
- ✅ Save checkpoints frequently (`save_cycle: 500`)
- ✅ Download checkpoints periodically
- ✅ Use cloud storage for backup

---

## 🔧 Troubleshooting

### Issue: Out of Memory on Colab

```yaml
# Reduce batch size in Config/kettle.yaml
dataloader:
  batch_size: 32  # Reduce from 64
```

### Issue: Colab Disconnects

```python
# Add this at the start of your notebook
from google.colab import drive
drive.mount('/content/drive')

# Save checkpoints to Google Drive
!mkdir -p /content/drive/MyDrive/DiffusionModel_Checkpoints
!ln -s /content/drive/MyDrive/DiffusionModel_Checkpoints .Checkpoints
```

### Issue: Training Too Slow

```yaml
# Reduce epochs for testing
solver:
  max_epochs: 2000  # Instead of 20000
```

---

## 📥 Downloading Results

### From Colab:
```python
from google.colab import files
files.download('.Checkpoints/Checkpoints_kettle/model-2000.pt')
```

### From Server:
```bash
# Download to local machine
scp user@server:/path/to/DiffusionModel_NILM/.Checkpoints/Checkpoints_kettle/model-2000.pt ./
```

---

## 🎯 Quick Start Checklist

- [ ] Push code to GitHub
- [ ] Open Google Colab
- [ ] Change runtime to GPU
- [ ] Clone repository
- [ ] Install requirements
- [ ] Upload data (if needed)
- [ ] Modify config (reduce epochs for testing)
- [ ] Start training
- [ ] Monitor progress
- [ ] Download checkpoints
- [ ] Generate samples
- [ ] Download results
