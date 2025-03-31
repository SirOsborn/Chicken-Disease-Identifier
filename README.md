# Chicken Disease Identification by Fecal Images

This project develops a machine learning model to classify chicken health status (diseases or healthy) using fecal images, as part of our final class project. This README provides instructions to set up your coding environment consistently across the team.

## Prerequisites
- **Operating System:** Windows 11 (adaptable to macOS/Linux with minor changes).
- **Git:** Installed for version control ([download here](https://git-scm.com/download/win)).
- **Python:** We’ll use version 3.9.13 (TensorFlow 2.10.0 compatible) for this project. You can keep your global Python version as the latest (e.g., 3.11.9 or 3.13.2) and use `pyenv-win` to set 3.9.13 locally for this project.
- **GPU (Optional):** NVIDIA GPU (e.g., RTX 4060) for acceleration—requires CUDA setup.

## Setup Instructions

### 1. Clone the Repository
1. Open Command Prompt (Win + S > “cmd” > Enter).
2. Navigate to your desired project directory:
   ```
   cd "C:\Users\YourUsername\YourProjectsFolder"
   ```
3. Clone the repo:
   ```
   git clone https://github.com/yourusername/diseaseIdentifyByChickShit.git
   cd diseaseIdentifyByChickShit
   ```

### 2. Install Python 3.9.13
We use Python 3.9.13 to ensure compatibility with TensorFlow 2.10.0 (the last version with native Windows GPU support). We recommend using `pyenv-win` to manage project-specific Python versions without affecting your global Python version.

- **Option A: Use `pyenv-win` (Recommended)**
  1. Install `pyenv-win`:
     ```
     pip install pyenv-win
     ```
  2. Add to PATH (via “Edit the system environment variables” > “User variables” > “Path”):
     ```
     C:\Users\YourUsername\.pyenv\pyenv-win\bin
     C:\Users\YourUsername\.pyenv\pyenv-win\shims
     ```
     **Important:** Ensure `C:\Users\YourUsername\.pyenv\pyenv-win\shims` is at the top of your PATH, above any other Python paths (e.g., `C:\Program Files\Python313`), to ensure `pyenv` takes precedence.
  3. Install Python 3.9.13:
     ```
     pyenv install 3.9.13
     ```
  4. Set the local version for this project (this won’t affect your global Python version):
     ```
     cd C:\Users\YourUsername\YourProjectsFolder\diseaseIdentifyByChickShit
     pyenv local 3.9.13
     ```
  5. Verify:
     ```
     python --version
     ```
     Should output `Python 3.9.13` in the project directory. Outside this directory, your global Python version (e.g., 3.11.9 or 3.13.2) will remain active.

- **Option B: Manual Install (Alternative)**
  1. Download Python 3.9.13 from [python.org/downloads](https://www.python.org/downloads/) (scroll to “Looking for a specific release?”).
  2. Run the installer:
     - Check “Add Python 3.9 to PATH”.
     - Install to default location (e.g., `C:\Users\YourUsername\AppData\Local\Programs\Python\Python39`).
  3. Verify:
     ```
     python --version
     ```
     Should output `Python 3.9.13`. If not, use the full path: `C:\Users\YourUsername\AppData\Local\Programs\Python\Python39\python.exe --version`.

### 3. Create and Activate Virtual Environment
1. Create a virtual environment named `chick-env`:
   - If Python 3.9.13 is set via `pyenv`:
     ```
     python -m venv chick-env
     ```
   - If using the manual install with full path:
     ```
     C:\Users\YourUsername\AppData\Local\Programs\Python\Python39\python.exe -m venv chick-env
     ```
2. Activate it:
   ```
   chick-env\Scripts\activate
   ```
   **Note:** If your project path contains spaces (e.g., `C:\Users\YourUsername\DS Projects\diseaseIdentifyByChickShit`), use quotes:
   ```
   "C:\Users\YourUsername\DS Projects\diseaseIdentifyByChickShit\chick-env\Scripts\activate"
   ```
   Your prompt should show `(chick-env)`.
3. Verify Python version:
   ```
   python --version
   ```
   Should be `Python 3.9.13`.

### 4. Install Dependencies
1. Install all required libraries from `requirements.txt`:
   ```
   pip install -r requirements.txt
   ```
   **Important:** For Windows users with GPU support, we use TensorFlow 2.10.0 (the last version with native Windows GPU support). TensorFlow 2.10.0 requires NumPy 1.x (e.g., 1.21.6), as it is incompatible with NumPy 2.x. First, ensure the correct NumPy version:
   ```
   pip install numpy==1.21.6
   ```
   Then install TensorFlow:
   ```
   pip uninstall tensorflow -y
   pip install tensorflow==2.10.0
   ```
   This includes TensorFlow 2.10.0, NumPy 1.21.6, Pandas, Scikit-learn, OpenCV, Matplotlib, Seaborn, and Pillow with GPU support.
2. Verify TensorFlow:
   ```
   python -c "import os; os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'; import tensorflow as tf; print(tf.__version__)"
   ```
   Should output `2.10.0`.
3. Verify GPU (if applicable):
   ```
   python -c "import os; os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'; import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
   ```
   Should list your GPU (e.g., `[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]`).

### 5. (Optional) Set Up CUDA for GPU Acceleration
If you have an NVIDIA GPU (e.g., RTX 4060), set up CUDA to accelerate training. TensorFlow 2.10.0 requires CUDA 11.2 and cuDNN 8.1:
1. **Install NVIDIA Driver:**
   - Download the latest driver for your GPU from [nvidia.com/drivers](https://www.nvidia.com/Download/index.aspx) (e.g., RTX 4060, Windows 11, version 551.23 or higher).
   - Install and reboot.
2. **Install CUDA Toolkit 11.2:**
   - Download from [developer.nvidia.com/cuda-11-2-0-download-archive](https://developer.nvidia.com/cuda-11-2-0-download-archive) (Windows > x86_64 > 11 > exe (local)).
   - Run installer (Express Installation).
   - Verify: `nvcc --version` (should show CUDA 11.2).
3. **Install cuDNN 8.1:**
   - Download from [developer.nvidia.com/cudnn](https://developer.nvidia.com/cudnn) (requires NVIDIA Developer account).
   - Select: cuDNN 8.1.x for CUDA 11.x > Windows.
   - Extract ZIP and copy the **contents** (not the folders themselves):
     - Files in `bin\` (e.g., `cudnn64_8.dll`) to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin`
     - Files in `include\` (e.g., `cudnn.h`) to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\include`
     - Files in `lib\x64\` (e.g., `cudnn.lib`) to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\lib\x64`
4. **Set Environment Variables:**
   - Add to “System variables” > “Path”:
     ```
     C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin
     C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\libnvvp
     ```
   - Restart Command Prompt.
5. **Verify GPU Usage:**
   ```
   python -c "import os; os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'; import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
   ```
   Should list your GPU (e.g., `PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')`).

### 6. Recommended Coding Environment
We suggest **VS Code with Jupyter** for a unified workflow:
1. Install [VS Code](https://code.visualstudio.com/).
2. Open the project:
   ```
   cd "C:\Users\YourUsername\YourProjectsFolder\diseaseIdentifyByChickShit"
   code .
   ```
3. Install extensions:
   - “Python” (Microsoft).
   - “Jupyter” (Microsoft).
4. Select interpreter: Ctrl+Shift+P > “Python: Select Interpreter” > choose `chick-env\Scripts\python.exe`.
5. Use `.ipynb` files for EDA and `.py` files for scripts.

### Troubleshooting
- **Wrong Python Version:** If `python --version` doesn’t show `Python 3.9.13` in the project directory, ensure `C:\Users\YourUsername\.pyenv\pyenv-win\shims` is at the top of your PATH, above other Python paths (e.g., `C:\Program Files\Python313`). Restart Command Prompt after adjusting PATH.
- **NumPy Compatibility Issues:** If you get errors related to NumPy 2.x, ensure you’ve installed `numpy==1.21.6` before installing TensorFlow 2.10.0. You can also try `numpy==1.19.2` if issues persist.
- **TensorFlow Installation Errors:** If `pip install tensorflow==2.10.0` fails, ensure you’re using Python 3.9.13 and that CUDA 11.2 and cuDNN 8.1 are installed correctly. You can also try a different PyPI mirror:
  ```
  pip install tensorflow==2.10.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
  ```
- **GPU Not Detected:** Reinstall CUDA/cuDNN, ensure driver is updated, and check environment variables.
- **Need Help?** Ask in our team chat or check TensorFlow’s [install guide](https://www.tensorflow.org/install).

## Project Structure
Ensure your local setup matches this structure:
```
diseaseIdentifyByChickShit/
├── chick-env/              # Virtual environment folder
├── dataset/                # Dataset folder
│   └── Train/              # Training data
│       └── train_data.csv  # CSV with image labels
├── .gitattributes          # Git attributes file
├── .gitignore              # Git ignore file
├── .python-version         # Specifies Python 3.9.13 (for pyenv users)
├── eda.ipynb               # Exploratory data analysis notebook
├── README.md               # This file
└── requirements.txt        # Dependency list
```
- **Note:** The `dataset/Train/` folder should contain your images (e.g., `img1.jpg`, `img2.jpg`). Add them manually if not already in the repo (they’re typically too large for Git).

### happy coding!