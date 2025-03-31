# Chicken Disease Identification by Fecal Images

This project develops a machine learning model to classify chicken health status (diseases or healthy) using fecal images, as part of our final class project. This README provides instructions to set up your coding environment consistently across the team.

## Prerequisites
- **Operating System:** Windows 11 (adaptable to macOS/Linux with minor changes).
- **Git:** Installed for version control ([download here](https://git-scm.com/download/win)).
- **Python:** We’ll use version 3.11.9 (TensorFlow 2.19.0 compatible).

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

### 2. Install Python 3.11.9
We use Python 3.11.9 to ensure compatibility with TensorFlow 2.19.0.
- **Option A: Manual Install (Recommended)**
  1. Download Python 3.11.9 from [python.org/downloads](https://www.python.org/downloads/) (scroll to “Looking for a specific release?”).
  2. Run the installer:
     - Check “Add Python 3.11 to PATH”.
     - Install to default location (e.g., `C:\Users\YourUsername\AppData\Local\Programs\Python\Python311`).
  3. Verify:
     ```
     python --version
     ```
     Should output `Python 3.11.9`. If not, use the full path: `C:\Users\YourUsername\AppData\Local\Programs\Python\Python311\python.exe --version`.

- **Option B: Use `pyenv-win` (Advanced)**
  1. Install `pyenv-win`:
     ```
     pip install pyenv-win
     ```
  2. Add to PATH (via “Edit the system environment variables” > “User variables” > “Path”):
     ```
     C:\Users\YourUsername\.pyenv\pyenv-win\bin
     C:\Users\YourUsername\.pyenv\pyenv-win\shims
     ```
  3. Install Python 3.11.9:
     ```
     pyenv install 3.11.9
     ```
  4. Set local version:
     ```
     pyenv local 3.11.9
     ```

### 3. Create and Activate Virtual Environment
1. Create a virtual environment named `chick-env`:
   - If Python 3.11.9 is in PATH:
     ```
     python -m venv chick-env
     ```
   - If using full path:
     ```
     C:\Users\YourUsername\AppData\Local\Programs\Python\Python311\python.exe -m venv chick-env
     ```
2. Activate it:
   ```
   chick-env\Scripts\activate
   ```
   Your prompt should show `(chick-env)`.
3. Verify Python version:
   ```
   python --version
   ```
   Should be `Python 3.11.9`.

### 4. Install Dependencies
1. Install all required libraries from `requirements.txt`:
   ```
   pip install -r requirements.txt
   ```
   This includes TensorFlow 2.19.0, Pandas, Scikit-learn, OpenCV, Matplotlib, Seaborn, and Pillow.
2. Verify TensorFlow:
   ```
   python -c "import tensorflow as tf; print(tf.__version__)"
   ```
   Should output `2.19.0`.

### 5. Recommended Coding Environment
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
- **Wrong Python Version:** Ensure `C:\Users\YourUsername\.pyenv\pyenv-win\shims` (if using `pyenv`) or `C:\Users\YourUsername\AppData\Local\Programs\Python\Python311` is before other Python paths in your PATH.
- **TensorFlow Errors:** Reinstall with `pip install tensorflow==2.19.0`.
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
├── .python-version         # Specifies Python 3.11.9 (for pyenv users)
├── eda.ipynb               # Exploratory data analysis notebook
├── README.md               # This file
└── requirements.txt        # Dependency list
```
- **Note:** The `dataset/Train/` folder should contain your images (e.g., `img1.jpg`, `img2.jpg`). Add them manually if not already in the repo (they’re typically too large for Git).

happy coding!