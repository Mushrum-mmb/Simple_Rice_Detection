# Simple_Rice_Detection

⭐ Star me on GitHub — it motivates me a lot!

🔥 Share it if you like it!!!

[![Share](https://img.shields.io/badge/share-000000?logo=x&logoColor=white)](https://x.com/intent/tweet?text=Check%20out%20this%20project%20on%20GitHub:%20https://github.com/Abblix/Oidc.Server%20%23OpenIDConnect%20%23Security%20%23Authentication)
[![Share](https://img.shields.io/badge/share-1877F2?logo=facebook&logoColor=white)](https://www.facebook.com/sharer/sharer.php?u=https://github.com/Abblix/Oidc.Server)
[![Share](https://img.shields.io/badge/share-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/sharing/share-offsite/?url=https://github.com/Abblix/Oidc.Server)
[![Share](https://img.shields.io/badge/share-FF4500?logo=reddit&logoColor=white)](https://www.reddit.com/submit?title=Check%20out%20this%20project%20on%20GitHub:%20https://github.com/Abblix/Oidc.Server)
[![Share](https://img.shields.io/badge/share-0088CC?logo=telegram&logoColor=white)](https://t.me/share/url?url=https://github.com/Abblix/Oidc.Server&text=Check%20out%20this%20project%20on%20GitHub)

### Table of Contents
- [About](#-about)
- [Features](#-features)
- [Installation](#%EF%B8%8F-installation)
- [Usage](#%EF%B8%8F-usage)
- [Google Colab Usage](#-google-colab-usage)
- [How It Works](#-how-it-works)
- [License](#-license)


### 🚀 About

A computer vision application that automatically detects and counts rice grains in images using a hybrid approach combining watershed algorithm and contour detection.

* Author: [Mushrum-mmb](https://github.com/Mushrum-mmb/)
* Framework: [gradio](https://www.gradio.app/)

![image](https://github.com/user-attachments/assets/506cf60d-e081-4e38-9aaf-42ed5f5fb257)
![image](https://github.com/user-attachments/assets/be6f8423-81b6-4008-ae29-de91a8ad1eac)

### 🎓 Features
* Advanced Detection Algorithm:
Utilizes a hybrid approach combining watershed algorithm and contour detection to accurately identify individual rice grains even when they're touching.
* Visual Processing Pipeline:
Implements a comprehensive image processing pipeline including contrast adjustment, segmentation, noise removal, and morphological operations.
* Rice Grain Counting:
Automatically counts and labels each detected rice grain on the input image.
* Process Visualization:
Provides a visual breakdown of each step in the detection process (available in the standalone version).
* Web Interface:
Features a user-friendly Gradio web interface that allows users to upload images and get instant results.
* Device Compatibility:
Compatible with various devices and can be run locally or on Google Colab for users with low-spec computers.

### ⬇️ Installation
***Ensure that you have already installed Git and set up your Python environment.***

To run this application locally, ensure you have opened the CMD and have the following dependencies installed:

```bash
pip install torch numpy opencv-python gradio matplotlib
```

### ▶️ Usage
Open CMD and clone the repository.
```bash
git clone https://github.com/Mushrum-mmb/Simple_Rice_Detection.git
```
Then cd to the clone path.
```bash
cd Rice_Grain_Detection
```

Then launch the application by running rice_detector.py.

```bash
python rice_detector.py

```
### 💻 Google Colab Usage

cd Rice-Grain-Detection

### [Click here for access my notebook](https://colab.research.google.com/drive/1uSx4NkhXVZqAetb9ug3tsZRLJ2CWwNpo?usp=sharing)

Run the first cell to install `Gradio`

![image](https://github.com/user-attachments/assets/85778e45-9bdf-4b05-a9d8-48efedd338f6)

Run the final cell and enjoy it ^V^.

### 👍 How It Works

**1. Image Pre-processing:**
* Enhances image by adjusting contrast and brightness for better segmentation
* Applies color-based segmentation to separate rice from background
* Converts to grayscale for further processing
* Performs additional segmentation for better rice grain definition

**2. Noise Removal:**
* Defines morphological kernel for noise removal operations
* Applies opening to remove small noise and smoothen grain boundaries
* Uses erosion to separate touching grains

**3. Watershed Segmentation:**
* Creates sure background area by dilating the eroded image
* Uses distance transform to find the centers of foreground objects
* Thresholds to get "sure foreground" areas (central regions of grains)
* Finds unknown region (area that's not sure background or sure foreground)
* Applies marker labeling for watershed algorithm
* Implements watershed algorithm to segment touching grains

**4. Grain Counting:**
* Processes each watershed-identified grain
* Creates binary mask for current grain
* Finds contour of the grain mask
* Calculates and marks the centroid of each grain
* Applies supplementary contour detection for grains that watershed may have missed
* Combines results from both methods for accurate counting

**5. Result Visualization:**
* Shows step-by-step processing in the standalone version
* Displays final annotated image with grain count in both versions

### 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.
