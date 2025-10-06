# ⚛️ Quantum Image Processing – Step-by-Step Visualization

### 🧠 Educational Simulation of Quantum Forensic Image Analysis

This project demonstrates how **Quantum Image Processing (QIP)** principles can be applied to **digital forensics**.  
It provides an **interactive Streamlit web app** that visualizes how images are encoded, processed, and decoded using simulated **quantum gates**.  
The tool helps users understand the quantum mechanics behind next-generation forensic imaging — all within a Python environment.

---

## 🚀 Features

- 🔍 **Quantum Encoding** — Converts classical pixel values into quantum rotation angles.  
- ⚙️ **Quantum Gate Simulation** — Applies RY, Phase, and Hadamard gates to transform image data.  
- 📊 **Measurement Visualization** — Simulates quantum measurement and probability collapse.  
- 🖼️ **Image Reconstruction** — Decodes the quantum-processed state back into a classical image.  
- 🎨 **Interactive Streamlit UI** — Step-by-step visualization with real-time plots and histograms.  

---

## 🧩 Project Overview

This project bridges **Quantum Computing** and **Digital Forensics**, simulating how quantum techniques can:
- Enhance image analysis and contrast detection,
- Embed tamper-proof forensic signatures,
- Secure evidence using quantum watermarking,
- Enable parallel image feature extraction.

### Quantum Concepts Used:
| Concept | Description |
|----------|-------------|
| **Qubit** | Quantum unit representing both 0 and 1 simultaneously. |
| **Superposition** | Allows all pixels to exist and be processed in parallel. |
| **Quantum Gates** | Transformations applied to qubits (RY, Phase, Hadamard). |
| **Measurement** | Collapses quantum states into observable values. |

---

## 🔬 How It Works

1. **Pixel-to-Angle Encoding**  
   Each grayscale pixel value (0–255) is mapped to a rotation angle between 0 and π/2.  
   \[
   \theta_i = \frac{\text{pixel\_value}_i}{255} \times \frac{\pi}{2}
   \]

2. **Quantum State Creation**  
   Each pixel becomes a quantum state:
   \[
   |\psi_i\rangle = \cos(\theta_i)|0\rangle + \sin(\theta_i)|1\rangle
   \]

3. **Quantum Gate Application**  
   Quantum gates (RY, Phase, Hadamard) are applied to modify or encode features.

4. **Quantum Measurement**  
   Simulates state collapse to classical probabilities \( P(|0\rangle), P(|1\rangle) \).

5. **Decoding Back to Image**  
   The processed amplitudes are converted back to pixel intensities for visualization.

---

## 🕵️ Applications in Digital Forensics

- **Quantum Watermarking:** Embed tamper-proof forensic signatures using phase shifts.  
- **Evidence Integrity:** Quantum states collapse upon observation, preventing alteration.  
- **Fast Pattern Recognition:** Quantum parallelism accelerates forensic image scanning.  
- **Secure Image Authentication:** Phase and amplitude act as unique cryptographic markers.  

---

## 🛠️ Installation

### Prerequisites
Make sure you have the following installed:
- Python 3.8 or higher  
- pip (Python package installer)

### Setup

```bash
# Clone this repository
https://github.com/Sindhu9365/quantum-image-encoder 

# Navigate to the folder
cd quantum-image-processing

# Install required dependencies
pip install -r requirements.txt

