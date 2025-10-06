"""
Enhanced Quantum Forensic Image Analysis - Step-by-Step Visualization
Shows detailed quantum encoding and processing stages

Author: Advanced Quantum Imaging Research
Version: 4.0 - Educational Visualization Edition
"""

import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import signal, ndimage
from datetime import datetime
import hashlib

# Page configuration
st.set_page_config(
    page_title="Quantum Image Processing - Step by Step",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .step-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        font-weight: bold;
    }
    .quantum-state {
        background-color: #f0f8ff;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #4169e1;
        font-family: monospace;
    }
    .formula-box {
        background-color: #fff9e6;
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #ffd700;
        margin: 15px 0;
    }
</style>
""", unsafe_allow_html=True)

class QuantumGates:
    """Quantum gates for image processing"""
    
    @staticmethod
    def RY(angle):
        """Rotation around Y-axis"""
        c, s = np.cos(angle/2), np.sin(angle/2)
        return np.array([[c, -s], [s, c]], dtype=complex)
    
    @staticmethod
    def Phase(phi):
        """Phase shift gate"""
        return np.array([[1, 0], [0, np.exp(1j*phi)]], dtype=complex)
    
    @staticmethod
    def Hadamard():
        """Hadamard gate"""
        return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

class QuantumImageProcessor:
    """Step-by-step quantum image processing with visualization"""
    
    def __init__(self, image_array, sample_size=8):
        """
        Initialize with image array
        sample_size: show detailed quantum state for first N×N pixels
        """
        self.original = image_array
        self.sample_size = sample_size
        self.steps = []
        
    def step1_pixel_to_angle(self):
        """Convert pixel values to quantum rotation angles"""
        st.markdown('<div class="step-box">STEP 1: Pixel Value → Quantum Angle Encoding</div>', 
                   unsafe_allow_html=True)
        
        # Formula explanation
        st.markdown('<div class="formula-box">', unsafe_allow_html=True)
        st.latex(r'\theta_i = \frac{\text{pixel\_value}_i}{255} \times \frac{\pi}{2}')
        st.markdown("""
        **What's happening:**
        - Each pixel value (0-255) is converted to an angle θ (0 to π/2)
        - Brighter pixels = larger angles
        - This angle will determine the quantum state
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Calculate angles
        thetas = (self.original / 255.0) * (np.pi / 2)
        
        # Show sample conversions
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Original Pixels")
            sample = self.original[:self.sample_size, :self.sample_size]
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(sample, cmap='gray', vmin=0, vmax=255)
            ax.set_title("Pixel Values (0-255)")
            for i in range(self.sample_size):
                for j in range(self.sample_size):
                    ax.text(j, i, f'{int(sample[i, j])}', 
                           ha='center', va='center', color='red', fontsize=8, weight='bold')
            plt.colorbar(im, ax=ax)
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.subheader("Quantum Angles θ")
            theta_sample = thetas[:self.sample_size, :self.sample_size]
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(theta_sample, cmap='viridis', vmin=0, vmax=np.pi/2)
            ax.set_title("Rotation Angles (radians)")
            for i in range(self.sample_size):
                for j in range(self.sample_size):
                    ax.text(j, i, f'{theta_sample[i, j]:.2f}', 
                           ha='center', va='center', color='white', fontsize=7, weight='bold')
            plt.colorbar(im, ax=ax, label='θ (radians)')
            st.pyplot(fig)
            plt.close()
        
        with col3:
            st.subheader("Angle Distribution")
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.hist(thetas.flatten(), bins=50, color='purple', alpha=0.7, edgecolor='black')
            ax.set_xlabel('θ (radians)')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of Quantum Angles')
            ax.axvline(np.pi/4, color='red', linestyle='--', label='π/4 (mid-gray)')
            ax.legend()
            st.pyplot(fig)
            plt.close()
        
        self.thetas = thetas
        return thetas
    
    def step2_create_quantum_state(self):
        """Create quantum state from angles"""
        st.markdown('<div class="step-box">STEP 2: Quantum State Creation (Superposition)</div>', 
                   unsafe_allow_html=True)
        
        st.markdown('<div class="formula-box">', unsafe_allow_html=True)
        st.latex(r'|\psi\rangle = \frac{1}{\sqrt{N}} \sum_{i=0}^{N-1} (\cos(\theta_i)|0\rangle + \sin(\theta_i)|1\rangle) \otimes |i\rangle')
        st.markdown("""
        **What's happening:**
        - Each pixel creates a quantum state with TWO amplitudes
        - **cos(θ)**: amplitude for |0⟩ state (ground state)
        - **sin(θ)**: amplitude for |1⟩ state (excited state)
        - The entire image exists in superposition simultaneously
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        H, W = self.original.shape
        N = H * W
        
        # Create quantum state
        amplitudes_0 = np.cos(self.thetas.flatten()) / np.sqrt(N)  # |0⟩ amplitudes
        amplitudes_1 = np.sin(self.thetas.flatten()) / np.sqrt(N)  # |1⟩ amplitudes
        
        # Show quantum state structure
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("cos(θ) - Ground State |0⟩")
            cos_sample = np.cos(self.thetas[:self.sample_size, :self.sample_size])
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(cos_sample, cmap='Blues', vmin=0, vmax=1)
            ax.set_title("Amplitude for |0⟩ state")
            for i in range(self.sample_size):
                for j in range(self.sample_size):
                    ax.text(j, i, f'{cos_sample[i, j]:.2f}', 
                           ha='center', va='center', color='darkblue', fontsize=7, weight='bold')
            plt.colorbar(im, ax=ax)
            st.pyplot(fig)
            plt.close()
            
        with col2:
            st.subheader("sin(θ) - Excited State |1⟩")
            sin_sample = np.sin(self.thetas[:self.sample_size, :self.sample_size])
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(sin_sample, cmap='Reds', vmin=0, vmax=1)
            ax.set_title("Amplitude for |1⟩ state")
            for i in range(self.sample_size):
                for j in range(self.sample_size):
                    ax.text(j, i, f'{sin_sample[i, j]:.2f}', 
                           ha='center', va='center', color='darkred', fontsize=7, weight='bold')
            plt.colorbar(im, ax=ax)
            st.pyplot(fig)
            plt.close()
        
        # Show that probabilities sum to 1
        st.markdown('<div class="quantum-state">', unsafe_allow_html=True)
        st.write("**Quantum Normalization Check:**")
        sample_probs = cos_sample[0, 0]**2 + sin_sample[0, 0]**2
        st.write(f"Example pixel [0,0]: cos²(θ) + sin²(θ) = {cos_sample[0, 0]:.3f}² + {sin_sample[0, 0]:.3f}² = {sample_probs:.6f} ≈ 1 ✓")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Store full state
        self.amplitudes_0 = amplitudes_0
        self.amplitudes_1 = amplitudes_1
        self.quantum_state = np.concatenate([amplitudes_0, amplitudes_1])
        
        return self.quantum_state
    
    def step3_apply_quantum_gate(self, gate_name="RY", angle=0.5):
        """Apply quantum gate transformation"""
        st.markdown(f'<div class="step-box">STEP 3: Apply Quantum Gate ({gate_name})</div>', 
                   unsafe_allow_html=True)
        
        if gate_name == "RY":
            gate = QuantumGates.RY(angle)
            st.markdown('<div class="formula-box">', unsafe_allow_html=True)
            st.latex(r'R_Y(\theta) = \begin{pmatrix} \cos(\theta/2) & -\sin(\theta/2) \\ \sin(\theta/2) & \cos(\theta/2) \end{pmatrix}')
            st.markdown("""
            **What's happening:**
            - The RY gate rotates each quantum state around the Y-axis
            - This enhances certain features while preserving quantum properties
            - Each pixel's quantum state is transformed independently
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        elif gate_name == "Phase":
            gate = QuantumGates.Phase(angle)
            st.markdown('<div class="formula-box">', unsafe_allow_html=True)
            st.latex(r'P(\phi) = \begin{pmatrix} 1 & 0 \\ 0 & e^{i\phi} \end{pmatrix}')
            st.markdown("""
            **What's happening:**
            - Adds a phase shift to the |1⟩ component
            - Creates forensic signatures through phase information
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Apply gate to each qubit
        N = len(self.amplitudes_0)
        new_amp_0 = np.zeros(N, dtype=complex)
        new_amp_1 = np.zeros(N, dtype=complex)
        
        for i in range(N):
            state = np.array([self.amplitudes_0[i], self.amplitudes_1[i]])
            new_state = gate @ state
            new_amp_0[i] = new_state[0]
            new_amp_1[i] = new_state[1]
        
        # Visualize transformation
        col1, col2, col3 = st.columns(3)
        
        H, W = self.original.shape
        
        with col1:
            st.subheader("Before Gate")
            before_magnitude = np.sqrt(np.abs(self.amplitudes_0[:self.sample_size*self.sample_size])**2 + 
                                      np.abs(self.amplitudes_1[:self.sample_size*self.sample_size])**2)
            before_magnitude = before_magnitude.reshape(self.sample_size, self.sample_size)
            fig, ax = plt.subplots(figsize=(5, 4))
            im = ax.imshow(before_magnitude, cmap='plasma')
            ax.set_title("State Magnitude")
            plt.colorbar(im, ax=ax)
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.subheader("Gate Applied")
            fig, ax = plt.subplots(figsize=(5, 4))
            gate_visual = np.abs(gate)
            im = ax.imshow(gate_visual, cmap='RdYlBu_r')
            ax.set_title(f"{gate_name} Gate Matrix")
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, f'{gate_visual[i, j]:.2f}', 
                           ha='center', va='center', fontsize=12, weight='bold')
            plt.colorbar(im, ax=ax)
            st.pyplot(fig)
            plt.close()
        
        with col3:
            st.subheader("After Gate")
            after_magnitude = np.sqrt(np.abs(new_amp_0[:self.sample_size*self.sample_size])**2 + 
                                     np.abs(new_amp_1[:self.sample_size*self.sample_size])**2)
            after_magnitude = after_magnitude.reshape(self.sample_size, self.sample_size)
            fig, ax = plt.subplots(figsize=(5, 4))
            im = ax.imshow(after_magnitude, cmap='plasma')
            ax.set_title("Transformed State")
            plt.colorbar(im, ax=ax)
            st.pyplot(fig)
            plt.close()
        
        self.amplitudes_0 = new_amp_0
        self.amplitudes_1 = new_amp_1
        self.quantum_state = np.concatenate([new_amp_0, new_amp_1])
        
        return self.quantum_state
    
    def step4_quantum_measurement(self, shots=1024):
        """Simulate quantum measurement"""
        st.markdown('<div class="step-box">STEP 4: Quantum Measurement (Collapse)</div>', 
                   unsafe_allow_html=True)
        
        st.markdown('<div class="formula-box">', unsafe_allow_html=True)
        st.latex(r'P(|0\rangle) = |\alpha_0|^2, \quad P(|1\rangle) = |\alpha_1|^2')
        st.markdown(f"""
        **What's happening:**
        - Quantum states are measured {shots} times
        - Measurement collapses superposition to definite values
        - Probability of each outcome = square of amplitude
        - More shots = more accurate results (like sampling)
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Calculate probabilities
        prob_0 = np.abs(self.amplitudes_0)**2
        prob_1 = np.abs(self.amplitudes_1)**2
        
        # Normalize
        total_prob = prob_0 + prob_1
        prob_0 /= total_prob
        prob_1 /= total_prob
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Measurement Probabilities")
            prob_sample = prob_1[:self.sample_size*self.sample_size].reshape(self.sample_size, self.sample_size)
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(prob_sample, cmap='RdYlGn', vmin=0, vmax=1)
            ax.set_title("P(|1⟩) for each pixel")
            for i in range(self.sample_size):
                for j in range(self.sample_size):
                    idx = i * self.sample_size + j
                    ax.text(j, i, f'{prob_sample[i, j]:.2f}', 
                           ha='center', va='center', color='black', fontsize=7, weight='bold')
            plt.colorbar(im, ax=ax, label='Probability')
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.subheader("Measurement Statistics")
            fig, ax = plt.subplots(figsize=(6, 5))
            
            # Simulate measurement for one pixel
            sample_idx = 0
            p0, p1 = prob_0[sample_idx], prob_1[sample_idx]
            
            # Simulate shots
            measurements = np.random.choice([0, 1], size=shots, p=[p0, p1])
            counts = np.bincount(measurements, minlength=2)
            
            ax.bar(['|0⟩', '|1⟩'], counts, color=['blue', 'red'], alpha=0.7, edgecolor='black')
            ax.set_ylabel('Count')
            ax.set_title(f'Measurement Results ({shots} shots)\nPixel [0,0]')
            ax.axhline(shots*p0, color='blue', linestyle='--', label=f'Expected |0⟩: {shots*p0:.0f}')
            ax.axhline(shots*p1, color='red', linestyle='--', label=f'Expected |1⟩: {shots*p1:.0f}')
            ax.legend()
            st.pyplot(fig)
            plt.close()
        
        # Update state after measurement
        self.measured_probs = prob_1
        
        return prob_1
    
    def step5_decode_to_image(self):
        """Decode quantum state back to classical image"""
        st.markdown('<div class="step-box">STEP 5: Quantum State → Classical Image</div>', 
                   unsafe_allow_html=True)
        
        st.markdown('<div class="formula-box">', unsafe_allow_html=True)
        st.latex(r'\theta_i = \arctan2(|\alpha_1|, |\alpha_0|)')
        st.latex(r'\text{pixel}_i = \frac{\theta_i \times 2 \times 255}{\pi}')
        st.markdown("""
        **What's happening:**
        - Extract angle from quantum amplitudes using arctan2
        - Convert angle back to pixel value (0-255)
        - This recovers the processed image
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Decode
        recovered_angles = np.arctan2(np.abs(self.amplitudes_1), np.abs(self.amplitudes_0))
        recovered_pixels = (recovered_angles * 2 * 255 / np.pi).reshape(self.original.shape)
        recovered_pixels = np.clip(recovered_pixels, 0, 255).astype(np.uint8)
        
        # Show full comparison
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Original Image")
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(self.original, cmap='gray', vmin=0, vmax=255)
            ax.set_title("Input")
            ax.axis('off')
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.subheader("Quantum Processed")
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(recovered_pixels, cmap='gray', vmin=0, vmax=255)
            ax.set_title("Output")
            ax.axis('off')
            st.pyplot(fig)
            plt.close()
        
        with col3:
            st.subheader("Difference Map")
            diff = np.abs(self.original.astype(float) - recovered_pixels.astype(float))
            fig, ax = plt.subplots(figsize=(6, 6))
            im = ax.imshow(diff, cmap='hot')
            ax.set_title(f"Mean Diff: {np.mean(diff):.2f}")
            ax.axis('off')
            plt.colorbar(im, ax=ax)
            st.pyplot(fig)
            plt.close()
        
        self.processed_image = recovered_pixels
        return recovered_pixels

def main():
    st.title("⚛️ Quantum Image Processing: Step-by-Step Visualization")
    st.markdown("### See exactly how quantum computing transforms images")
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        
        sample_size = st.slider("Detail View Size", 4, 16, 8, 
                               help="Size of detailed quantum state visualization")
        
        gate_type = st.selectbox("Quantum Gate", ["RY", "Phase", "Hadamard"])
        gate_angle = st.slider("Gate Angle", 0.0, np.pi, 0.5, 0.1)
        
        shots = st.slider("Measurement Shots", 256, 4096, 1024, 256,
                         help="Number of quantum measurements")
        
        st.markdown("---")
        st.info("This tool shows the actual quantum mechanics behind quantum image processing")
    
    # File upload
    uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg', 'bmp'])
    
    if uploaded_file is not None:
        # Load and prepare image
        img = Image.open(uploaded_file).convert('L')  # Grayscale
        img = img.resize((64, 64), Image.LANCZOS)  # Resize for faster processing
        img_array = np.array(img, dtype=np.float64)
        
        st.success(f"Image loaded: {img_array.shape}")
        
        if st.button("Run Step-by-Step Quantum Processing", type="primary", use_container_width=True):
            processor = QuantumImageProcessor(img_array, sample_size=sample_size)
            
            with st.spinner("Running quantum processing..."):
                # Step 1
                st.markdown("---")
                thetas = processor.step1_pixel_to_angle()
                
                # Step 2
                st.markdown("---")
                quantum_state = processor.step2_create_quantum_state()
                
                # Step 3
                st.markdown("---")
                transformed_state = processor.step3_apply_quantum_gate(gate_type, gate_angle)
                
                # Step 4
                st.markdown("---")
                measured_state = processor.step4_quantum_measurement(shots)
                
                # Step 5
                st.markdown("---")
                final_image = processor.step5_decode_to_image()
            
            st.success("Quantum processing complete!")
            st.balloons()
    else:
        st.info("Upload an image to see quantum processing in action")
        
        # Show example visualization
        st.markdown("### Example: How a single pixel is encoded")
        
        example_pixel = st.slider("Example pixel value", 0, 255, 128)
        theta = (example_pixel / 255.0) * (np.pi / 2)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="quantum-state">', unsafe_allow_html=True)
            st.write(f"**Pixel value:** {example_pixel}")
            st.write(f"**Quantum angle θ:** {theta:.4f} radians ({np.degrees(theta):.2f}°)")
            st.write(f"**Amplitude |0⟩:** cos(θ) = {np.cos(theta):.4f}")
            st.write(f"**Amplitude |1⟩:** sin(θ) = {np.sin(theta):.4f}")
            st.write(f"**Probability |0⟩:** {np.cos(theta)**2:.4f}")
            st.write(f"**Probability |1⟩:** {np.sin(theta)**2:.4f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='polar'))
            ax.plot([0, theta], [0, 1], 'b-', linewidth=3, label='Quantum State')
            ax.plot([0, 0], [0, np.cos(theta)], 'g--', linewidth=2, label=f'|0⟩: {np.cos(theta):.2f}')
            ax.plot([np.pi/2, np.pi/2], [0, np.sin(theta)], 'r--', linewidth=2, label=f'|1⟩: {np.sin(theta):.2f}')
            ax.set_ylim(0, 1)
            ax.set_title("Quantum State Visualization", pad=20)
            ax.legend(loc='upper right')
            st.pyplot(fig)
            plt.close()

if __name__ == "__main__":
    main()
