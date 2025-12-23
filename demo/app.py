"""Streamlit demo application for medical image classification."""

import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
from typing import Optional, Tuple

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.architectures import create_model
from utils.explainability import GradCAM, UncertaintyEstimator
from utils.core import get_device, load_config


# Page configuration
st.set_page_config(
    page_title="Medical Image Classification Demo",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .disclaimer {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
    }
    .prediction-result {
        background-color: #e8f5e8;
        border-left: 5px solid #4caf50;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">🏥 Medical Image Classification Demo</h1>', unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div class="disclaimer">
    <h3>⚠️ IMPORTANT DISCLAIMER</h3>
    <p><strong>THIS SOFTWARE IS FOR RESEARCH AND EDUCATIONAL PURPOSES ONLY</strong></p>
    <ul>
        <li>NOT approved for clinical diagnosis or treatment decisions</li>
        <li>NOT a medical device</li>
        <li>NOT validated for clinical use</li>
        <li>NOT intended to replace professional medical judgment</li>
    </ul>
    <p><strong>Always consult qualified healthcare professionals for medical decisions.</strong></p>
</div>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the trained model."""
    try:
        # Load configuration
        config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'default.yaml')
        config = load_config(config_path)
        
        # Create model
        model = create_model(
            model_name=config['model']['name'],
            num_classes=config['model']['num_classes'],
            pretrained=False,  # We'll load trained weights
            dropout=config['model']['dropout']
        )
        
        # Load checkpoint if available
        checkpoint_path = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'best.pth')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            st.success("✅ Model loaded successfully!")
        else:
            st.warning("⚠️ No trained model found. Using pretrained weights for demo.")
            # Use pretrained weights as fallback
            model = create_model(
                model_name=config['model']['name'],
                num_classes=config['model']['num_classes'],
                pretrained=True,
                dropout=config['model']['dropout']
            )
        
        model.eval()
        return model, config
        
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None


def preprocess_image(image: Image.Image, target_size: Tuple[int, int] = (224, 224)) -> torch.Tensor:
    """Preprocess uploaded image for model inference.
    
    Args:
        image: PIL Image
        target_size: Target image size
        
    Returns:
        Preprocessed image tensor
    """
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize image
    image = image.resize(target_size)
    
    # Convert to numpy array
    img_array = np.array(image) / 255.0
    
    # Normalize using ImageNet statistics
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std
    
    # Convert to tensor
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()
    
    return img_tensor.unsqueeze(0)  # Add batch dimension


def predict_image(model, image_tensor: torch.Tensor, device: str) -> Tuple[int, float, np.ndarray]:
    """Make prediction on image.
    
    Args:
        model: Trained model
        image_tensor: Preprocessed image tensor
        device: Device to run inference on
        
    Returns:
        Tuple of (predicted_class, confidence, probabilities)
    """
    model.eval()
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        predicted_class = torch.argmax(outputs, dim=1).item()
        confidence = probabilities[0, predicted_class].item()
    
    return predicted_class, confidence, probabilities[0].cpu().numpy()


def generate_gradcam(model, image_tensor: torch.Tensor, device: str, class_idx: int) -> np.ndarray:
    """Generate GradCAM visualization.
    
    Args:
        model: Trained model
        image_tensor: Preprocessed image tensor
        device: Device to run inference on
        class_idx: Class index for GradCAM
        
    Returns:
        GradCAM heatmap
    """
    try:
        gradcam = GradCAM(model)
        image_tensor = image_tensor.to(device)
        cam = gradcam.generate_cam(image_tensor, class_idx)
        gradcam.cleanup()
        return cam
    except Exception as e:
        st.warning(f"GradCAM generation failed: {e}")
        return None


def create_uncertainty_plot(probabilities: np.ndarray) -> go.Figure:
    """Create uncertainty visualization.
    
    Args:
        probabilities: Class probabilities
        
    Returns:
        Plotly figure
    """
    classes = ['Normal', 'Abnormal']
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-8))
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Class Probabilities', 'Uncertainty (Entropy)'),
        specs=[[{"type": "bar"}, {"type": "indicator"}]]
    )
    
    # Probability bar chart
    fig.add_trace(
        go.Bar(x=classes, y=probabilities, name='Probability'),
        row=1, col=1
    )
    
    # Uncertainty gauge
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=entropy,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Uncertainty"},
            gauge={'axis': {'range': [None, 1]},
                   'bar': {'color': "darkblue"},
                   'steps': [
                       {'range': [0, 0.3], 'color': "lightgreen"},
                       {'range': [0.3, 0.7], 'color': "yellow"},
                       {'range': [0.7, 1], 'color': "red"}
                   ],
                   'threshold': {'line': {'color': "red", 'width': 4},
                               'thickness': 0.75, 'value': 0.7}}
        ),
        row=1, col=2
    )
    
    fig.update_layout(height=400, showlegend=False)
    return fig


def main():
    """Main demo application."""
    
    # Load model
    model, config = load_model()
    if model is None:
        st.stop()
    
    device = get_device()
    model = model.to(device)
    
    # Sidebar
    st.sidebar.title("🔧 Configuration")
    
    # Model info
    st.sidebar.markdown("### Model Information")
    st.sidebar.info(f"**Model:** {config['model']['name']}\n\n**Classes:** {config['model']['num_classes']}\n\n**Device:** {device}")
    
    # Analysis options
    st.sidebar.markdown("### Analysis Options")
    show_gradcam = st.sidebar.checkbox("Show GradCAM", value=True)
    show_uncertainty = st.sidebar.checkbox("Show Uncertainty Analysis", value=True)
    show_metrics = st.sidebar.checkbox("Show Detailed Metrics", value=False)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📤 Upload Medical Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a medical image file",
            type=['png', 'jpg', 'jpeg', 'dcm', 'nii'],
            help="Supported formats: PNG, JPG, JPEG, DICOM, NIfTI"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Preprocess image
            try:
                image_tensor = preprocess_image(image)
                
                # Make prediction
                predicted_class, confidence, probabilities = predict_image(model, image_tensor, device)
                
                # Display results
                class_names = config.get('class_names', ['Normal', 'Abnormal'])
                predicted_label = class_names[predicted_class]
                
                st.markdown(f"""
                <div class="prediction-result">
                    <h3>🎯 Prediction Result</h3>
                    <p><strong>Predicted Class:</strong> {predicted_label}</p>
                    <p><strong>Confidence:</strong> {confidence:.3f}</p>
                    <p><strong>Class Probabilities:</strong></p>
                    <ul>
                        <li>Normal: {probabilities[0]:.3f}</li>
                        <li>Abnormal: {probabilities[1]:.3f}</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ Error processing image: {e}")
                st.stop()
    
    with col2:
        if uploaded_file is not None:
            st.markdown("### 📊 Analysis Results")
            
            # Uncertainty analysis
            if show_uncertainty:
                st.markdown("#### Uncertainty Analysis")
                uncertainty_fig = create_uncertainty_plot(probabilities)
                st.plotly_chart(uncertainty_fig, use_container_width=True)
            
            # GradCAM visualization
            if show_gradcam:
                st.markdown("#### Explainability (GradCAM)")
                try:
                    cam = generate_gradcam(model, image_tensor, device, predicted_class)
                    
                    if cam is not None:
                        # Create visualization
                        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                        
                        # Original image
                        img_array = np.array(image)
                        axes[0].imshow(img_array)
                        axes[0].set_title('Original Image')
                        axes[0].axis('off')
                        
                        # GradCAM overlay
                        cam_resized = cv2.resize(cam, (img_array.shape[1], img_array.shape[0]))
                        axes[1].imshow(img_array)
                        axes[1].imshow(cam_resized, alpha=0.6, cmap='jet')
                        axes[1].set_title('GradCAM Overlay')
                        axes[1].axis('off')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                except Exception as e:
                    st.warning(f"GradCAM visualization failed: {e}")
            
            # Detailed metrics
            if show_metrics:
                st.markdown("#### Detailed Metrics")
                
                # Calculate additional metrics
                entropy = -np.sum(probabilities * np.log(probabilities + 1e-8))
                max_prob = np.max(probabilities)
                prediction_strength = max_prob - np.max(probabilities[probabilities != max_prob])
                
                metrics_data = {
                    "Confidence": f"{confidence:.3f}",
                    "Entropy": f"{entropy:.3f}",
                    "Max Probability": f"{max_prob:.3f}",
                    "Prediction Strength": f"{prediction_strength:.3f}"
                }
                
                for metric, value in metrics_data.items():
                    st.markdown(f"""
                    <div class="metric-card">
                        <strong>{metric}:</strong> {value}
                    </div>
                    """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <p>🏥 Medical Image Classification Demo | Research & Educational Use Only</p>
        <p>⚠️ Not for clinical diagnosis | Always consult healthcare professionals</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
