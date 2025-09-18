"""
Script to download pre-trained models and required files for the concentration analysis system.
"""

import os
import urllib.request
import zipfile
import logging
from pathlib import Path


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def download_file(url: str, filepath: str, description: str = None):
    """
    Download a file from URL to filepath.
    
    Args:
        url: URL to download from
        filepath: Local file path to save to
        description: Description for logging
    """
    if description:
        logging.info(f"Downloading {description}...")
    else:
        logging.info(f"Downloading {os.path.basename(filepath)}...")
    
    try:
        urllib.request.urlretrieve(url, filepath)
        logging.info(f"Successfully downloaded to {filepath}")
    except Exception as e:
        logging.error(f"Failed to download {url}: {e}")
        raise


def extract_zip(zip_path: str, extract_to: str):
    """Extract ZIP file to specified directory."""
    logging.info(f"Extracting {zip_path} to {extract_to}...")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        logging.info("Extraction completed")
        
        # Remove the zip file after extraction
        os.remove(zip_path)
        logging.info(f"Removed {zip_path}")
    except Exception as e:
        logging.error(f"Failed to extract {zip_path}: {e}")
        raise


def download_dlib_models():
    """Download dlib facial landmark predictor."""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Download shape predictor for 68 facial landmarks
    landmark_url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    landmark_path = models_dir / "shape_predictor_68_face_landmarks.dat.bz2"
    
    if not (models_dir / "shape_predictor_68_face_landmarks.dat").exists():
        download_file(landmark_url, str(landmark_path), "dlib 68-point facial landmark predictor")
        
        # Extract bz2 file
        import bz2
        with bz2.BZ2File(landmark_path, 'rb') as f_in:
            with open(models_dir / "shape_predictor_68_face_landmarks.dat", 'wb') as f_out:
                f_out.write(f_in.read())
        
        # Remove compressed file
        os.remove(landmark_path)
        logging.info("dlib landmark predictor extracted and ready")
    else:
        logging.info("dlib landmark predictor already exists")


def create_placeholder_models():
    """Create placeholder model files for development."""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    placeholder_models = [
        "gaze_model.pth",
        "blink_model.pth", 
        "head_pose_model.pth",
        "engagement_model.pth"
    ]
    
    for model_name in placeholder_models:
        model_path = models_dir / model_name
        if not model_path.exists():
            # Create empty placeholder file
            model_path.touch()
            logging.info(f"Created placeholder: {model_path}")
            
            # Add a note about the placeholder
            with open(model_path.with_suffix('.txt'), 'w') as f:
                f.write(f"Placeholder for {model_name}\n")
                f.write("Replace this with actual trained model weights.\n")
                f.write("Training scripts and instructions available in docs/\n")


def download_sample_data():
    """Download or create sample data for testing."""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Create sample data directories
    sample_dirs = [
        "MPIIGaze/sample",
        "GazeCapture/sample", 
        "ZJU_Eyeblink/sample",
        "NTHU-DDD/sample",
        "BIWI_Kinect/sample",
        "DAiSEE/sample"
    ]
    
    for dir_path in sample_dirs:
        full_path = data_dir / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        
        # Create README for each dataset
        readme_path = full_path / "README.md"
        if not readme_path.exists():
            with open(readme_path, 'w') as f:
                dataset_name = dir_path.split('/')[0]
                f.write(f"# {dataset_name} Dataset\n\n")
                f.write(f"This directory should contain the {dataset_name} dataset.\n")
                f.write("Please download the dataset from the official source and place it here.\n\n")
                f.write("For more information, see the project documentation.\n")
        
        logging.info(f"Prepared directory: {full_path}")


def setup_directory_structure():
    """Ensure all necessary directories exist."""
    directories = [
        "models",
        "data", 
        "logs",
        "outputs",
        "configs",
        "scripts",
        "docs",
        "tests"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logging.info(f"Directory ready: {directory}")


def main():
    """Main function to set up the project environment."""
    setup_logging()
    logging.info("Setting up Multi-Modal Concentration Analysis System...")
    
    try:
        # Setup directory structure
        setup_directory_structure()
        
        # Download dlib models
        download_dlib_models()
        
        # Create placeholder models
        create_placeholder_models()
        
        # Setup sample data directories
        download_sample_data()
        
        logging.info("Setup completed successfully!")
        logging.info("\nNext steps:")
        logging.info("1. Install dependencies: pip install -r requirements.txt")
        logging.info("2. Download actual datasets to data/ directories")
        logging.info("3. Train models or download pre-trained weights")
        logging.info("4. Run the system: python src/main.py")
        
    except Exception as e:
        logging.error(f"Setup failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())


