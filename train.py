import cv2
import numpy as np
from skimage.feature import greycomatrix, greycoprops

def extract_features(image_path):
    """
    Extracts texture and color features from an image.

    Parameters:
        image_path (str): Path to the input image.

    Returns:
        dict: A dictionary containing texture and color features.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or invalid image path.")
    
    # Convert to grayscale for texture analysis
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Texture features using GLCM (Gray Level Co-occurrence Matrix)
    glcm = greycomatrix(gray_image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = greycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = greycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = greycoprops(glcm, 'homogeneity')[0, 0]
    energy = greycoprops(glcm, 'energy')[0, 0]
    correlation = greycoprops(glcm, 'correlation')[0, 0]

    # Color features (mean and standard deviation for each channel)
    mean_color = cv2.mean(image)[:3]
    std_color = np.std(image, axis=(0, 1))

    # Combine features into a dictionary
    features = {
        "contrast": contrast,
        "dissimilarity": dissimilarity,
        "homogeneity": homogeneity,
        "energy": energy,
        "correlation": correlation,
        "mean_color": mean_color,
        "std_color": std_color.tolist()
    }

    return features