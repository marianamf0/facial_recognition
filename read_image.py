import os
import numpy as np
from PIL import Image

def read_image(path:str, file:str):
    """
    Open an image file as grayscale (L mode).

    Args:
        path (str): Directory containing the image.
        file (str): Basename without extension (e.g., "subject01.normal").

    Returns:
        PIL.Image.Image: Loaded grayscale image.
    """
    if not os.path.exists(path + file):
        print("Arquivo não encontrado:", path + file)
    else: 
        return Image.open(path + file).convert("L")
    
def get_image_data(dimension, number_individuos:int = 15, path: str = "image/"):
    """
    Load face images for multiple subjects and expressions, build feature and one-hot label arrays.

    Each image is resized to `dimension`, converted to grayscale in [0, 1], flattened in
    column-major order (Fortran), and stacked.

    Args:
        dimension (tuple[int, int]): Target size (width, height) for resizing.
        number_individuos (int): Number of distinct subjects/classes.
        path (str): Directory containing the image files (default "image/").

    Returns:
        tuple[np.ndarray, np.ndarray]:
            x_values (np.ndarray): Shape (n_features, n_samples). Flattened images transposed to
                                   features-by-samples (note this layout).
            y_values (np.ndarray): Shape (n_samples, n_classes). One-hot labels where the class
                                   index corresponds to the subject.

    Notes:
        Expressions expected per subject:
        [".centerlight", ".glasses", ".happy", ".leftlight", ".noglasses",
         ".normal", ".rightlight", ".sad", ".sleepy", ".surprised", ".wink"].
    """
    expressions = [".centerlight", ".glasses", ".happy", ".leftlight", ".noglasses",
                   ".normal", ".rightlight", ".sad", ".sleepy", ".surprised", ".wink"]

    x_values, y_values = [], [] 
    for index in range(number_individuos):
        file = "subject0"+str(index+1) if index < 9 else "subject"+str(index+1)
        for expression in expressions: 
            img = read_image(path="image/", file=file+expression)
            img = img.resize(dimension)
            
            A = np.asarray(img, dtype=np.float64) / 255.0
            a = A.flatten(order="F") 
            
            x_values.append(a)
            
            y_value = [0]*number_individuos
            y_value[index] = 1
            y_values.append(y_value)
            
    x_values = np.array(x_values).T 
    y_values = np.array(y_values)
    
    return x_values, y_values
    
def pcacov(value, value_q:int): 
    """
    PCA via covariance eigen-decomposition with variables in rows.

    Computes the covariance matrix of `value`, obtains eigenpairs, sorts components by
    descending eigenvalue, and projects the data onto the top `value_q` eigenvectors.

    Args:
        value (np.ndarray): Data matrix of shape (n_features, n_samples). Rows are variables,
                            columns are observations (as expected by `np.cov` default).
        value_q (int): Number of principal components to keep (1 ≤ value_q ≤ n_features).

    Returns:
        tuple[np.ndarray, np.ndarray]:
            ve_cum (np.ndarray): Shape (n_features,). Cumulative explained variance ratio
                                 across all components.
            projected (np.ndarray): Shape (value_q, n_samples). Data projected onto the top
                                    `value_q` eigenvectors (i.e., scores), computed as
                                    eigvecs_q.T @ value.
    """
    cov_matrix = np.cov(value)
    eigvals, eigvecs = np.linalg.eigh(cov_matrix)
    index = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[index], eigvecs[:, index]

    eigvecs_q = eigvecs[:, :value_q]
    
    ve = eigvals / sum(eigvals) 
    ve_cum = ve.cumsum()
    
    return ve_cum, eigvecs_q.T @ value
