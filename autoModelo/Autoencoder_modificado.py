import os
import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset

# Set a fixed seed for reproducibility across runs
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
# If using CUDA (GPU), set CUDA specific seed for reproducibility
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Autoencoder(nn.Module):
    """
    A simple Convolutional Autoencoder for image feature extraction.
    It consists of an encoder to compress images into a latent space
    and a decoder to reconstruct images from the latent space.
    """
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder part: Reduces spatial dimensions and increases feature channels
        self.encoder = nn.Sequential(
            # First convolutional layer: Input 1 channel (grayscale), output 32 channels
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32), # Batch normalization for stable training
            nn.ReLU(True),      # ReLU activation for non-linearity
            # Second convolutional layer: Stride 2 reduces spatial dimensions (e.g., 28x28 -> 14x14)
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64), # Batch normalization
            nn.ReLU(True),
            # Third convolutional layer: Stride 2 reduces spatial dimensions further (e.g., 14x14 -> 7x7)
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),# Batch normalization
            nn.ReLU(True),
        )
        # Decoder part: Increases spatial dimensions and reduces feature channels
        self.decoder = nn.Sequential(
            # First transpose convolutional layer: Stride 2 to upsample (e.g., 7x7 -> 14x14)
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64), # Batch normalization
            nn.ReLU(True),
            # Second transpose convolutional layer: Stride 2 to upsample (e.g., 14x14 -> 28x28)
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32), # Batch normalization
            nn.ReLU(True),
            # Final convolutional layer: Output 1 channel to reconstruct grayscale image
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid(),       # Sigmoid activation to output pixel values between 0 and 1
        )

    def forward(self, x):
        """
        Forward pass through the autoencoder.
        :param x: Input image tensor.
        :return: Reconstructed image tensor.
        """
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon

    def encode(self, x):
        """
        Encodes the input image into its latent representation.
        :param x: Input image tensor.
        :return: Latent space representation (features).
        """
        return self.encoder(x)

def segment_characters(image_path):
    """
    Segments individual characters from an input image using OpenCV.
    Applies adaptive thresholding and morphological operations for better results
    on handwritten text. Filters contours based on area and aspect ratio.

    :param image_path: Path to the input image file.
    :return: A list of PIL Image objects, each representing a segmented character.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not read image from {image_path}")
        return []

    # Apply adaptive thresholding: Calculates threshold for small regions
    # ADAPTIVE_THRESH_GAUSSIAN_C uses a Gaussian weighted sum of neighborhood values
    # Block size 11, C=2 (constant subtracted from mean or weighted sum)
    thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Morphological operations to clean up segmentation
    # Dilate to connect broken parts of characters, then erode to smooth edges
    kernel_dilate = np.ones((2,2), np.uint8) # Kernel for dilation
    kernel_erode = np.ones((2,2), np.uint8)  # Kernel for erosion

    thresh = cv2.dilate(thresh, kernel_dilate, iterations=1) # Connect nearby components
    thresh = cv2.erode(thresh, kernel_erode, iterations=1)   # Remove small noise and thin lines

    # Find contours in the processed binary image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    chars = []
    # Define heuristic filters for character bounding boxes
    min_char_area = 50   # Minimum area to consider a contour a character
    max_char_area = 5000 # Maximum area (to exclude lines, merged words, etc.)
    min_aspect_ratio = 0.1 # Minimum width/height ratio
    max_aspect_ratio = 10.0 # Maximum width/height ratio

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt) # Get bounding box of the contour
        area = w * h
        
        # Avoid division by zero for aspect ratio
        if h == 0: continue
        aspect_ratio = w / float(h)

        # Filter contours based on size and aspect ratio
        if min_char_area < area < max_char_area and min_aspect_ratio < aspect_ratio < max_aspect_ratio:
            # Extract the character region from the original thresholded image
            char_img = thresh[y:y+h, x:x+w]
            # Resize the character to a standard size (e.g., 28x28 for MNIST-like input)
            resized = cv2.resize(char_img, (28, 28), interpolation=cv2.INTER_AREA)
            chars.append(Image.fromarray(resized)) # Convert to PIL Image
    return chars

def preprocess_images(char_images):
    """
    Preprocesses a list of PIL character images for PyTorch.
    Converts them to tensors and stacks them into a single tensor.
    :param char_images: List of PIL Image objects.
    :return: A stacked PyTorch tensor of preprocessed images.
    """
    # Define a transform to convert PIL Image to PyTorch Tensor (scales pixels to [0, 1])
    transform = transforms.Compose([transforms.ToTensor()])
    # Apply the transform to each image and stack them into a single batch tensor
    return torch.stack([transform(img) for img in char_images])

def extract_features(model, data, device):
    """
    Extracts latent features from the given data using the autoencoder's encoder.
    Sets the model to evaluation mode and performs a forward pass without gradient calculation.
    :param model: The trained Autoencoder model.
    :param data: Input image data tensor.
    :param device: The device (CPU or CUDA) to perform computations on.
    :return: A NumPy array of flattened latent features for each input image.
    """
    model.eval() # Set model to evaluation mode (disables dropout, batchnorm updates)
    with torch.no_grad(): # Disable gradient calculations to save memory and speed up
        data = data.to(device) # Move data to the specified device
        z = model.encode(data) # Encode data to get latent features
        # Flatten the latent features for KMeans (from [batch_size, channels, height, width] to [batch_size, -1])
        return z.view(z.size(0), -1).cpu().numpy() # Move to CPU and convert to NumPy array

def main():
    """
    Main function to run the autoencoder pipeline:
    1. Segments characters from images in the 'imagenes' folder.
    2. Preprocesses segmented characters.
    3. Trains an Autoencoder to learn latent features.
    4. Extracts latent features.
    5. Applies KMeans clustering to the features to generate labels.
    6. Saves segmented characters, features, and labels.
    7. Splits data into train/test sets and saves them.
    """
    # Determine the device to use (CUDA if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    carpeta = "imagenes" # Directory containing input images

    print("📷 Searching for images in folder...")
    # Get the directory where the script is being run
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the full path to the 'imagenes' folder
    full_carpeta_path = os.path.join(script_dir, carpeta)

    # Ensure the 'imagenes' directory exists. If not, print an error and suggest creation.
    if not os.path.exists(full_carpeta_path):
        print(f"Error: The directory '{full_carpeta_path}' was not found.")
        print(f"Please create an '{carpeta}' directory in the same location as this script ('{os.path.basename(__file__)}') and place your image files (5.jpg, 6.jpg, etc.) inside it.")
        return # Exit if directory not found, as images cannot be read

    # List all image files within the 'imagenes' folder
    rutas_imagenes = [
        os.path.join(full_carpeta_path, fname)
        for fname in os.listdir(full_carpeta_path)
        if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]

    if not rutas_imagenes:
        print(f"Error: No image files found in '{full_carpeta_path}'.")
        print("Please ensure your image files (e.g., 5.jpg, 6.jpg) are placed in this directory.")
        return # Exit if no images are found


    print(f"🔍 {len(rutas_imagenes)} images found.")

    all_chars = []
    # Segment characters from each image
    for path in rutas_imagenes:
        chars = segment_characters(path)
        all_chars.extend(chars)
        print(f"🟩 {len(chars)} characters segmented from {os.path.basename(path)}")

    if len(all_chars) < 27:
        raise ValueError(f"Only {len(all_chars)} characters detected in total. At least 27 are required for KMeans (matching 26 letters + 1 for other symbols/digits). Please provide more diverse input images.")

    print("Preprocessing images...")
    # Preprocess segmented characters (convert to tensors, normalize)
    data = preprocess_images(all_chars)

    # Create a TensorDataset and DataLoader for batching
    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=32, shuffle=True) # Increased batch size, added shuffle

    # Initialize the Autoencoder model and move to the selected device
    model = Autoencoder().to(device)
    # Adam optimizer for training
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # Binary Cross-Entropy Loss for image reconstruction (pixel values are 0-1)
    criterion = nn.BCELoss()

    print("Training autoencoder...")
    model.train() # Set model to training mode
    # Train for more epochs to learn better features
    num_epochs_ae = 50 # Increased autoencoder epochs
    for epoch in range(num_epochs_ae):
        total_loss = 0
        for (x,) in loader:
            x = x.to(device) # Move input batch to device
            optimizer.zero_grad() # Clear gradients
            x_hat = model(x)      # Forward pass: reconstruct image
            loss = criterion(x_hat, x) # Calculate reconstruction loss
            loss.backward()       # Backpropagation: compute gradients
            optimizer.step()      # Update model parameters
            total_loss += loss.item() # Accumulate loss
        print(f"Epoch {epoch+1}/{num_epochs_ae}, Loss: {total_loss:.4f}")

    print("Extrayendo vectores latentes...")
    # Extract features using the trained encoder
    features = extract_features(model, data, device)

    print("Training KMeans...")
    # Apply KMeans clustering to the extracted features to generate labels
    # n_init=10 is good practice for KMeans
    kmeans = KMeans(n_clusters=27, random_state=SEED, n_init=10) # 27 for A-Z + 1 (e.g., space/digit)
    labels = kmeans.fit_predict(features)
    print(f"KMeans clustered data into {len(np.unique(labels))} clusters.")

    print("Guardando vectores y etiquetas...")
    # Save the segmented character images, extracted features, and KMeans labels
    torch.save(data, "caracteres_segmentados.pt")
    np.save("features.npy", features)
    np.save("labels.npy", labels)

    print("Separando en train/test...")
    # Split the features and generated labels into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.3, random_state=SEED, shuffle=False # Keep shuffle=False for consistency
    )
    # Save the split dataset
    np.savez("dataset_latente.npz", X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

    print("Listo. Dataset y etiquetas guardadas.")

if __name__ == "__main__":
    main()
