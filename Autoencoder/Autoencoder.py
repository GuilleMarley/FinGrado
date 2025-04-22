import os
import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon

    def encode(self, x):
        return self.encoder(x)

def segment_characters(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    chars = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 5 and h > 5:  # evitar ruido
            char = thresh[y:y+h, x:x+w]
            resized = cv2.resize(char, (28, 28))
            chars.append(Image.fromarray(resized))
    return chars

def preprocess_images(char_images):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    return torch.stack([transform(img) for img in char_images])

def extract_features(model, data, device):
    model.eval()
    with torch.no_grad():
        data = data.to(device)
        z = model.encode(data)
        return z.view(z.size(0), -1).cpu().numpy()

def cluster_features(features, n_clusters=27):
    kmeans = KMeans(n_clusters=n_clusters, n_init='auto')
    labels = kmeans.fit_predict(features)
    return labels, kmeans

# MAIN
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_path = "./0000SWEL.JPG"

    print("Segmentando caracteres...")
    char_images = segment_characters(image_path)
    if len(char_images) < 27:
        raise ValueError(f"Se encontraron solo {len(char_images)} caracteres, necesitas al menos 27.")

    print("Preparando datos...")
    data = preprocess_images(char_images)
    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = ConvAutoencoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    print("Entrenando autoencoder...")
    model.train()
    for epoch in range(20):
        total_loss = 0
        for (x,) in loader:
            x = x.to(device)
            optimizer.zero_grad()
            x_hat = model(x)
            loss = criterion(x_hat, x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    print("Extrayendo características latentes...")
    features = extract_features(model, data, device)

    print("Agrupando en 27 clústeres...")
    labels, kmeans = cluster_features(features, n_clusters=27)

    np.save("labels.npy", labels)
    torch.save(data, "caracteres_segmentados.pt")
    print("Clustering completo. Etiquetas guardadas en labels.npy")

if __name__ == "__main__":
    main()
