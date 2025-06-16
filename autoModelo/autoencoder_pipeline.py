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

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
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
    if img is None:
        print(f"No se pudo leer: {image_path}")
        return []

    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    chars = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 5 and h > 5:
            char = thresh[y:y+h, x:x+w]
            resized = cv2.resize(char, (28, 28))
            chars.append(Image.fromarray(resized))
    return chars

def preprocess_images(char_images):
    transform = transforms.Compose([transforms.ToTensor()])
    return torch.stack([transform(img) for img in char_images])

def extract_features(model, data, device):
    model.eval()
    with torch.no_grad():
        data = data.to(device)
        z = model.encode(data)
        return z.view(z.size(0), -1).cpu().numpy()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    carpeta = "imagenes"

    print("📷 Buscando imágenes en carpeta...")
    rutas_imagenes = [
        os.path.join(carpeta, fname)
        for fname in os.listdir(carpeta)
        if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]

    print(f"🔍 {len(rutas_imagenes)} imágenes encontradas.")

    all_chars = []
    for path in rutas_imagenes:
        chars = segment_characters(path)
        all_chars.extend(chars)
        print(f"🟩 {len(chars)} caracteres segmentados de {os.path.basename(path)}")

    if len(all_chars) < 27:
        raise ValueError(f"Solo se detectaron {len(all_chars)} caracteres en total. Se requieren al menos 27.")

    print("Preprocesando imágenes...")
    data = preprocess_images(all_chars)

    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    model = Autoencoder().to(device)
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

    print("Extrayendo vectores latentes...")
    features = extract_features(model, data, device)

    print("Entrenando KMeans...")
    kmeans = KMeans(n_clusters=27, random_state=SEED, n_init=10)
    labels = kmeans.fit_predict(features)

    print("Guardando vectores y etiquetas...")
    torch.save(data, "caracteres_segmentados.pt")
    np.save("features.npy", features)
    np.save("labels.npy", labels)

    print("Separando en train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.3, random_state=SEED, shuffle=False
    )
    np.savez("dataset_latente.npz", X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

    print("Listo. Dataset y etiquetas guardadas.")

if __name__ == "__main__":
    main()
