import os
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import tensorflow as tf

def cargar_imagenes(directorio, size=(28, 28)):
    imagenes = []
    for archivo in sorted(os.listdir(directorio)):
        if archivo.endswith((".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")): 
            img = Image.open(os.path.join(directorio, archivo)).convert("L").resize(size)
            imagenes.append(np.array(img, dtype=np.float32) / 255.0)
    X = np.array(imagenes)[..., np.newaxis]
    print(f"📸 Imágenes cargadas: {X.shape}")
    return X

def construir_autoencoder():
    inp = tf.keras.Input(shape=(28, 28, 1))
    x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(inp)
    x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', strides=2)(x)
    x = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', strides=2)(x)
    z = tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same', name="latente")(x)
    x = tf.keras.layers.Conv2DTranspose(64, 3, strides=2, activation='relu', padding='same')(z)
    x = tf.keras.layers.Conv2DTranspose(32, 3, strides=2, activation='relu', padding='same')(x)
    out = tf.keras.layers.Conv2D(1, 3, activation='sigmoid', padding='same')(x)
    modelo = tf.keras.Model(inputs=inp, outputs=out)
    modelo.compile(optimizer='adam', loss='binary_crossentropy')
    return modelo

def main():
    np.random.seed(42)
    tf.random.set_seed(42)

    X = cargar_imagenes("imagenes/")
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

    autoencoder = construir_autoencoder()
    autoencoder.fit(X_train, X_train, epochs=50, batch_size=16, validation_data=(X_test, X_test))

    encoder = tf.keras.Model(inputs=autoencoder.input,
                             outputs=autoencoder.get_layer("latente").output)

    # Extraer características latentes
    Z_train = encoder.predict(X_train)
    Z_test = encoder.predict(X_test)

    Z_flat_train = Z_train.reshape((Z_train.shape[0], -1))
    Z_flat_test = Z_test.reshape((Z_test.shape[0], -1))

    # Clustering en TRAIN
    n_clusters = min(27, Z_flat_train.shape[0])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    y_train_clusters = kmeans.fit_predict(Z_flat_train)

    # Guardar datos
    np.savez("datos_latentes_clusterizados.npz",
             X_train=Z_flat_train,
             X_test=Z_flat_test,
             y_train=y_train_clusters,
             kmeans_centroids=kmeans.cluster_centers_)

    print("✅ Dataset latente clusterizado guardado en datos_latentes_clusterizados.npz")

if __name__ == "__main__":
    main()
