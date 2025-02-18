from keras import layers
from keras.models import Model
from mltu.tensorflow.model_utils import residual_block  # Verifica que esta función esté disponible

def train_model(input_dim, output_dim, activation="leaky_relu", dropout=0.2):
    # Definir la entrada del modelo con las dimensiones adecuadas
    inputs = layers.Input(shape=input_dim, name="input")

    # Normalizar los valores de entrada a un rango [0, 1]
    input = layers.Lambda(lambda x: x / 255)(inputs)

    # Primer bloque residual
    x1 = residual_block(input, filters=16, activation=activation, skip_conv=True, strides=1, dropout=dropout)

    # Segundo bloque residual con downsampling
    x2 = residual_block(x1, filters=16, activation=activation, skip_conv=True, strides=2, dropout=dropout)
    x3 = residual_block(x2, filters=16, activation=activation, skip_conv=False, strides=1, dropout=dropout)

    # Tercer bloque residual
    x4 = residual_block(x3, filters=32, activation=activation, skip_conv=True, strides=2, dropout=dropout)
    x5 = residual_block(x4, filters=32, activation=activation, skip_conv=False, strides=1, dropout=dropout)

    # Cuarto bloque residual
    x6 = residual_block(x5, filters=64, activation=activation, skip_conv=True, strides=1, dropout=dropout)
    x7 = residual_block(x6, filters=64, activation=activation, skip_conv=False, strides=1, dropout=dropout)

    # Aplanar los tensores espaciales en secuencias para la capa BLSTM
    squeezed = layers.Reshape((x7.shape[-3] * x7.shape[-2], x7.shape[-1]))(x7)

    # Capa Bidireccional LSTM para aprender secuencias
    blstm = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(squeezed)

    # Capa densa de salida con activación softmax
    output = layers.Dense(output_dim + 1, activation="softmax", name="output")(blstm)

    # Definir el modelo
    model = Model(inputs=inputs, outputs=output)
    
    return model
