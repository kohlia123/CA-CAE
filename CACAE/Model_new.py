import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

class StandardAE:
    def __init__(self, input_dim, encoding_dims=[512, 128], latent_dim=30, learning_rate=0.0005):
        self.input_dim = input_dim
        self.encoding_dims = encoding_dims
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate

        # Encoder
        inputs = Input(shape=(self.input_dim,))
        x = inputs
        for dim in encoding_dims:
            x = Dense(dim, activation='relu')(x)
            x = BatchNormalization()(x)
        
        # Latent Space (HiddenLayer)
        latent = Dense(latent_dim, name='HiddenLayer')(x)
        
        # Decoder
        x = latent
        for dim in reversed(encoding_dims):
            x = Dense(dim, activation='relu')(x)
            x = BatchNormalization()(x)
            
        outputs = Dense(input_dim, activation='sigmoid')(x)
        
        self.model = Model(inputs, outputs)
        self.model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')

    def fit(self, data, epochs=100, batch_size=32):
        # Removed the Reshape logic from CA-CAE because Standard AE takes flat data
        return self.model.fit(data, data, 
                              epochs=epochs, 
                              batch_size=batch_size, 
                              validation_split=0.2,
                              verbose=1)

    def extract_feature(self, x):
        # Direct extraction from the latent layer
        encoder = Model(inputs=self.model.input, outputs=self.model.get_layer('HiddenLayer').output)
        return encoder.predict(x)