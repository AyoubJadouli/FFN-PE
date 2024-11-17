import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.regularizers import l2

class StandardFFN(Model):
    """
    Standard Feedforward Neural Network implementation.
    Serves as a baseline model for comparison with FFN-PE.
    """
    def __init__(
        self,
        input_dim,
        hidden_layers=[256, 128, 64],
        dropout_rate=0.3,
        l2_reg=0.01,
        batch_norm=True
    ):
        super(StandardFFN, self).__init__()
        
        self.model_layers = []
        
        # Input layer
        self.input_layer = Input(shape=(input_dim,))
        
        # Hidden layers
        for units in hidden_layers:
            self.model_layers.extend([
                Dense(
                    units,
                    activation='relu',
                    kernel_regularizer=l2(l2_reg)
                ),
                Dropout(dropout_rate)
            ])
            if batch_norm:
                self.model_layers.append(BatchNormalization())
        
        # Output layer
        self.output_layer = Dense(1, activation='sigmoid')
        
    def call(self, inputs, training=False):
        """Forward pass of the model."""
        x = inputs
        for layer in self.model_layers:
            if isinstance(layer, Dropout) or isinstance(layer, BatchNormalization):
                x = layer(x, training=training)
            else:
                x = layer(x)
        return self.output_layer(x)
    
    def build_model(self):
        """Builds and returns a compiled model."""
        inputs = Input(shape=(self.input_layer.shape[1],))
        outputs = self.call(inputs)
        model = Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(curve='PR', name='auc_pr'),
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall()
            ]
        )
        return model

class WildfireFFN(StandardFFN):
    """
    Specialized FFN for wildfire prediction.
    Includes domain-specific features and architecture choices.
    """
    def __init__(
        self,
        input_dim,
        hidden_layers=[512, 256, 128, 64],
        dropout_rate=0.4,
        l2_reg=0.015,
        batch_norm=True
    ):
        super(WildfireFFN, self).__init__(
            input_dim,
            hidden_layers,
            dropout_rate,
            l2_reg,
            batch_norm
        )
        
        # Additional layers specific to wildfire prediction
        self.feature_extractor = Dense(
            128,
            activation='relu',
            kernel_regularizer=l2(l2_reg),
            name='feature_extractor'
        )
        
    def call(self, inputs, training=False):
        """
        Forward pass with additional feature extraction layer.
        """
        # Extract high-level features
        x = self.feature_extractor(inputs)
        
        # Apply standard FFN layers
        for layer in self.model_layers:
            if isinstance(layer, Dropout) or isinstance(layer, BatchNormalization):
                x = layer(x, training=training)
            else:
                x = layer(x)
                
        return self.output_layer(x)
    
    @staticmethod
    def create_callbacks():
        """Creates callbacks for training."""
        return [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3,
                min_lr=1e-6
            )
        ]

# Usage example
def create_and_train_ffn(X_train, y_train, X_valid, y_valid):
    """
    Creates and trains a WildfireFFN model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_valid: Validation features
        y_valid: Validation labels
        
    Returns:
        Trained model and training history
    """
    # Create model
    model = WildfireFFN(input_dim=X_train.shape[1])
    compiled_model = model.build_model()
    
    # Train model
    history = compiled_model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=128,
        validation_data=(X_valid, y_valid),
        callbacks=WildfireFFN.create_callbacks(),
        verbose=1
    )
    
    return compiled_model, history