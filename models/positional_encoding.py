import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer

class PositionalEncoding(Layer):
    """
    Positional encoding layer that adds temporal information to input features.
    Particularly useful for datasets with lagged features, where temporal order matters.
    """
    def __init__(self, max_position, d_model):
        super(PositionalEncoding, self).__init__()
        self.max_position = max_position
        self.d_model = d_model
        self.pos_encoding = self.create_positional_encoding()
    
    def create_positional_encoding(self):
        """
        Creates a matrix of positional encodings using sine and cosine functions.
        Based on the approach from "Attention is All You Need" paper.
        """
        angles = self._get_angles()
        pos_encoding = self._apply_trigonometric_functions(angles)
        return tf.cast(pos_encoding[np.newaxis, ...], dtype=tf.float32)
    
    def _get_angles(self):
        """Calculate the angles for positional encoding."""
        positions = np.arange(self.max_position)[:, np.newaxis]
        dims = np.arange(self.d_model)[np.newaxis, :]
        rates = 1 / np.power(10000, (2 * (dims // 2)) / np.float32(self.d_model))
        return positions * rates
    
    def _apply_trigonometric_functions(self, angles):
        """Apply sine and cosine functions to the angles."""
        encoding = np.zeros_like(angles)
        encoding[:, 0::2] = np.sin(angles[:, 0::2])  # Apply sin to even indices
        encoding[:, 1::2] = np.cos(angles[:, 1::2])  # Apply cos to odd indices
        return encoding
    
    def call(self, inputs):
        """
        Add positional encoding to the input tensor.
        
        Args:
            inputs: Input tensor of shape (batch_size, sequence_length, d_model)
            
        Returns:
            Tensor with positional encoding added
        """
        batch_size = tf.shape(inputs)[0]
        seq_length = tf.shape(inputs)[1]
        
        # Ensure positional encoding matches input dimensions
        pos_encoding = self.pos_encoding[:, :seq_length, :]
        pos_encoding = tf.tile(pos_encoding, [batch_size, 1, 1])
        
        return inputs + pos_encoding

    def get_config(self):
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update({
            "max_position": self.max_position,
            "d_model": self.d_model,
        })
        return config

# Time-based positional encoding for environmental data
class EnvironmentalPositionalEncoding(PositionalEncoding):
    """
    Specialized positional encoding for environmental time series data.
    Incorporates seasonal and cyclical patterns common in environmental data.
    """
    def __init__(self, max_position, d_model, seasonal_period=365):
        super().__init__(max_position, d_model)
        self.seasonal_period = seasonal_period
        
    def create_positional_encoding(self):
        """
        Creates positional encoding with additional seasonal components.
        """
        base_encoding = super().create_positional_encoding()
        seasonal_encoding = self._create_seasonal_encoding()
        return base_encoding + seasonal_encoding
    
    def _create_seasonal_encoding(self):
        """
        Creates seasonal encoding component based on yearly cycles.
        """
        positions = np.arange(self.max_position)[:, np.newaxis]
        seasonal_angles = 2 * np.pi * positions / self.seasonal_period
        
        encoding = np.zeros((self.max_position, self.d_model))
        encoding[:, 0::2] = np.sin(seasonal_angles)
        encoding[:, 1::2] = np.cos(seasonal_angles)
        
        return tf.cast(encoding[np.newaxis, ...], dtype=tf.float32)

# Usage example
def create_environmental_encoding(sequence_length, feature_dim, seasonal_period=365):
    """
    Creates positional encoding specifically for environmental data.
    
    Args:
        sequence_length: Length of the input sequence
        feature_dim: Dimension of the feature space
        seasonal_period: Number of days in seasonal cycle (default: 365)
        
    Returns:
        EnvironmentalPositionalEncoding layer
    """
    return EnvironmentalPositionalEncoding(
        max_position=sequence_length,
        d_model=feature_dim,
        seasonal_period=seasonal_period
    )