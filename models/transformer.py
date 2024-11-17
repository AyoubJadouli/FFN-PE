import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from .positional_encoding import EnvironmentalPositionalEncoding

class MultiHeadAttention(Layer):
    """
    Multi-head attention mechanism for processing temporal features.
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % self.num_heads == 0
        
        self.depth = d_model // self.num_heads
        
        self.query_dense = Dense(d_model)
        self.key_dense = Dense(d_model)
        self.value_dense = Dense(d_model)
        
        self.dense = Dense(d_model)
        self.dropout = Dropout(dropout)
        
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, inputs, training):
        q, k, v = inputs
        batch_size = tf.shape(q)[0]
        
        q = self.query_dense(q)
        k = self.key_dense(k)
        v = self.value_dense(v)
        
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        # Scaled dot-product attention
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        attention_weights = self.dropout(attention_weights, training=training)
        
        output = tf.matmul(attention_weights, v)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, -1, self.d_model))
        
        return self.dense(output)

class TransformerBlock(Layer):
    """
    Transformer block combining multi-head attention and feed-forward layers.
    """
    def __init__(self, d_model, num_heads, dff, dropout=0.1):
        super(TransformerBlock, self).__init__()
        
        self.mha = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = tf.keras.Sequential([
            Dense(dff, activation='relu'),
            Dense(d_model)
        ])
        
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
    
    def call(self, inputs, training):
        attn_output = self.mha([inputs, inputs, inputs], training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class WildfireTransformer(Model):
    """
    Transformer model specifically designed for wildfire prediction.
    """
    def __init__(
        self,
        input_dim,
        d_model=128,
        num_heads=8,
        dff=512,
        num_blocks=4,
        dropout=0.1
    ):
        super(WildfireTransformer, self).__init__()
        
        self.input_projection = Dense(d_model)
        self.pos_encoding = EnvironmentalPositionalEncoding(
            max_position=input_dim,
            d_model=d_model
        )
        
        self.transformer_blocks = [
            TransformerBlock(d_model, num_heads, dff, dropout)
            for _ in range(num_blocks)
        ]
        
        self.dropout = Dropout(dropout)
        self.final_layer = Dense(1, activation='sigmoid')
        
    def call(self, inputs, training=False):
        # Project inputs to model dimension
        x = self.input_projection(inputs)
        x = tf.expand_dims(x, axis=1)  # Add sequence dimension
        
        # Add positional encoding
        x = self.pos_encoding(x)
        x = self.dropout(x, training=training)
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x, training)
            
        # Global average pooling
        x = tf.reduce_mean(x, axis=1)
        
        return self.final_layer(x)

def create_wildfire_transformer(
    input_dim,
    d_model=128,
    num_heads=8,
    dff=512,
    num_blocks=4,
    dropout=0.1
):
    """
    Creates and compiles a transformer model for wildfire prediction.
    """
    model = WildfireTransformer(
        input_dim=input_dim,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        num_blocks=num_blocks,
        dropout=dropout
    )
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-4,
            beta_1=0.9,
            beta_2=0.98,
            epsilon=1e-9
        ),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(curve='PR', name='auc_pr'),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall()
        ]
    )
    
    return model

# Custom learning rate scheduler for transformer training
class TransformerLearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Custom learning rate schedule with warmup for transformer training.
    """
    def __init__(self, d_model, warmup_steps=4000):
        super(TransformerLearningRateSchedule, self).__init__()
        
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps
        
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)