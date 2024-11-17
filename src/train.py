from data_preprocessing import preprocess_data
from model_architectures import create_ffn_with_positional_encoding, create_transformer_model
from sklearn.model_selection import train_test_split

def train_models(processed_data):
    # Split the data into training and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(processed_data)
    
    # Create and train the FFN with Positional Encoding model
    ffn_model = create_ffn_with_positional_encoding(input_dim, output_dim)
    ffn_model.fit(X_train, y_train, validation_data=(X_valid, y_valid))
    
    # Create and train the Transformer model
    transformer_model = create_transformer_model(input_dim, output_dim)
    transformer_model.fit(X_train, y_train, validation_data=(X_valid, y_valid))
    
    return ffn_model, transformer_model

if __name__ == '__main__':
    data_path = 'data/Date_final_dataset_balanced_float32.parquet'
    processed_data = preprocess_data(data_path)
    ffn_model, transformer_model = train_models(processed_data)
    # Save the trained models