import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go
import plotly.express as px

class WildfireVisualizer:
    """
    Visualization tools for wildfire prediction analysis.
    """
    def __init__(self, style='darkgrid'):
        plt.style.use('seaborn')
        sns.set_style(style)
        self.colors = sns.color_palette("husl", 8)
    
    def plot_training_history(self, history):
        """
        Plot training history metrics.
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        metrics = ['loss', 'accuracy', 'auc_pr', 'precision']
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            ax.plot(history.history[metric], label='Training')
            ax.plot(history.history[f'val_{metric}'], label='Validation')
            ax.set_title(f'Model {metric.capitalize()}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.capitalize())
            ax.legend()
        
        plt.tight_layout()
        return fig
    
    def plot_confusion_matrix(self, y_true, y_pred, normalize=True):
        """
        Plot confusion matrix with optional normalization.
        """
        cm = confusion_matrix(y_true, y_pred)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                   cmap='Blues', cbar=True)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        return plt.gcf()
    
    def plot_feature_importance(self, model, feature_names):
        """
        Plot feature importance based on model weights.
        """
        # Get weights from the first dense layer
        weights = np.abs(model.layers[1].get_weights()[0])
        importance = np.mean(weights, axis=1)
        
        # Create dataframe of feature importance
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(data=importance_df.head(20),
                   x='Importance', y='Feature',
                   palette='viridis')
        plt.title('Top 20 Feature Importance')
        plt.xlabel('Absolute Weight')
        
        return plt.gcf()
    
    def plot_positional_encoding_impact(self, time_periods, metrics_with_pe, metrics_without_pe):
        """
        Plot the impact of positional encoding across time periods.
        """
        plt.figure(figsize=(12, 6))
        x = np.arange(len(time_periods))
        width = 0.35
        
        plt.bar(x - width/2, metrics_with_pe, width, label='With PE',
                color=self.colors[0])
        plt.bar(x + width/2, metrics_without_pe, width, label='Without PE',
                color=self.colors[1])
        
        plt.xlabel('Time Period')
        plt.ylabel('AUC-PR Score')
        plt.title('Impact of Positional Encoding Across Time')
        plt.xticks(x, time_periods)
        plt.legend()
        
        return plt.gcf()
    
    def plot_temporal_patterns(self, dates, predictions, actual):
        """
        Plot temporal patterns in predictions vs actual values.
        """
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=actual,
            name="Actual",
            line=dict(color="blue", width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=predictions,
            name="Predicted",
            line=dict(color="red", width=2)
        ))
        
        fig.update_layout(
            title="Temporal Pattern Analysis",
            xaxis_title="Date",
            yaxis_title="Probability",
            hovermode='x'
        )
        
        return fig
    
    def plot_spatial_predictions(self, lat, lon, predictions, zoom=8):
        """
        Plot spatial distribution of predictions.
        """
        fig = px.scatter_mapbox(
            lat=lat,
            lon=lon,
            color=predictions,
            zoom=zoom,
            mapbox_style="carto-positron",
            color_continuous_scale="Viridis",
            title="Spatial Distribution of Predictions"
        )
        
        return fig
    
    def create_prediction_dashboard(self, results_df):
        """
        Create an interactive dashboard for prediction analysis.
        """
        # Create subplot figure
        fig = go.Figure()
        
        # Time series plot
        fig.add_trace(go.Scatter(
            x=results_df['date'],
            y=results_df['actual'],
            name="Actual",
            yaxis="y1"
        ))
        
        fig.add_trace(go.Scatter(
            x=results_df['date'],
            y=results_df['predicted'],
            name="Predicted",
            yaxis="y1"
        ))
        
        # Update layout for multiple subplots
        fig.update_layout(
            title="Wildfire Prediction Analysis Dashboard",
            xaxis=dict(domain=[0, 0.45]),
            yaxis=dict(title="Probability", anchor="x"),
            yaxis2=dict(title="Feature Value", anchor="x2"),
            xaxis2=dict(domain=[0.55, 1.0]),
            showlegend=True
        )
        
        return fig

def plot_model_comparison(models_results):
    """
    Plot comparison of different model architectures.
    """
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_pr']
    
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4*len(metrics)))
    
    for idx, metric in enumerate(metrics):
        values = [results[metric] for results in models_results.values()]
        sns.barplot(x=list(models_results.keys()), y=values, ax=axes[idx])
        axes[idx].set_title(f'Model Comparison - {metric.upper()}')
        axes[idx].set_ylabel(metric)
    
    plt.tight_layout()
    return fig