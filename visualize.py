'''
Visualization utilities for training and evaluation results
'''

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import os

sns.set_style('whitegrid')


class Visualizer:
    '''Visualization utilities for results'''
    
    def __init__(self, config):
        self.config = config
        self.results_path = config.RESULTS_PATH
    
    def plot_training_history(self, history, save=True):
        '''Plot training and validation metrics'''
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Training History', fontsize=16, fontweight='bold')
        
        # Accuracy
        axes[0, 0].plot(history.history['accuracy'], label='Train', linewidth=2)
        axes[0, 0].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
        axes[0, 0].set_title('Accuracy', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Loss
        axes[0, 1].plot(history.history['loss'], label='Train', linewidth=2)
        axes[0, 1].plot(history.history['val_loss'], label='Validation', linewidth=2)
        axes[0, 1].set_title('Loss', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Precision
        axes[0, 2].plot(history.history['precision'], label='Train', linewidth=2)
        axes[0, 2].plot(history.history['val_precision'], label='Validation', linewidth=2)
        axes[0, 2].set_title('Precision', fontsize=12, fontweight='bold')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Precision')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Recall
        axes[1, 0].plot(history.history['recall'], label='Train', linewidth=2)
        axes[1, 0].plot(history.history['val_recall'], label='Validation', linewidth=2)
        axes[1, 0].set_title('Recall', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Recall')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # AUC
        axes[1, 1].plot(history.history['auc'], label='Train', linewidth=2)
        axes[1, 1].plot(history.history['val_auc'], label='Validation', linewidth=2)
        axes[1, 1].set_title('AUC', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('AUC')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Learning rate (if available)
        if 'lr' in history.history:
            axes[1, 2].plot(history.history['lr'], linewidth=2, color='red')
            axes[1, 2].set_title('Learning Rate', fontsize=12, fontweight='bold')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Learning Rate')
            axes[1, 2].set_yscale('log')
            axes[1, 2].grid(True, alpha=0.3)
        else:
            axes[1, 2].text(0.5, 0.5, 'Learning Rate\nNot Tracked', 
                          ha='center', va='center', fontsize=12)
            axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.results_path, 'training_history.png'), 
                       dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {self.results_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, cm, save=True):
        '''Plot confusion matrix'''
        plt.figure(figsize=(10, 8))
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create annotations
        annotations = np.empty_like(cm).astype(str)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annotations[i, j] = f'{cm[i, j]}\n({cm_normalized[i, j]:.2%})'
        
        sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues', 
                   xticklabels=['Real', 'Fake'],
                   yticklabels=['Real', 'Fake'],
                   cbar_kws={'label': 'Count'},
                   linewidths=2, linecolor='black')
        
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=12, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
        
        if save:
            plt.savefig(os.path.join(self.results_path, 'confusion_matrix.png'), 
                       dpi=300, bbox_inches='tight')
            print(f"Confusion matrix plot saved to {self.results_path}")
        
        plt.show()
    
    def plot_roc_curve(self, y_true, y_pred_proba, save=True):
        '''Plot ROC curve'''
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=3, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier')
        
        # Find optimal threshold (Youden's J statistic)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        plt.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=10,
                label=f'Optimal Threshold = {optimal_threshold:.3f}')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        plt.title('Receiver Operating Characteristic (ROC) Curve', 
                 fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(True, alpha=0.3)
        
        if save:
            plt.savefig(os.path.join(self.results_path, 'roc_curve.png'), 
                       dpi=300, bbox_inches='tight')
            print(f"ROC curve plot saved to {self.results_path}")
        
        plt.show()
        
        return optimal_threshold
    
    def plot_precision_recall_curve(self, y_true, y_pred_proba, save=True):
        '''Plot Precision-Recall curve'''
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, color='blue', lw=3, 
                label=f'PR curve (AUC = {pr_auc:.4f})')
        
        plt.xlabel('Recall', fontsize=12, fontweight='bold')
        plt.ylabel('Precision', fontsize=12, fontweight='bold')
        plt.title('Precision-Recall Curve', fontsize=16, fontweight='bold')
        plt.legend(loc="lower left", fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        
        if save:
            plt.savefig(os.path.join(self.results_path, 'precision_recall_curve.png'), 
                       dpi=300, bbox_inches='tight')
            print(f"Precision-Recall curve saved to {self.results_path}")
        
        plt.show()
    
    def plot_sample_predictions(self, X_test, y_test, y_pred, y_pred_proba, 
                               num_samples=12, save=True):
        '''Plot sample predictions'''
        indices = np.random.choice(len(X_test), min(num_samples, len(X_test)), replace=False)
        
        rows = 3
        cols = 4
        fig, axes = plt.subplots(rows, cols, figsize=(16, 12))
        axes = axes.ravel()
        
        for i, idx in enumerate(indices):
            if i >= rows * cols:
                break
            
            axes[i].imshow(X_test[idx])
            
            true_label = 'Fake' if y_test[idx] == 1 else 'Real'
            pred_label = 'Fake' if y_pred[idx] == 1 else 'Real'
            confidence = y_pred_proba[idx][0] if y_pred[idx] == 1 else 1 - y_pred_proba[idx][0]
            
            color = 'green' if y_test[idx] == y_pred[idx] else 'red'
            axes[i].set_title(f'True: {true_label}\nPred: {pred_label} ({confidence:.1%})', 
                            color=color, fontsize=10, fontweight='bold')
            axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(len(indices), rows * cols):
            axes[i].axis('off')
        
        plt.suptitle('Sample Predictions (Green=Correct, Red=Wrong)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.results_path, 'sample_predictions.png'), 
                       dpi=300, bbox_inches='tight')
            print(f"Sample predictions plot saved to {self.results_path}")
        
        plt.show()
    
    def plot_all_visualizations(self, history, y_test, y_pred, y_pred_proba, 
                               X_test, cm):
        '''Generate all visualization plots'''
        print("\nGenerating all visualizations...")
        print("-" * 80)
        
        self.plot_training_history(history)
        self.plot_confusion_matrix(cm)
        optimal_threshold = self.plot_roc_curve(y_test, y_pred_proba)
        self.plot_precision_recall_curve(y_test, y_pred_proba)
        self.plot_sample_predictions(X_test, y_test, y_pred, y_pred_proba)
        
        print("\nAll visualizations generated successfully!")
        return optimal_threshold


if __name__ == "__main__":
    print("Visualizer module loaded successfully!")
