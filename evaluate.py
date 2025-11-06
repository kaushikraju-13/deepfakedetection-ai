'''
Model evaluation and metrics calculation
'''

import numpy as np
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_curve, 
    auc,
    precision_recall_curve,
    f1_score
)
import json
import os


class ModelEvaluator:
    '''Evaluate trained model and compute metrics'''
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
    
    def evaluate(self, X_test, y_test, verbose=1):
        '''Evaluate model on test set'''
        print("\n" + "="*80)
        print("EVALUATING MODEL ON TEST SET")
        print("="*80)
        
        # Get predictions
        y_pred_proba = self.model.predict(X_test, verbose=verbose)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        # Calculate metrics
        results = self.model.evaluate(X_test, y_test, verbose=verbose)
        metrics = dict(zip(self.model.metrics_names, results))
        
        # Additional metrics
        metrics['f1_score'] = f1_score(y_test, y_pred)
        
        # Print metrics
        print("\nTest Metrics:")
        print("-" * 80)
        for name, value in metrics.items():
            print(f"{name.capitalize()}: {value:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print("-" * 80)
        print(classification_report(y_test, y_pred, target_names=['Real', 'Fake'], digits=4))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print("-" * 80)
        print(f"                Predicted")
        print(f"              Real    Fake")
        print(f"Actual Real   {cm[0][0]:<6}  {cm[0][1]:<6}")
        print(f"       Fake   {cm[1][0]:<6}  {cm[1][1]:<6}")
        
        # Calculate additional stats
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        print(f"\nAdditional Statistics:")
        print(f"True Negatives: {tn}")
        print(f"False Positives: {fp}")
        print(f"False Negatives: {fn}")
        print(f"True Positives: {tp}")
        print(f"Specificity: {specificity:.4f}")
        print(f"Sensitivity (Recall): {sensitivity:.4f}")
        
        return {
            'metrics': metrics,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'confusion_matrix': cm
        }
    
    def save_results(self, results, filename='evaluation_results.json'):
        '''Save evaluation results to file'''
        filepath = os.path.join(self.config.RESULTS_PATH, filename)
        
        # Convert numpy arrays to lists for JSON serialization
        save_data = {
            'metrics': {k: float(v) for k, v in results['metrics'].items()},
            'confusion_matrix': results['confusion_matrix'].tolist(),
            'predictions_sample': results['predictions'][:100].tolist(),
            'probabilities_sample': results['probabilities'][:100].flatten().tolist()
        }
        
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=4)
        
        print(f"\nResults saved to: {filepath}")
    
    def get_misclassified_samples(self, X_test, y_test, results, num_samples=10):
        '''Get misclassified samples for analysis'''
        y_pred = results['predictions']
        
        # Find misclassified indices
        misclassified_idx = np.where(y_test != y_pred)[0]
        
        if len(misclassified_idx) == 0:
            print("No misclassified samples found!")
            return None
        
        # Randomly sample
        sample_size = min(num_samples, len(misclassified_idx))
        sample_idx = np.random.choice(misclassified_idx, sample_size, replace=False)
        
        misclassified_data = []
        for idx in sample_idx:
            misclassified_data.append({
                'index': int(idx),
                'true_label': 'Fake' if y_test[idx] == 1 else 'Real',
                'predicted_label': 'Fake' if y_pred[idx] == 1 else 'Real',
                'confidence': float(results['probabilities'][idx][0]),
                'image': X_test[idx]
            })
        
        return misclassified_data


if __name__ == "__main__":
    print("Evaluator module loaded successfully!")
