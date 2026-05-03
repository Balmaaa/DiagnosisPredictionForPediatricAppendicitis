"""
Simple Feature Importance Visualization - Decision Tree Model
Loads saved model and displays feature importance with horizontal bar graph
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from unified_data_preprocessing import prepare_unified_data

def load_decision_tree_model():
    """Load the most recent Decision Tree model"""
    model_dir = Path(__file__).parent
    
    # Look for the most recent model file
    model_files = list(model_dir.glob("decision_tree_model_*.pkl"))
    if not model_files:
        raise FileNotFoundError("No Decision Tree model files found")
    
    # Use the most recent model (fair comparison if available)
    fair_models = [f for f in model_files if "fair" in f.name]
    if fair_models:
        model_file = fair_models[0]
    else:
        model_file = sorted(model_files)[-1]
    
    print(f"Loading model: {model_file.name}")
    
    with open(model_file, 'rb') as f:
        model_data = pickle.load(f)
    
    # Extract model and feature names from dictionary
    if isinstance(model_data, dict):
        if 'model' in model_data:
            model = model_data['model']
        elif 'best_estimator_' in model_data:
            model = model_data['best_estimator_']
        elif 'trained_model' in model_data:
            model = model_data['trained_model']
        else:
            # Assume the dictionary itself contains the model
            model = model_data
        
        # Extract feature names from saved model
        feature_names = model_data.get('feature_names', None)
    else:
        model = model_data
        feature_names = None
    
    return model, feature_names

def get_feature_names():
    """Get feature names from unified data preprocessing"""
    _, _, _, _, feature_names, _, _ = prepare_unified_data('CSV')
    return feature_names

def plot_feature_importance_horizontal(model, feature_names, top_n=15):
    """Create horizontal bar plot of feature importance"""
    
    # Get feature importance
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    else:
        raise AttributeError("Model does not have feature_importances_ attribute")
    
    # Create DataFrame for easier handling
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    })
    
    # Sort by importance and get top features
    feature_importance_df = feature_importance_df.sort_values('Importance', ascending=True)
    top_features = feature_importance_df.tail(top_n)
    
    # Create horizontal bar plot
    plt.figure(figsize=(10, 8))
    bars = plt.barh(range(len(top_features)), top_features['Importance'], 
                    color='skyblue', edgecolor='navy', alpha=0.8)
    
    # Customize plot
    plt.title(f'Decision Tree - Top {top_n} Feature Importance', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    
    # Set y-axis labels
    plt.yticks(range(len(top_features)), top_features['Feature'])
    
    # Add value labels on bars
    for i, (bar, imp) in enumerate(zip(bars, top_features['Importance'])):
        plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{imp:.3f}', ha='left', va='center', fontsize=10)
    
    # Add grid
    plt.grid(axis='x', alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    output_file = Path(__file__).parent / "feature_importance_horizontal.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Feature importance plot saved as: {output_file}")
    
    # Show plot
    plt.show()
    
    return top_features

def print_feature_ranking(top_features):
    """Print feature ranking in a nice format"""
    print("\n" + "="*60)
    print("DECISION TREE - FEATURE IMPORTANCE RANKING")
    print("="*60)
    
    # Print in ascending order (least to most important)
    for i, (idx, row) in enumerate(top_features.iterrows()):
        rank = len(top_features) - i
        print(f"{rank:2d}. {row['Feature']:<30} {row['Importance']:.4f}")
    
    print("="*60)

def main():
    """Main function to run feature importance analysis"""
    try:
        print("Decision Tree Feature Importance Analysis")
        print("-" * 50)
        
        # Load model and feature names
        model, saved_feature_names = load_decision_tree_model()
        print(f"Model type: {type(model).__name__}")
        
        # Use saved feature names if available, otherwise try unified preprocessing
        if saved_feature_names is not None:
            feature_names = saved_feature_names
            print(f"Using saved feature names: {len(feature_names)}")
        else:
            feature_names = get_feature_names()
            print(f"Using unified feature names: {len(feature_names)}")
        
        # Create and display horizontal bar plot
        top_features = plot_feature_importance_horizontal(model, feature_names, top_n=15)
        
        # Print ranking
        print_feature_ranking(top_features)
        
        print(f"\nAnalysis completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
