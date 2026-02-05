"""
Dual Scoring Model Trainer
===========================
Trains conversion and repayment models
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, roc_auc_score,
                            confusion_matrix, roc_curve, auc)
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import joblib
import os
import sys

# Add src to path
sys.path.append(os.path.dirname(__file__))
from feature_engineering import KenyaFeatureEngineering

class DualScoringTrainer:
    """
    Train conversion and repayment models
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.conversion_model = None
        self.repayment_model = None
        
    def train_conversion_model(self, X_train, y_train, X_test, y_test):
        """Train conversion prediction model"""
        print("="*70)
        print("TRAINING CONVERSION MODEL")
        print("="*70)
        
        print(f"\nTraining samples: {len(X_train)}")
        print(f"Conversion rate: {y_train.mean()*100:.1f}%")
        
        model = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            eval_metric='logloss'
        )
        
        print("\nFitting model...")
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        
        # Evaluate
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        accuracy = (y_pred == y_test).mean()
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"\nRESULTS:")
        print(f"Accuracy: {accuracy*100:.2f}%")
        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"Predicted conversion rate: {y_pred.mean()*100:.1f}%")
        
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        print(f"\nConfusion Matrix:")
        print(f"TN: {tn}, FP: {fp}")
        print(f"FN: {fn}, TP: {tp}")
        
        self.conversion_model = model
        return model, roc_auc
    
    def train_repayment_model(self, X_train, y_train, X_test, y_test):
        """Train repayment prediction model"""
        print("\n" + "="*70)
        print("TRAINING REPAYMENT MODEL")
        print("="*70)
        
        print(f"\nOriginal repayment rate: {y_train.mean()*100:.1f}%")
        print("Applying SMOTE...")
        
        smote = SMOTE(random_state=self.random_state, sampling_strategy=0.3)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        print(f"Balanced repayment rate: {y_train_balanced.mean()*100:.1f}%")
        
        scale_pos_weight = (y_train_balanced == 0).sum() / (y_train_balanced == 1).sum()
        
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            eval_metric='logloss'
        )
        
        print("\nFitting model...")
        model.fit(
            X_train_balanced, y_train_balanced,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # Evaluate
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        repayment_rate = (tn + tp) / (tn + fp + fn + tp)
        default_rate = 1 - repayment_rate
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"\nRESULTS:")
        print(f"Repayment Rate: {repayment_rate*100:.2f}%")
        print(f"Default Rate: {default_rate*100:.2f}%")
        print(f"ROC-AUC: {roc_auc:.4f}")
        
        meets_target = repayment_rate >= 0.95 and default_rate <= 0.03
        print(f"\nTarget Metrics:")
        print(f"{'PASS' if repayment_rate >= 0.95 else 'FAIL'} Repayment >95%: {repayment_rate*100:.2f}%")
        print(f"{'PASS' if default_rate <= 0.03 else 'FAIL'} Default <3%: {default_rate*100:.2f}%")
        
        self.repayment_model = model
        return model, repayment_rate, default_rate
    
    def save_models(self, prefix='models/'):
        """Save models"""
        os.makedirs(prefix, exist_ok=True)
        
        joblib.dump(self.conversion_model, f'{prefix}conversion_model.pkl')
        joblib.dump(self.repayment_model, f'{prefix}repayment_model.pkl')
        
        print(f"\nModels saved to {prefix}")

if __name__ == "__main__":
    print("="*70)
    print("DUAL SCORING MODEL TRAINING")
    print("="*70)
    
    # Load data
    print("\nLoading processed data...")
    df = pd.read_csv('data/processed/kenya_features_processed.csv')
    print(f"Loaded {len(df)} records")
    
    # Load feature engineering
    fe = KenyaFeatureEngineering()
    fe.load_transformers()
    
    # Get feature lists
    conv_features = fe.feature_names['conversion']
    rep_features = fe.feature_names['repayment']
    
    # Initialize trainer
    trainer = DualScoringTrainer()
    
    # PART 1: Train Conversion Model
    print("\n" + "="*70)
    print("PART 1: CONVERSION MODEL")
    print("="*70)
    
    X_conv = df[conv_features]
    y_conv = df['converted']
    
    X_conv_train, X_conv_test, y_conv_train, y_conv_test = train_test_split(
        X_conv, y_conv, test_size=0.2, random_state=42, stratify=y_conv
    )
    
    conv_model, conv_auc = trainer.train_conversion_model(
        X_conv_train, y_conv_train, X_conv_test, y_conv_test
    )
    
    # PART 2: Train Repayment Model
    print("\n" + "="*70)
    print("PART 2: REPAYMENT MODEL")
    print("="*70)
    
    # Filter to converted customers only
    df_repayment = df[
        (df['converted'] == 1) &
        (df['repaid_on_time'] != -1)
    ].copy()
    
    print(f"\nFiltered to {len(df_repayment)} converted customers")
    
    X_rep = df_repayment[rep_features]
    y_rep = df_repayment['repaid_on_time']
    
    X_rep_train, X_rep_test, y_rep_train, y_rep_test = train_test_split(
        X_rep, y_rep, test_size=0.2, random_state=42, stratify=y_rep
    )
    
    rep_model, rep_rate, def_rate = trainer.train_repayment_model(
        X_rep_train, y_rep_train, X_rep_test, y_rep_test
    )
    
    # Save models
    trainer.save_models()
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"\nFinal Metrics:")
    print(f"Conversion AUC: {conv_auc:.4f}")
    print(f"Repayment Rate: {rep_rate*100:.2f}%")
    print(f"Default Rate: {def_rate*100:.2f}%")