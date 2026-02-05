"""
Feature Engineering for Kenya Credit Scoring
=============================================
Creates derived features from alternative data
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

class KenyaFeatureEngineering:
    """
    Feature engineering for Kenya digital lending
    """
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_names = {}
        
    def create_features(self, df, is_training=True):
        """
        Create engineered features
        """
        print("="*70)
        print("FEATURE ENGINEERING")
        print("="*70)
        
        df = df.copy()
        original_cols = df.shape[1]
        
        # Encode categoricals
        categorical_cols = [
            'gender', 'location', 'mobile_money_provider',
            'employment_type', 'loan_purpose', 'internet_usage_frequency',
            'device_type'
        ]
        
        print("\n1. Encoding categorical variables...")
        for col in categorical_cols:
            if col in df.columns:
                if is_training:
                    self.encoders[col] = LabelEncoder()
                    df[f'{col}_encoded'] = self.encoders[col].fit_transform(
                        df[col].fillna('Unknown').astype(str)
                    )
                else:
                    df[f'{col}_encoded'] = df[col].fillna('Unknown').astype(str).apply(
                        lambda x: self.encoders[col].transform([x])[0]
                        if x in self.encoders[col].classes_ else -1
                    )
        
        # Create derived features
        print("2. Creating financial health features...")
        df['income_to_loan_ratio'] = df['monthly_income'] / (df['loan_amount_requested'] + 1)
        df['savings_rate'] = df['avg_monthly_savings'] / (df['monthly_income'] + 1)
        df['cash_flow_surplus'] = (df['avg_monthly_inflow'] - df['avg_monthly_outflow']) / (df['avg_monthly_inflow'] + 1)
        
        print("3. Creating mobile money features...")
        df['mm_activity_score'] = (
            df['avg_monthly_transactions'] / 100 * 0.3 +
            df['mobile_money_tenure_months'] / 36 * 0.3 +
            df['transaction_consistency'] / 100 * 0.4
        )
        df['payment_activity_score'] = (df['paybill_frequency'] + df['till_number_frequency']) / 20
        
        print("4. Creating digital engagement features...")
        df['digital_engagement_score'] = (
            df['smartphone_ownership'] * 0.3 +
            df['social_media_presence'] * 0.2 +
            df['has_used_digital_loan_before'] * 0.3 +
            (df['num_digital_loan_apps'] / 5) * 0.2
        )
        
        print("5. Creating CRB quality score...")
        df['crb_quality_score'] = np.where(
            df['has_crb_record'] == 1,
            (df['crb_score'] / 700 * 0.6 - df['crb_delinquency_flag'] * 0.4),
            0
        )
        
        print("6. Creating social capital score...")
        df['social_capital_score'] = (
            df['is_in_chama'] * 0.5 +
            df['chama_contribution_regularity'] / 100 * 0.3 +
            (df['num_mobile_contacts'] / 200) * 0.2
        )
        
        print("7. Creating loan characteristic features...")
        df['is_microloan'] = (df['loan_amount_requested'] < 5000).astype(int)
        df['is_short_term'] = (df['loan_term_requested'] <= 30).astype(int)
        
        print("8. Creating risk flags...")
        df['high_loan_to_income'] = (df['loan_to_income_ratio'] > 0.5).astype(int)
        df['new_customer_flag'] = (df['has_used_digital_loan_before'] == 0).astype(int)
        df['no_credit_history'] = (df['has_crb_record'] == 0).astype(int)
        
        # Handle missing values
        df = df.fillna(0)
        
        print(f"\nFeature engineering complete!")
        print(f"Original columns: {original_cols}")
        print(f"New columns: {df.shape[1]}")
        print(f"Features added: {df.shape[1] - original_cols}")
        
        return df
    
    def get_conversion_features(self):
        """Features for conversion model"""
        return [
            'age', 'gender_encoded', 'location_encoded',
            'smartphone_ownership', 'has_used_digital_loan_before',
            'num_digital_loan_apps', 'digital_engagement_score',
            'has_mobile_money', 'mm_activity_score', 'mobile_money_tenure_months',
            'avg_monthly_transactions', 'transaction_consistency',
            'employment_type_encoded', 'monthly_income', 'income_stability_score',
            'loan_amount_requested', 'loan_to_income_ratio', 'loan_term_requested',
            'loan_purpose_encoded', 'is_microloan', 'is_short_term',
            'is_in_chama', 'social_capital_score',
            'application_hour', 'is_weekend_application', 'is_new_device',
            'new_customer_flag'
        ]
    
    def get_repayment_features(self):
        """Features for repayment model"""
        conv_features = self.get_conversion_features()
        additional = [
            'has_crb_record', 'crb_score', 'crb_quality_score',
            'crb_delinquency_flag', 'num_credit_accounts',
            'income_to_loan_ratio', 'savings_rate', 'cash_flow_surplus',
            'has_mobile_savings', 'avg_monthly_savings',
            'paybill_frequency', 'till_number_frequency', 'payment_activity_score',
            'transaction_velocity',
            'high_loan_to_income', 'no_credit_history'
        ]
        return conv_features + additional
    
    def scale_features(self, df, feature_cols, is_training=True):
        """Scale features"""
        df = df.copy()
        
        if is_training:
            self.scalers['standard'] = StandardScaler()
            df[feature_cols] = self.scalers['standard'].fit_transform(df[feature_cols])
        else:
            df[feature_cols] = self.scalers['standard'].transform(df[feature_cols])
        
        return df
    
    def save_transformers(self, filepath='models/feature_transformers.pkl'):
        """Save encoders and scalers"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            'encoders': self.encoders,
            'scalers': self.scalers,
            'feature_names': self.feature_names
        }, filepath)
        print(f"\nTransformers saved to {filepath}")
    
    def load_transformers(self, filepath='models/feature_transformers.pkl'):
        """Load encoders and scalers"""
        loaded = joblib.load(filepath)
        self.encoders = loaded['encoders']
        self.scalers = loaded['scalers']
        self.feature_names = loaded.get('feature_names', {})
        print(f"Transformers loaded from {filepath}")

if __name__ == "__main__":
    print("="*70)
    print("STARTING FEATURE ENGINEERING PIPELINE")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    df = pd.read_csv('data/raw/kenya_digital_lending_data.csv')
    print(f"Loaded {len(df)} records")
    
    # Initialize
    fe = KenyaFeatureEngineering()
    
    # Create features
    df_engineered = fe.create_features(df, is_training=True)
    
    # Get feature lists
    conv_features = fe.get_conversion_features()
    rep_features = fe.get_repayment_features()
    
    fe.feature_names['conversion'] = conv_features
    fe.feature_names['repayment'] = rep_features
    
    print(f"\nConversion features: {len(conv_features)}")
    print(f"Repayment features: {len(rep_features)}")
    
    # Scale conversion features
    df_scaled_conv = fe.scale_features(df_engineered.copy(), conv_features, is_training=True)
    
    # Save
    output_path = 'data/processed/kenya_features_processed.csv'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_engineered.to_csv(output_path, index=False)
    print(f"\nProcessed data saved to {output_path}")
    
    # Save transformers
    fe.save_transformers()
    
    print("="*70)
    print("FEATURE ENGINEERING COMPLETE")
    print("="*70)