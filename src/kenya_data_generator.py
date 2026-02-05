"""
Kenya Digital Lending Data Generator
=====================================
Generates realistic loan application data for Kenya market
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

# Set seed
np.random.seed(42)
random.seed(42)

def generate_kenya_lending_data(n_samples=15000):
    """
    Generate synthetic loan applications for Kenya digital lending market
    
    Features:
    - CRB data (30-40% coverage)
    - M-Pesa and Airtel Money (90%+ penetration)
    - Chama (savings groups) participation
    - Employment types (Formal, Informal, Self-employed)
    - Kenya-specific locations and demographics
    """
    
    print("="*70)
    print("GENERATING KENYA DIGITAL LENDING DATA")
    print("="*70)
    print(f"\nGenerating {n_samples} loan applications...")
    
    # Kenya locations and weights
    locations = {
        'Nairobi': 0.30,
        'Mombasa': 0.15,
        'Kisumu': 0.10,
        'Nakuru': 0.10,
        'Eldoret': 0.10,
        'Rural': 0.25
    }
    
    # Employment types
    employment_types = {
        'Formal': 0.20,
        'Informal': 0.40,
        'Self-Employed': 0.30,
        'Unemployed': 0.10
    }
    
    # Loan purposes
    loan_purposes = {
        'Business': 0.30,
        'Emergency': 0.25,
        'School Fees': 0.15,
        'Bills': 0.20,
        'Personal': 0.10
    }
    
    data = []
    
    for i in range(n_samples):
        # Demographics
        age = max(18, min(65, int(np.random.normal(32, 8))))
        gender = random.choice(['M', 'F'])
        location = random.choices(
            list(locations.keys()),
            weights=list(locations.values())
        )[0]
        
        # CRB Data (only 35% have records)
        has_crb_record = random.random() < 0.35
        
        if has_crb_record:
            crb_score = int(np.clip(np.random.normal(450, 100), 300, 700))
            num_credit_accounts = np.random.poisson(2)
            crb_delinquency_flag = random.random() < 0.15
            crb_debt_to_income = np.random.beta(2, 5)
            months_since_last_loan = max(0, int(np.random.exponential(6)))
        else:
            crb_score = 0
            num_credit_accounts = 0
            crb_delinquency_flag = False
            crb_debt_to_income = 0
            months_since_last_loan = 999
        
        # Mobile Money (92% penetration in Kenya)
        has_mobile_money = random.random() < 0.92
        
        if has_mobile_money:
            mobile_money_provider = random.choices(
                ['M-Pesa', 'Airtel Money', 'Both'],
                weights=[0.70, 0.20, 0.10]
            )[0]
            
            avg_monthly_transactions = max(5, int(np.random.exponential(40)))
            avg_transaction_value = np.random.lognormal(5.5, 1.5)
            mobile_money_tenure_months = max(3, int(np.random.exponential(24)))
            
            has_mobile_savings = random.random() < 0.40
            avg_monthly_savings = np.random.lognormal(6, 1.2) if has_mobile_savings else 0
            
            paybill_frequency = max(0, int(np.random.exponential(5)))
            till_number_frequency = max(0, int(np.random.exponential(10)))
            
            avg_monthly_inflow = np.random.lognormal(9, 0.8)
            avg_monthly_outflow = avg_monthly_inflow * np.random.uniform(0.85, 1.05)
            
            transaction_velocity = np.random.uniform(-0.2, 0.3)
            transaction_consistency = np.random.beta(5, 2) * 100
        else:
            mobile_money_provider = 'None'
            avg_monthly_transactions = 0
            avg_transaction_value = 0
            mobile_money_tenure_months = 0
            has_mobile_savings = False
            avg_monthly_savings = 0
            paybill_frequency = 0
            till_number_frequency = 0
            avg_monthly_inflow = 0
            avg_monthly_outflow = 0
            transaction_velocity = 0
            transaction_consistency = 0
        
        # Employment
        employment_type = random.choices(
            list(employment_types.keys()),
            weights=list(employment_types.values())
        )[0]
        
        monthly_income_base = {
            'Formal': 10.5,
            'Informal': 9.5,
            'Self-Employed': 9.8,
            'Unemployed': 8.0
        }[employment_type]
        
        monthly_income = np.random.lognormal(monthly_income_base, 0.6)
        
        income_stability_score_params = {
            'Formal': (7, 2),
            'Informal': (4, 4),
            'Self-Employed': (5, 3),
            'Unemployed': (2, 5)
        }[employment_type]
        
        income_stability_score = np.random.beta(*income_stability_score_params) * 100
        
        # Digital Footprint
        smartphone_ownership = random.random() < 0.75
        internet_usage = random.choice(['Daily', 'Weekly', 'Monthly', 'Rarely'])
        social_media_presence = random.random() < 0.65
        has_used_digital_loan_before = random.random() < 0.45
        num_digital_loan_apps = np.random.poisson(2) if has_used_digital_loan_before else 0
        
        # Social Capital - Chama participation
        is_in_chama = random.random() < 0.50
        chama_contribution_regularity = np.random.beta(5, 2) * 100 if is_in_chama else 0
        
        num_mobile_contacts = max(10, int(np.random.exponential(100)))
        avg_call_duration = np.random.exponential(3)
        
        # Loan Request
        loan_amount_requested = np.clip(np.random.lognormal(8, 1.2), 1000, 100000)
        
        loan_purpose = random.choices(
            list(loan_purposes.keys()),
            weights=list(loan_purposes.values())
        )[0]
        
        loan_term_requested = random.choice([7, 14, 30, 60, 90])
        
        # Behavioral signals
        application_hour = int(np.clip(np.random.normal(14, 4), 0, 23))
        is_weekend_application = random.random() < 0.20
        device_type = 'Smartphone' if smartphone_ownership else 'Feature Phone'
        is_new_device = random.random() < 0.15
        
        # Calculate Conversion Probability
        conv_base = (
            (1 if has_mobile_money else 0) * 0.20 +
            (1 if smartphone_ownership else 0) * 0.15 +
            (loan_amount_requested / monthly_income if monthly_income > 0 else 0) * 0.15 +
            (1 if has_used_digital_loan_before else 0) * 0.20 +
            (income_stability_score / 100) * 0.15 +
            (1 if is_in_chama else 0) * 0.10 +
            (0.5 if loan_purpose in ['Emergency', 'Bills'] else 0.25) * 0.05
        )
        
        conversion_probability = np.clip(conv_base * np.random.uniform(0.8, 1.2), 0.05, 0.98)
        converted = random.random() < conversion_probability
        
        # Calculate Repayment Probability (if converted)
        if converted:
            rep_base = 0.5
            
            # CRB contribution
            if has_crb_record:
                crb_contrib = (crb_score - 300) / 400 * 0.25
                if crb_delinquency_flag:
                    crb_contrib *= 0.5
                rep_base += crb_contrib
            
            # Mobile money contribution
            if has_mobile_money:
                mm_contrib = (
                    (transaction_consistency / 100) * 0.15 +
                    min(mobile_money_tenure_months / 36, 1) * 0.10 +
                    ((avg_monthly_inflow - avg_monthly_outflow) / avg_monthly_inflow if avg_monthly_inflow > 0 else 0) * 0.10 +
                    (1 if transaction_velocity > 0 else 0) * 0.05
                )
                rep_base += mm_contrib
            
            # Employment contribution
            emp_contrib = (
                (income_stability_score / 100) * 0.10 +
                {'Formal': 0.10, 'Self-Employed': 0.05, 'Informal': 0.025, 'Unemployed': 0}[employment_type]
            )
            rep_base += emp_contrib
            
            # Social capital
            social_contrib = (
                (1 if is_in_chama else 0) * 0.08 +
                (chama_contribution_regularity / 100) * 0.05
            )
            rep_base += social_contrib
            
            # Loan characteristics
            loan_contrib = (
                (1 if loan_amount_requested / monthly_income < 0.3 else 0) * 0.07 +
                (1 if loan_term_requested <= 30 else 0) * 0.05
            )
            rep_base += loan_contrib
            
            repayment_probability = np.clip(rep_base * np.random.uniform(0.85, 1.15), 0.60, 0.99)
            repaid_on_time = random.random() < repayment_probability
        else:
            repayment_probability = 0
            repaid_on_time = -1
        
        # Create record
        record = {
            'customer_id': f'CUS_{i+1:06d}',
            'age': age,
            'gender': gender,
            'location': location,
            
            # CRB
            'has_crb_record': int(has_crb_record),
            'crb_score': crb_score,
            'num_credit_accounts': num_credit_accounts,
            'crb_delinquency_flag': int(crb_delinquency_flag),
            'crb_debt_to_income': round(crb_debt_to_income, 4),
            'months_since_last_loan': months_since_last_loan,
            
            # Mobile Money
            'has_mobile_money': int(has_mobile_money),
            'mobile_money_provider': mobile_money_provider,
            'avg_monthly_transactions': avg_monthly_transactions,
            'avg_transaction_value': round(avg_transaction_value, 2),
            'mobile_money_tenure_months': mobile_money_tenure_months,
            'has_mobile_savings': int(has_mobile_savings),
            'avg_monthly_savings': round(avg_monthly_savings, 2),
            'paybill_frequency': paybill_frequency,
            'till_number_frequency': till_number_frequency,
            'avg_monthly_inflow': round(avg_monthly_inflow, 2),
            'avg_monthly_outflow': round(avg_monthly_outflow, 2),
            'transaction_velocity': round(transaction_velocity, 3),
            'transaction_consistency': round(transaction_consistency, 2),
            
            # Employment
            'employment_type': employment_type,
            'monthly_income': round(monthly_income, 2),
            'income_stability_score': round(income_stability_score, 2),
            
            # Digital
            'smartphone_ownership': int(smartphone_ownership),
            'internet_usage_frequency': internet_usage,
            'social_media_presence': int(social_media_presence),
            'has_used_digital_loan_before': int(has_used_digital_loan_before),
            'num_digital_loan_apps': num_digital_loan_apps,
            
            # Social
            'is_in_chama': int(is_in_chama),
            'chama_contribution_regularity': round(chama_contribution_regularity, 2),
            'num_mobile_contacts': num_mobile_contacts,
            'avg_call_duration': round(avg_call_duration, 2),
            
            # Loan
            'loan_amount_requested': round(loan_amount_requested, 2),
            'loan_purpose': loan_purpose,
            'loan_term_requested': loan_term_requested,
            'loan_to_income_ratio': round(loan_amount_requested / monthly_income if monthly_income > 0 else 0, 4),
            
            # Behavioral
            'application_hour': application_hour,
            'is_weekend_application': int(is_weekend_application),
            'device_type': device_type,
            'is_new_device': int(is_new_device),
            
            # Outcomes
            'conversion_probability': round(conversion_probability, 4),
            'converted': int(converted),
            'repayment_probability': round(repayment_probability, 4) if converted else 0,
            'repaid_on_time': int(repaid_on_time) if repaid_on_time != -1 else -1,
            
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        data.append(record)
    
    df = pd.DataFrame(data)
    
    # Statistics
    print(f"\nDataset generated successfully!")
    print(f"Total samples: {len(df)}")
    print(f"Conversion Rate: {df['converted'].mean()*100:.1f}%")
    
    repaid = df[df['repaid_on_time'] != -1]
    if len(repaid) > 0:
        print(f"Repayment Rate: {repaid['repaid_on_time'].mean()*100:.1f}%")
    
    print(f"\nMobile Money Coverage: {df['has_mobile_money'].mean()*100:.1f}%")
    print(f"CRB Coverage: {df['has_crb_record'].mean()*100:.1f}%")
    print(f"Chama Participation: {df['is_in_chama'].mean()*100:.1f}%")
    
    print(f"\nLocation Distribution:")
    print(df['location'].value_counts().to_string())
    
    return df

if __name__ == "__main__":
    # Generate data
    df = generate_kenya_lending_data(n_samples=15000)
    
    # Save
    output_path = 'data/raw/kenya_digital_lending_data.csv'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"\nData saved to: {output_path}")
    print("="*70)
    print("DATA GENERATION COMPLETE")
    print("="*70)