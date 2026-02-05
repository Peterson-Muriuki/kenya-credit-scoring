"""
Kenya Credit Scoring Dashboard
===============================
Streamlit app for credit risk assessment
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Page config
st.set_page_config(
    page_title="Kenya Credit Scoring",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {padding: 0rem 1rem;}
    .stMetric {background-color: #f0f2f6; padding: 15px; border-radius: 10px;}
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load trained models"""
    try:
        conv_model = joblib.load('models/conversion_model.pkl')
        rep_model = joblib.load('models/repayment_model.pkl')
        transformers = joblib.load('models/feature_transformers.pkl')
        return conv_model, rep_model, transformers
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.info("Please run training pipeline first")
        return None, None, None

def create_input_form():
    """Create loan application input form"""
    
    st.sidebar.header("Loan Application Details")
    
    # Demographics
    with st.sidebar.expander("Demographics", expanded=True):
        age = st.slider("Age", 18, 65, 28)
        gender = st.selectbox("Gender", ['M', 'F'])
        location = st.selectbox(
            "Location",
            ['Nairobi', 'Mombasa', 'Kisumu', 'Nakuru', 'Eldoret', 'Rural']
        )
    
    # Loan Details
    with st.sidebar.expander("Loan Details", expanded=True):
        loan_amount = st.number_input(
            "Loan Amount (KES)",
            min_value=1000, max_value=100000, value=5000, step=500
        )
        loan_term = st.selectbox("Loan Term (days)", [7, 14, 30, 60, 90])
        loan_purpose = st.selectbox(
            "Purpose",
            ['Business', 'Emergency', 'School Fees', 'Bills', 'Personal']
        )
    
    # CRB Data
    with st.sidebar.expander("CRB Data (if available)"):
        has_crb = st.checkbox("Has CRB Record", value=False)
        if has_crb:
            crb_score = st.slider("CRB Score", 300, 700, 450)
            crb_delinquent = st.checkbox("Has Delinquency", value=False)
            num_accounts = st.number_input("Credit Accounts", 0, 10, 1)
        else:
            crb_score = 0
            crb_delinquent = False
            num_accounts = 0
    
    # Mobile Money
    with st.sidebar.expander("Mobile Money", expanded=True):
        has_mm = st.checkbox("Has Mobile Money", value=True)
        if has_mm:
            mm_provider = st.selectbox("Provider", ['M-Pesa', 'Airtel Money', 'Both'])
            mm_transactions = st.slider("Monthly Transactions", 5, 200, 50)
            mm_tenure = st.slider("Account Age (months)", 3, 60, 24)
            transaction_consistency = st.slider("Consistency Score", 0, 100, 75)
        else:
            mm_provider = 'None'
            mm_transactions = 0
            mm_tenure = 0
            transaction_consistency = 0
    
    # Employment
    with st.sidebar.expander("Employment & Income"):
        employment = st.selectbox(
            "Employment Type",
            ['Formal', 'Informal', 'Self-Employed', 'Unemployed']
        )
        monthly_income = st.number_input(
            "Monthly Income (KES)",
            min_value=0, max_value=200000, value=35000, step=1000
        )
        income_stability = st.slider("Income Stability Score", 0, 100, 70)
    
    # Social Capital
    with st.sidebar.expander("Social Capital"):
        is_in_chama = st.checkbox("Member of Chama", value=True)
        if is_in_chama:
            chama_regularity = st.slider("Contribution Regularity", 0, 100, 80)
        else:
            chama_regularity = 0
        
        num_contacts = st.slider("Mobile Contacts", 10, 500, 100)
    
    # Digital Footprint
    with st.sidebar.expander("Digital Footprint"):
        smartphone = st.checkbox("Owns Smartphone", value=True)
        social_media = st.checkbox("Active on Social Media", value=True)
        used_digital_loans = st.checkbox("Used Digital Loans Before", value=False)
        num_loan_apps = st.number_input("Number of Loan Apps Used", 0, 10, 0)
    
    # Create feature dictionary
    features = {
        'age': age,
        'gender': gender,
        'location': location,
        'has_crb_record': int(has_crb),
        'crb_score': crb_score,
        'crb_delinquency_flag': int(crb_delinquent),
        'num_credit_accounts': num_accounts,
        'crb_debt_to_income': 0,
        'months_since_last_loan': 999 if not has_crb else 6,
        'has_mobile_money': int(has_mm),
        'mobile_money_provider': mm_provider,
        'avg_monthly_transactions': mm_transactions,
        'avg_transaction_value': 200,
        'mobile_money_tenure_months': mm_tenure,
        'has_mobile_savings': int(has_mm),
        'avg_monthly_savings': monthly_income * 0.1 if has_mm else 0,
        'paybill_frequency': mm_transactions // 10 if has_mm else 0,
        'till_number_frequency': mm_transactions // 5 if has_mm else 0,
        'avg_monthly_inflow': monthly_income,
        'avg_monthly_outflow': monthly_income * 0.9,
        'transaction_velocity': 0.1,
        'transaction_consistency': transaction_consistency,
        'employment_type': employment,
        'monthly_income': monthly_income,
        'income_stability_score': income_stability,
        'smartphone_ownership': int(smartphone),
        'internet_usage_frequency': 'Daily' if smartphone else 'Rarely',
        'social_media_presence': int(social_media),
        'has_used_digital_loan_before': int(used_digital_loans),
        'num_digital_loan_apps': num_loan_apps,
        'is_in_chama': int(is_in_chama),
        'chama_contribution_regularity': chama_regularity,
        'num_mobile_contacts': num_contacts,
        'avg_call_duration': 3,
        'loan_amount_requested': loan_amount,
        'loan_purpose': loan_purpose,
        'loan_term_requested': loan_term,
        'loan_to_income_ratio': loan_amount / monthly_income if monthly_income > 0 else 0,
        'application_hour': 14,
        'is_weekend_application': 0,
        'device_type': 'Smartphone' if smartphone else 'Feature Phone',
        'is_new_device': 0,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    return pd.DataFrame([features])

def engineer_features(df, transformers):
    """Apply feature engineering"""
    from src.feature_engineering import KenyaFeatureEngineering
    
    fe = KenyaFeatureEngineering()
    fe.encoders = transformers['encoders']
    fe.scalers = transformers['scalers']
    fe.feature_names = transformers['feature_names']
    
    df_engineered = fe.create_features(df, is_training=False)
    
    return df_engineered, fe

def predict_scores(conv_model, rep_model, df_engineered, fe):
    """Make predictions"""
    
    # Get features
    conv_features = fe.feature_names['conversion']
    rep_features = fe.feature_names['repayment']
    
    # Scale features
    X_conv = df_engineered[conv_features]
    X_rep = df_engineered[rep_features]
    
    # Predict
    conv_prob = conv_model.predict_proba(X_conv)[0, 1]
    rep_prob = rep_model.predict_proba(X_rep)[0, 1]
    
    # Hybrid score (40% conversion, 60% repayment - Kenya optimal)
    hybrid = conv_prob * 0.4 + rep_prob * 0.6
    
    return conv_prob, rep_prob, hybrid

def display_gauge(value, title):
    """Display gauge chart"""
    
    if value >= 0.95:
        color = "green"
    elif value >= 0.85:
        color = "lightgreen"
    elif value >= 0.75:
        color = "yellow"
    else:
        color = "red"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 20}},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 75], 'color': 'rgba(255, 0, 0, 0.2)'},
                {'range': [75, 85], 'color': 'rgba(255, 255, 0, 0.2)'},
                {'range': [85, 95], 'color': 'rgba(144, 238, 144, 0.2)'},
                {'range': [95, 100], 'color': 'rgba(0, 128, 0, 0.2)'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 95
            }
        }
    ))
    
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def main():
    """Main application"""
    
    # Title
    st.title("Kenya Credit Scoring System")
    st.markdown("### Alternative Data-Driven Credit Assessment")
    st.markdown("---")
    
    # Load models
    conv_model, rep_model, transformers = load_models()
    
    if conv_model is None:
        st.stop()
    
    # Input form
    input_df = create_input_form()
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs([
        "Risk Assessment",
        "Application Details",
        "About System"
    ])
    
    with tab1:
        st.header("Credit Risk Assessment")
        
        if st.button("Analyze Application", type="primary", use_container_width=True):
            with st.spinner("Analyzing..."):
                # Engineer features
                df_engineered, fe = engineer_features(input_df, transformers)
                
                # Predict
                conv_prob, rep_prob, hybrid = predict_scores(
                    conv_model, rep_model, df_engineered, fe
                )
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Conversion Probability",
                        f"{conv_prob*100:.1f}%",
                        help="Likelihood customer accepts loan offer"
                    )
                
                with col2:
                    st.metric(
                        "Repayment Probability",
                        f"{rep_prob*100:.1f}%",
                        delta=f"{(rep_prob-0.95)*100:.1f}% vs 95% target"
                    )
                
                with col3:
                    st.metric(
                        "Hybrid Score",
                        f"{hybrid*100:.1f}%",
                        help="Combined conversion + repayment score"
                    )
                
                st.markdown("---")
                
                # Gauges
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = display_gauge(conv_prob, "Conversion Score")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = display_gauge(rep_prob, "Repayment Score")
                    st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")
                
                # Recommendation
                st.subheader("Lending Decision")
                
                if rep_prob >= 0.95:
                    st.success("RECOMMENDATION: APPROVE")
                    st.write("Excellent repayment probability. Low risk customer.")
                    rec_amount = input_df['loan_amount_requested'].values[0]
                    rec_term = input_df['loan_term_requested'].values[0]
                elif rep_prob >= 0.90:
                    st.success("RECOMMENDATION: APPROVE")
                    st.write("Good repayment probability. Consider approved amount.")
                    rec_amount = input_df['loan_amount_requested'].values[0] * 0.8
                    rec_term = min(input_df['loan_term_requested'].values[0], 30)
                elif rep_prob >= 0.85:
                    st.warning("RECOMMENDATION: REVIEW")
                    st.write("Fair repayment probability. Reduce loan amount and term.")
                    rec_amount = input_df['loan_amount_requested'].values[0] * 0.6
                    rec_term = min(input_df['loan_term_requested'].values[0], 14)
                else:
                    st.error("RECOMMENDATION: DECLINE")
                    st.write("High default risk. Not recommended for lending.")
                    rec_amount = 0
                    rec_term = 0
                
                if rep_prob >= 0.85:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Recommended Amount", f"KES {rec_amount:,.0f}")
                    with col2:
                        st.metric("Recommended Term", f"{rec_term} days")
                
                # Risk factors
                st.markdown("---")
                st.subheader("Key Decision Factors")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Positive Factors**")
                    if input_df['has_mobile_money'].values[0]:
                        st.write("- Has mobile money account")
                    if input_df['is_in_chama'].values[0]:
                        st.write("- Member of Chama")
                    if input_df['has_crb_record'].values[0] and not input_df['crb_delinquency_flag'].values[0]:
                        st.write("- Clean CRB record")
                    if input_df['employment_type'].values[0] == 'Formal':
                        st.write("- Formal employment")
                
                with col2:
                    st.markdown("**Risk Factors**")
                    if input_df['loan_to_income_ratio'].values[0] > 0.5:
                        st.write("- High loan-to-income ratio")
                    if not input_df['has_mobile_money'].values[0]:
                        st.write("- No mobile money history")
                    if input_df['crb_delinquency_flag'].values[0]:
                        st.write("- CRB delinquency flag")
                    if input_df['employment_type'].values[0] == 'Unemployed':
                        st.write("- Unemployed")
    
    with tab2:
        st.header("Application Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Age", f"{input_df['age'].values[0]} years")
            st.metric("Location", input_df['location'].values[0])
            st.metric("Employment", input_df['employment_type'].values[0])
        
        with col2:
            st.metric("Monthly Income", f"KES {input_df['monthly_income'].values[0]:,.0f}")
            st.metric("Loan Requested", f"KES {input_df['loan_amount_requested'].values[0]:,.0f}")
            st.metric("Loan Purpose", input_df['loan_purpose'].values[0])
        
        with col3:
            st.metric("Mobile Money", "Yes" if input_df['has_mobile_money'].values[0] else "No")
            st.metric("CRB Record", "Yes" if input_df['has_crb_record'].values[0] else "No")
            st.metric("Chama Member", "Yes" if input_df['is_in_chama'].values[0] else "No")
        
        st.markdown("---")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Income vs Loan Amount")
            data = pd.DataFrame({
                'Category': ['Monthly Income', 'Loan Requested'],
                'Amount': [
                    input_df['monthly_income'].values[0],
                    input_df['loan_amount_requested'].values[0]
                ]
            })
            fig = px.bar(data, x='Category', y='Amount', color='Category')
            fig.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Data Availability")
            data = pd.DataFrame({
                'Source': ['CRB', 'Mobile Money', 'Chama', 'Employment'],
                'Available': [
                    int(input_df['has_crb_record'].values[0]),
                    int(input_df['has_mobile_money'].values[0]),
                    int(input_df['is_in_chama'].values[0]),
                    1 if input_df['employment_type'].values[0] != 'Unemployed' else 0
                ]
            })
            fig = px.bar(data, x='Source', y='Available', color='Available')
            fig.update_layout(showlegend=False, yaxis_range=[0, 1], height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("About This System")
        
        st.markdown("""
        ### Purpose
        
        This credit scoring system is specifically designed for Kenya's digital lending market,
        addressing the limitations of traditional CRB-only scoring.
        
        ### Data Sources
        
        **Primary Sources (Kenya-specific):**
        - M-Pesa and Airtel Money transaction history
        - Chama (savings group) participation
        - Mobile phone usage patterns
        - Social capital indicators
        
        **Secondary Sources:**
        - CRB credit bureau data (when available)
        - Employment verification
        - Digital footprint
        
        ### Dual Scoring Approach
        
        **Conversion Model:**
        - Predicts likelihood customer accepts loan offer
        - Optimizes for customer acquisition
        
        **Repayment Model:**
        - Predicts probability of on-time repayment
        - Optimizes for portfolio quality
        
        **Hybrid Score:**
        - Combines both models (40% conversion, 60% repayment)
        - Optimized for Kenya market dynamics
        
        ### Performance Targets
        
        - Repayment Rate: >95%
        - Default Rate: <3%
        - Conversion Rate: Optimized for each lender
        
        ### Coverage
        
        - CRB-only systems: 30-40% of population
        - This system: 90%+ of population
        - Specifically serves unbanked/underbanked customers
        
        ### Technical Details
        
        - Models: XGBoost ensemble
        - Features: 40+ derived features
        - Training: 15,000+ loan applications
        - Validation: Stratified cross-validation
        
        ### Contact
        
        For pilot program inquiries or integration support:
        - Email: contact@kenyacreditscoring.com
        - Website: www.kenyacreditscoring.com
        """)

if __name__ == "__main__":
    main()