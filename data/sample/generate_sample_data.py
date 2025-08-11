"""
Generate sample customer churn dataset for demonstration
"""
import pandas as pd
import numpy as np
from pathlib import Path

def generate_sample_data(n_samples=1000):
    """Generate sample customer churn dataset"""
    np.random.seed(42)
    
    # Customer demographics
    age = np.random.normal(40, 15, n_samples).astype(int)
    age = np.clip(age, 18, 80)
    
    # Account information
    tenure = np.random.exponential(2, n_samples) * 12  # months
    tenure = np.clip(tenure, 1, 72).astype(int)
    
    monthly_charges = np.random.normal(65, 20, n_samples)
    monthly_charges = np.clip(monthly_charges, 20, 150)
    
    total_charges = monthly_charges * tenure + np.random.normal(0, 100, n_samples)
    total_charges = np.maximum(total_charges, monthly_charges)
    
    # Services
    internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.4, 0.4, 0.2])
    contract = np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.5, 0.3, 0.2])
    payment_method = np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], 
                                    n_samples, p=[0.3, 0.2, 0.25, 0.25])
    
    # Introduce some missing values and outliers for ETL to handle
    missing_mask = np.random.random(n_samples) < 0.05
    total_charges[missing_mask] = np.nan
    
    # Add some outliers
    outlier_mask = np.random.random(n_samples) < 0.02
    monthly_charges[outlier_mask] *= 5
    
    # Add some duplicate rows
    duplicate_indices = np.random.choice(n_samples, size=int(n_samples * 0.03), replace=False)
    
    # Target variable (churn) - influenced by features
    churn_prob = (
        0.1 +  # base probability
        0.3 * (contract == 'Month-to-month') +
        0.2 * (tenure < 12) +
        0.15 * (monthly_charges > 80) +
        0.1 * (payment_method == 'Electronic check') +
        0.05 * (age < 30)
    )
    churn = np.random.binomial(1, churn_prob, n_samples)
    
    # Create DataFrame
    data = pd.DataFrame({
        'customer_id': [f'CUST_{i:06d}' for i in range(n_samples)],
        'age': age,
        'tenure': tenure,
        'monthly_charges': monthly_charges,
        'total_charges': total_charges,
        'internet_service': internet_service,
        'contract': contract,
        'payment_method': payment_method,
        'target': churn  # 1 = churned, 0 = retained
    })
    
    # Add duplicate rows
    duplicate_rows = data.iloc[duplicate_indices].copy()
    data = pd.concat([data, duplicate_rows], ignore_index=True)
    
    return data

if __name__ == "__main__":
    # Create directories
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    
    # Generate and save sample data
    sample_data = generate_sample_data(1000)
    sample_data.to_csv("data/raw/customer_data.csv", index=False)
    
    print(f"Generated sample dataset with {len(sample_data)} rows")
    print(f"Columns: {list(sample_data.columns)}")
    print(f"Missing values: {sample_data.isnull().sum().sum()}")
    print(f"Target distribution: {sample_data['target'].value_counts().to_dict()}")