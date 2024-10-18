import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

def generate_sample_data(num_records=1000, output_path='data/raw/credit_data.csv'):
    """Generate synthetic consumer credit data for testing the pipeline."""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate customer IDs
    customer_ids = [f'CUST{str(i).zfill(6)}' for i in range(num_records)]
    loan_ids = [f'LOAN{str(i).zfill(8)}' for i in range(num_records)]
    
    # Generate dates
    base_date = datetime(2023, 1, 1)
    start_dates = [base_date + timedelta(days=random.randint(0, 365)) for _ in range(num_records)]
    loan_durations = np.random.choice([180, 365, 730, 1095], size=num_records)  # 6 months, 1,2,3 years
    end_dates = [start + timedelta(days=duration) for start, duration in zip(start_dates, loan_durations)]
    
    # Generate numerical data with realistic distributions
    loan_amounts = np.random.lognormal(mean=9, sigma=1, size=num_records)  # Centers around £8000
    interest_rates = np.random.normal(loc=5, scale=2, size=num_records)  # Average 5% APR
    credit_scores = np.random.normal(loc=700, scale=100, size=num_records)  # Credit scores
    debt_to_income = np.random.normal(loc=0.4, scale=0.2, size=num_records)  # DTI ratio
    payment_history = np.random.normal(loc=85, scale=15, size=num_records)  # Payment history score
    
    # Generate categorical data
    payment_frequencies = np.random.choice(['MONTHLY', 'WEEKLY', 'BIWEEKLY'], size=num_records, p=[0.7, 0.1, 0.2])
    loan_purposes = np.random.choice(
        ['DEBT_CONSOLIDATION', 'HOME_IMPROVEMENT', 'MAJOR_PURCHASE', 'BUSINESS', 'EDUCATION'],
        size=num_records,
        p=[0.4, 0.2, 0.2, 0.1, 0.1]
    )
    
    # Create DataFrame
    df = pd.DataFrame({
        'loan_id': loan_ids,
        'customer_id': customer_ids,
        'loan_amount': loan_amounts.round(2),
        'interest_rate': interest_rates.round(2).clip(1, 15),  # Cap between 1% and 15%
        'start_date': start_dates,
        'end_date': end_dates,
        'payment_frequency': payment_frequencies,
        'loan_purpose': loan_purposes,
        'credit_score': credit_scores.round().clip(300, 850),  # Standard credit score range
        'debt_to_income_ratio': debt_to_income.round(3).clip(0, 1),
        'payment_history_score': payment_history.round().clip(0, 100)
    })
    
    # Add some data quality issues for testing
    # Add some nulls
    mask = np.random.choice([True, False], size=len(df), p=[0.02, 0.98])  # 2% null rate
    df.loc[mask, 'credit_score'] = None
    
    # Add some outliers
    outlier_mask = np.random.choice([True, False], size=len(df), p=[0.01, 0.99])  # 1% outlier rate
    df.loc[outlier_mask, 'loan_amount'] *= 5
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Generated {num_records} records of sample data at {output_path}")
    
    # Save a small sample as JSON for documentation
    df.head(3).to_json('data/sample_data.json', orient='records', indent=2)

if __name__ == "__main__":
    generate_sample_data()
