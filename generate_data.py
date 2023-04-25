import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(123)

# Define number of samples and range of values for each feature
n_samples = 1000
usage_minutes_range = (20, 120)
data_usage_gb_range = (1, 10)
payment_history_days_range = (30, 90)
support_calls_range = (1, 10)

# Generate random values for each feature
usage_minutes = np.random.randint(*usage_minutes_range, size=n_samples)
data_usage_gb = np.random.randint(*data_usage_gb_range, size=n_samples)
payment_history_days = np.random.randint(*payment_history_days_range, size=n_samples)
support_calls = np.random.randint(*support_calls_range, size=n_samples)

# Generate target variable based on random probabilities
churn_prob = 1 / (1 + np.exp(-(0.2 * usage_minutes - 0.1 * data_usage_gb + 0.05 * payment_history_days - 0.1 * support_calls - 3)))
target = np.random.binomial(n=1, p=churn_prob, size=n_samples)

# Create pandas dataframe with generated data
data = pd.DataFrame({'usage_minutes': usage_minutes, 'data_usage_gb': data_usage_gb,
                     'payment_history_days': payment_history_days, 'support_calls': support_calls,
                     'target': target})

# Save dataframe to CSV file
data.to_csv('data.csv', index=False)