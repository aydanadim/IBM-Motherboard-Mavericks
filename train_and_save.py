import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

print("Generating balanced synthetic transaction dataset...")
np.random.seed(42)
n_samples = 10000

steps = np.random.randint(1, 744, size=n_samples)
type_codes = np.random.randint(0, 5, size=n_samples)
amounts = np.random.uniform(10.0, 500000.0, size=n_samples)
old_org = np.random.uniform(0.0, 600000.0, size=n_samples)

is_fraud = np.zeros(n_samples, dtype=int)
new_org = old_org - amounts
new_org[new_org < 0] = 0

old_dest = np.random.uniform(0.0, 200000.0, size=n_samples)
new_dest = old_dest + amounts

# Inject positive fraud signatures (account drain with unadjusted destination balance)
fraud_indices = np.random.choice(n_samples, size=1000, replace=False)
for idx in fraud_indices:
    is_fraud[idx] = 1
    old_org[idx] = amounts[idx]
    new_org[idx] = 0.0
    old_dest[idx] = 5000.0
    new_dest[idx] = 5000.0  # Mismatch signature

X = pd.DataFrame({
    'step': steps,
    'type_code': type_codes,
    'amount': amounts,
    'oldbalanceOrg': old_org,
    'newbalanceOrig': new_org,
    'oldbalanceDest': old_dest,
    'newbalanceDest': new_dest
})
y = is_fraud

print("Training model with class_weight='balanced'...")
model = RandomForestClassifier(n_estimators=50, class_weight='balanced', random_state=42)
model.fit(X, y)

joblib.dump(model, 'fraud_model.joblib')
print("Successfully generated and saved new 'fraud_model.joblib'!")
