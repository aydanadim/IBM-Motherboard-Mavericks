import time
import numpy as np
import joblib

model = joblib.load('fraud_model.joblib')
sample_data = np.array([[1, 2, 45000.0, 50000.0, 5000.0, 0.0, 45000.0]], dtype=np.float32)

# Warmup
model.predict_proba(sample_data)

# Measure 1,000 rapid inferences
start_time = time.perf_counter()
iterations = 1000
for _ in range(iterations):
    model.predict_proba(sample_data)
end_time = time.perf_counter()

total_time_ms = (end_time - start_time) * 1000
per_inference_ms = total_time_ms / iterations

print(f"Total time for {iterations} inferences: {total_time_ms:.2f} ms")
print(f"Average Core Inference Latency: {per_inference_ms:.4f} milliseconds per transaction!")
