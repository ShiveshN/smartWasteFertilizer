import pandas as pd
import random

# Function to generate a balanced dataset
def generate_balanced_sensor_data(num_samples=2000):
    data = []
    threshold = 0.7 * 1023  # 70% of the sensor range

    # Generate half samples where FertilizerConverted is 'Yes'
    positive_samples = num_samples // 2
    while len(data) < positive_samples:
        sensor_values = [random.randint(0, 1023) for _ in range(4)]
        if sum(s > threshold for s in sensor_values) >= 2:
            data.append({
                "Sensor1": sensor_values[0],
                "Sensor2": sensor_values[1],
                "Sensor3": sensor_values[2],
                "Sensor4": sensor_values[3],
                "FertilizerConverted": "Yes"
            })

    # Generate remaining samples where FertilizerConverted is 'No'
    while len(data) < num_samples:
        sensor_values = [random.randint(0, 1023) for _ in range(4)]
        if sum(s > threshold for s in sensor_values) < 2:
            data.append({
                "Sensor1": sensor_values[0],
                "Sensor2": sensor_values[1],
                "Sensor3": sensor_values[2],
                "Sensor4": sensor_values[3],
                "FertilizerConverted": "No"
            })

    return data

# Generate the balanced dataset
data = generate_balanced_sensor_data(num_samples=2000)

# Convert to DataFrame
df = pd.DataFrame(data)

# Save to CSV file
df.to_csv("balanced_sensor_fertilizer_data.csv", index=False)

print("Balanced dataset generated and saved to 'balanced_sensor_fertilizer_data.csv'")
