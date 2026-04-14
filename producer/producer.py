from kafka import KafkaProducer
import json
import time
import pandas as pd

# Load dataset
df = pd.read_csv("Reviews.csv")

# Keep only useful columns
df = df[['Text', 'Score']].dropna()

# Create producer
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Stream data
for _, row in df.iterrows():
    message = {
        "text": row["Text"],
        "score": int(row["Score"])
    }

    producer.send("amazon_reviews", message)
    print(f"Sent: {message}")

    time.sleep(0.5)  # simulate real-time
