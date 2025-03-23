import json
import time
import uuid
import random
import os
from datetime import datetime, timedelta
from kafka import KafkaProducer
from faker import Faker
import numpy as np

# Initialize Faker for generating realistic data
fake = Faker()

# Configure Kafka producer
KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:29092')
TRANSACTION_TOPIC = os.getenv('TRANSACTION_TOPIC', 'raw-transactions')
GENERATION_INTERVAL = float(os.getenv('GENERATION_INTERVAL', '2'))

# Create Kafka producer
producer = KafkaProducer(
    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
    value_serializer=lambda x: json.dumps(x).encode('utf-8'),
    key_serializer=lambda x: str(x).encode('utf-8')
)

# Create a pool of users for consistent behavior patterns
def create_user_pool(size=1000):
    users = []
    for _ in range(size):
        user_id = str(uuid.uuid4())
        card_types = ['visa', 'mastercard', 'amex']
        users.append({
            'user_id': user_id,
            'name': fake.name(),
            'spending_pattern': random.choice(['low', 'medium', 'high']),
            'usual_merchants': [fake.company() for _ in range(random.randint(3, 8))],
            'usual_locations': [fake.city() for _ in range(random.randint(1, 3))],
            'card_type': random.choice(card_types),
            'risk_score': random.random()  # 0 to 1, higher means more risky
        })
    return users

# Generate a transaction for a specific user
def generate_transaction(user):
    # Decide if this will be a fraudulent transaction (rare)
    is_fraudulent = random.random() < (0.005 + user['risk_score'] * 0.01)
    
    # Transaction base data
    transaction = {
        'transaction_id': str(uuid.uuid4()),
        'user_id': user['user_id'],
        'timestamp': datetime.now().isoformat(),
        'card_type': user['card_type'],
        'card_present': random.random() > 0.2  # 80% of transactions are card present
    }
    
    # Normal transaction
    if not is_fraudulent:
        transaction['merchant'] = random.choice(user['usual_merchants'])
        transaction['location'] = random.choice(user['usual_locations'])
        
        # Amount based on spending pattern
        if user['spending_pattern'] == 'low':
            transaction['amount'] = round(random.uniform(1, 100), 2)
        elif user['spending_pattern'] == 'medium':
            transaction['amount'] = round(random.uniform(10, 500), 2)
        else:  # high
            transaction['amount'] = round(random.uniform(50, 2000), 2)
    
    # Fraudulent transaction
    else:
        # Unusual merchant
        while True:
            merchant = fake.company()
            if merchant not in user['usual_merchants']:
                transaction['merchant'] = merchant
                break
        
        # Unusual location
        while True:
            location = fake.city()
            if location not in user['usual_locations']:
                transaction['location'] = location
                break
        
        # Unusual amount (often higher than normal)
        base_amount = {
            'low': 100,
            'medium': 500,
            'high': 2000
        }[user['spending_pattern']]
        
        transaction['amount'] = round(random.uniform(base_amount, base_amount * 5), 2)
        
        # Often not card present for fraud
        transaction['card_present'] = random.random() > 0.7
    
    return transaction, is_fraudulent

# Main generator function
def generate_transactions():
    print(f"Initializing transaction generator. Publishing to {KAFKA_BOOTSTRAP_SERVERS}, topic: {TRANSACTION_TOPIC}")
    
    # Create a pool of users
    user_pool = create_user_pool(1000)
    
    # Generate and send transactions continuously
    while True:
        # Select a random user
        user = random.choice(user_pool)
        
        # Generate a transaction
        transaction, is_fraud = generate_transaction(user)
        
        # Send to Kafka
        future = producer.send(
            TRANSACTION_TOPIC,
            key=transaction['user_id'],
            value=transaction
        )
        
        # Log the transaction
        status = "FRAUD" if is_fraud else "NORMAL"
        print(f"Generated {status} transaction {transaction['transaction_id']} for user {user['name']} - ${transaction['amount']}")
        
        # Wait for the message to be sent
        try:
            record_metadata = future.get(timeout=10)
            print(f"Message sent to {record_metadata.topic} partition {record_metadata.partition} offset {record_metadata.offset}")
        except Exception as e:
            print(f"Failed to send message: {e}")
        
        # Sleep for a random interval
        time.sleep(GENERATION_INTERVAL * random.uniform(0.5, 1.5))

if __name__ == "__main__":
    # Wait for Kafka to be ready
    retries = 0
    max_retries = 10
    connected = False
    
    while not connected and retries < max_retries:
        try:
            producer.bootstrap_connected()
            connected = True
            print("Successfully connected to Kafka")
        except Exception as e:
            print(f"Waiting for Kafka to be ready... ({retries}/{max_retries})")
            retries += 1
            time.sleep(5)
    
    if not connected:
        print("Failed to connect to Kafka after several attempts. Exiting.")
        exit(1)
    
    # Start generating transactions
    try:
        generate_transactions()
    except KeyboardInterrupt:
        print("Transaction generator stopped.")
    finally:
        producer.close()