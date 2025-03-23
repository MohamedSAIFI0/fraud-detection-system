-- Create a keyspace for our fraud detection system
CREATE KEYSPACE IF NOT EXISTS fraud_detection
WITH replication = {
    'class': 'SimpleStrategy',
    'replication_factor': 3
};

USE fraud_detection;

-- Users table to store user information
CREATE TABLE IF NOT EXISTS users (
    user_id UUID PRIMARY KEY,
    name TEXT,
    spending_pattern TEXT,
    usual_merchants LIST<TEXT>,
    usual_locations LIST<TEXT>,
    card_type TEXT,
    risk_score FLOAT,
    created_at TIMESTAMP,
    last_updated TIMESTAMP
);

-- Transactions by user (for user history queries)
CREATE TABLE IF NOT EXISTS transactions_by_user (
    user_id UUID,
    transaction_id UUID,
    timestamp TIMESTAMP,
    merchant TEXT,
    location TEXT,
    amount DECIMAL,
    card_type TEXT,
    card_present BOOLEAN,
    is_flagged_fraud BOOLEAN,
    is_confirmed_fraud BOOLEAN,
    PRIMARY KEY ((user_id), timestamp, transaction_id)
) WITH CLUSTERING ORDER BY (timestamp DESC, transaction_id ASC);

-- Transactions by time (for real-time monitoring and batch analysis)
CREATE TABLE IF NOT EXISTS transactions_by_time (
    date_bucket TEXT,
    timestamp TIMESTAMP,
    transaction_id UUID,
    user_id UUID,
    merchant TEXT,
    location TEXT,
    amount DECIMAL,
    card_type TEXT,
    card_present BOOLEAN,
    is_flagged_fraud BOOLEAN,
    is_confirmed_fraud BOOLEAN,
    PRIMARY KEY ((date_bucket), timestamp, transaction_id)
) WITH CLUSTERING ORDER BY (timestamp DESC, transaction_id ASC);

-- Unusual transactions (those flagged as potentially fraudulent)
CREATE TABLE IF NOT EXISTS flagged_transactions (
    is_reviewed BOOLEAN,
    timestamp TIMESTAMP,
    transaction_id UUID,
    user_id UUID,
    merchant TEXT,
    location TEXT,
    amount DECIMAL,
    card_type TEXT,
    card_present BOOLEAN,
    fraud_score FLOAT,
    review_notes TEXT,
    PRIMARY KEY ((is_reviewed), timestamp, transaction_id)
) WITH CLUSTERING ORDER BY (timestamp DESC, transaction_id ASC);

-- User activity metrics (aggregated daily for analysis)
CREATE TABLE IF NOT EXISTS user_activity_metrics (
    user_id UUID,
    date TEXT,
    transaction_count INT,
    total_amount DECIMAL,
    avg_transaction_amount DECIMAL,
    unusual_locations_count INT,
    unusual_merchants_count INT,
    PRIMARY KEY ((user_id), date)
) WITH CLUSTERING ORDER BY (date DESC);

-- Location-based transaction metrics
CREATE TABLE IF NOT EXISTS location_metrics (
    location TEXT,
    date TEXT,
    transaction_count INT,
    fraud_count INT,
    total_amount DECIMAL,
    avg_transaction_amount DECIMAL,
    PRIMARY KEY ((location), date)
) WITH CLUSTERING ORDER BY (date DESC);

-- Merchant-based transaction metrics
CREATE TABLE IF NOT EXISTS merchant_metrics (
    merchant TEXT,
    date TEXT,
    transaction_count INT,
    fraud_count INT,
    total_amount DECIMAL,
    avg_transaction_amount DECIMAL,
    PRIMARY KEY ((merchant), date)
) WITH CLUSTERING ORDER BY (date DESC);