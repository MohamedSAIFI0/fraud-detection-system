from pyspark.sql import SparkSession
import os
from pyspark.sql.functions import col, sum, avg, count, datediff, to_date, current_date
import time

def create_spark_session():
    """Create a Spark session for batch processing"""
    return SparkSession.builder \
        .appName("FraudDetectionBatchProcessor") \
        .config("spark.jars.packages", "com.datastax.spark:spark-cassandra-connector_2.12:3.2.0") \
        .config("spark.cassandra.connection.host", os.getenv("CASSANDRA_CONTACT_POINTS", "cassandra")) \
        .getOrCreate()

def process_batch():
    """Process batch data to generate insights and aggregations"""
    spark = create_spark_session()
    
    # Read transactions from Cassandra
    transactions = spark.read \
        .format("org.apache.spark.sql.cassandra") \
        .option("keyspace", "fraud_detection") \
        .option("table", "transactions") \
        .load()
    
    # Read alerts from Cassandra
    alerts = spark.read \
        .format("org.apache.spark.sql.cassandra") \
        .option("keyspace", "fraud_detection") \
        .option("table", "fraud_alerts") \
        .load()
    
    # Register as temp views for SQL queries
    transactions.createOrReplaceTempView("transactions")
    alerts.createOrReplaceTempView("alerts")
    
    # Perform analytics
    
    # 1. User transaction statistics
    user_stats = spark.sql("""
        SELECT 
            user_id,
            COUNT(*) as transaction_count,
            SUM(amount) as total_amount,
            AVG(amount) as avg_amount,
            MIN(amount) as min_amount,
            MAX(amount) as max_amount
        FROM transactions
        GROUP BY user_id
    """)
    
    # 2. Merchant statistics
    merchant_stats = spark.sql("""
        SELECT 
            merchant,
            COUNT(*) as transaction_count,
            COUNT(DISTINCT user_id) as unique_users,
            SUM(amount) as total_amount,
            AVG(amount) as avg_amount
        FROM transactions
        GROUP BY merchant
    """)
    
    # 3. Fraud statistics
    fraud_stats = spark.sql("""
        SELECT 
            COUNT(*) as alert_count,
            COUNT(DISTINCT user_id) as affected_users,
            SUM(score) as total_risk_score,
            AVG(score) as avg_risk_score
        FROM alerts
    """)
    
    # Save aggregated data back to Cassandra
    
    # User statistics
    user_stats.write \
        .format("org.apache.spark.sql.cassandra") \
        .mode("append") \
        .option("keyspace", "fraud_detection") \
        .option("table", "user_statistics") \
        .save()
    
    # Merchant statistics
    merchant_stats.write \
        .format("org.apache.spark.sql.cassandra") \
        .mode("append") \
        .option("keyspace", "fraud_detection") \
        .option("table", "merchant_statistics") \
        .save()

    # Log statistics
    print("Batch processing completed successfully.")
    print(f"Processed {transactions.count()} transactions.")
    print(f"Found {alerts.count()} potential fraud alerts.")

if __name__ == "__main__":
    # Wait a bit for services to be ready
    print("Waiting for services to be ready...")
    time.sleep(20)
    
    process_batch()