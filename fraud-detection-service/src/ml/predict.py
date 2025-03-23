from pyspark.sql import SparkSession
from pyspark.ml.pipeline import PipelineModel
from pyspark.sql.functions import (
    col, from_json, to_json, struct, udf, explode, array, lit, 
    current_timestamp, window, count, avg, max, min
)
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, 
    BooleanType, TimestampType, ArrayType
)
import os
import time
import json
import numpy as np
from datetime import datetime

def create_spark_session():
    """Create a Spark session for prediction service"""
    return SparkSession.builder \
        .appName("FraudDetectionPredictionService") \
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.3.2,com.datastax.spark:spark-cassandra-connector_2.12:3.2.0") \
        .config("spark.cassandra.connection.host", os.getenv("CASSANDRA_CONTACT_POINTS", "cassandra")) \
        .getOrCreate()

def load_model(model_path="/app/models/fraud_model"):
    """Load the trained ML model for fraud detection"""
    try:
        # Wait for model to be available
        max_attempts = 10
        attempts = 0
        
        while attempts < max_attempts:
            try:
                model = PipelineModel.load(model_path)
                print(f"Successfully loaded model from {model_path}")
                return model
            except Exception as e:
                print(f"Waiting for model to be available: {e}")
                attempts += 1
                time.sleep(10)
        
        print("Could not load model, using fallback")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def define_schema():
    """Define the schema for the incoming transaction data"""
    return StructType([
        StructField("transaction_id", StringType(), True),
        StructField("user_id", StringType(), True),
        StructField("timestamp", StringType(), True),
        StructField("merchant", StringType(), True),
        StructField("location", StringType(), True),
        StructField("amount", DoubleType(), True),
        StructField("card_present", BooleanType(), True),
        StructField("card_type", StringType(), True)
    ])

def load_user_data(spark):
    """Load user data from Cassandra for enrichment"""
    return spark.read \
        .format("org.apache.spark.sql.cassandra") \
        .option("keyspace", "fraud_detection") \
        .option("table", "users") \
        .load()

def load_user_history(spark):
    """Load user transaction history from Cassandra"""
    return spark.read \
        .format("org.apache.spark.sql.cassandra") \
        .option("keyspace", "fraud_detection") \
        .option("table", "transactions_by_user") \
        .load()

def enrich_transaction_data(transactions_df, users_df, history_df):
    """Enrich transaction data with user info and historical features"""
    # Join with user data
    enriched_df = transactions_df.join(users_df, on="user_id", how="left")
    
    # Create user history window
    user_window = window(history_df.timestamp, "30 days")
    
    # Calculate historical aggregates
    user_history_aggs = history_df.groupBy("user_id") \
        .agg(
            count("transaction_id").alias("transactions_last_30days"),
            avg("amount").alias("avg_amount_last_30days"),
            max("amount").alias("max_amount_last_30days"),
            sum("amount").alias("total_amount_last_30days"),
            count(when(col("card_present") == False, 1)).alias("card_not_present_last_30days")
        )
    
    # User-merchant history
    user_merchant_history = history_df.groupBy("user_id", "merchant") \
        .agg(
            count("transaction_id").alias("user_merchant_txn_count"),
            avg("amount").alias("user_merchant_avg_amount"),
            max("amount").alias("user_merchant_max_amount")
        )
    
    # Join with historical aggregates
    enriched_df = enriched_df \
        .join(user_history_aggs, on="user_id", how="left") \
        .join(user_merchant_history, on=["user_id", "merchant"], how="left")
    
    # Add derived features
    enriched_df = enriched_df \
        .withColumn("hour_of_day", expr("hour(timestamp)")) \
        .withColumn("day_of_week", expr("dayofweek(timestamp)")) \
        .withColumn("is_weekend", when(expr("dayofweek(timestamp) IN (1, 7)"), 1).otherwise(0)) \
        .withColumn("amount_vs_avg_ratio", 
                   when(col("avg_amount_last_30days").isNull() | (col("avg_amount_last_30days") == 0), 1.0)
                   .otherwise(col("amount") / col("avg_amount_last_30days")))
    
    # Fill nulls with defaults
    for column in enriched_df.columns:
        if enriched_df.schema[column].dataType == DoubleType():
            enriched_df = enriched_df.withColumn(column, 
                                              when(col(column).isNull(), 0.0).otherwise(col(column)))
    
    return enriched_df

def apply_model_prediction(model, enriched_df):
    """Apply fraud detection model to make predictions"""
    if model:
        # Transform using the loaded ML pipeline
        predictions = model.transform(enriched_df)
        
        # Extract prediction results
        results = predictions.select(
            "transaction_id",
            "user_id",
            "timestamp",
            "amount",
            "merchant",
            "location",
            "card_type",
            "card_present",
            col("prediction").alias("is_fraud_prediction"),
            col("probability").getItem(1).alias("fraud_probability")
        )
    else:
        # Fallback method if model isn't available
        fallback_prediction = udf(fallback_fraud_detection, DoubleType())
        
        results = enriched_df.withColumn(
            "fraud_probability", 
            fallback_prediction(col("amount"), col("amount_vs_avg_ratio"))
        ).withColumn(
            "is_fraud_prediction",
            when(col("fraud_probability") > 0.5, 1.0).otherwise(0.0)
        )
    
    return results

def fallback_fraud_detection(amount, amount_ratio):
    """Simple rule-based fallback fraud detection"""
    score = 0.1  # Default low probability
    
    # High amount transactions
    if amount > 1000:
        score += 0.3
    
    # Unusual amount compared to average
    if amount_ratio > 5.0:
        score += 0.4
    
    return min(score, 0.99)  # Cap at 0.99

def save_predictions(predictions_df):
    """Save prediction results to Cassandra"""
    # Add timestamp for when prediction was made
    results_with_timestamp = predictions_df.withColumn("prediction_timestamp", current_timestamp())
    
    # Write to predictions table
    results_with_timestamp.write \
        .format("org.apache.spark.sql.cassandra") \
        .mode("append") \
        .option("keyspace", "fraud_detection") \
        .option("table", "predictions") \
        .save()
    
    # Write high-risk transactions to alerts table
    alerts = results_with_timestamp.filter(col("fraud_probability") > 0.5)
    
    if alerts.count() > 0:
        alerts = alerts.withColumn("alert_id", col("transaction_id")) \
            .withColumn("alert_timestamp", current_timestamp()) \
            .withColumn("status", lit("NEW"))
        
        alerts.write \
            .format("org.apache.spark.sql.cassandra") \
            .mode("append") \
            .option("keyspace", "fraud_detection") \
            .option("table", "fraud_alerts") \
            .save()

def process_streaming_predictions():
    """Process streaming data for real-time predictions"""
    spark = create_spark_session()
    model = load_model()
    schema = define_schema()
    
    # Load reference data
    users_df = load_user_data(spark)
    history_df = load_user_history(spark)
    
    # Create Kafka source stream
    kafka_stream = spark \
        .readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:29092")) \
        .option("subscribe", os.getenv("INPUT_TOPIC", "raw-transactions")) \
        .option("startingOffsets", "latest") \
        .load()
    
    # Parse JSON data
    parsed_stream = kafka_stream.select(
        from_json(col("value").cast("string"), schema).alias("data")
    ).select("data.*")
    
    # Convert timestamp string to timestamp type
    parsed_stream = parsed_stream.withColumn("timestamp", col("timestamp").cast(TimestampType()))
    
    # Process each batch
    def process_batch(batch_df, batch_id):
        if batch_df.isEmpty():
            print(f"Batch {batch_id} is empty, skipping")
            return
        
        print(f"Processing batch {batch_id} with {batch_df.count()} transactions")
        
        # Enrich with user data and historical features
        enriched_batch = enrich_transaction_data(batch_df, users_df, history_df)
        
        # Apply model for predictions
        predictions = apply_model_prediction(model, enriched_batch)
        
        # Save predictions to Cassandra
        save_predictions(predictions)
        
        # Send high-risk transactions to Kafka alert topic
        alerts = predictions.filter(col("fraud_probability") > 0.5)
        if alerts.count() > 0:
            alerts.select(
                col("transaction_id").alias("key"),
                to_json(struct("*")).alias("value")
            ).write \
                .format("kafka") \
                .option("kafka.bootstrap.servers", os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:29092")) \
                .option("topic", os.getenv("OUTPUT_TOPIC", "fraud-alerts")) \
                .save()
        
        print(f"Completed processing batch {batch_id}, found {alerts.count()} alerts")
    
    # Start streaming query
    query = parsed_stream \
        .writeStream \
        .foreachBatch(process_batch) \
        .outputMode("update") \
        .option("checkpointLocation", "/app/checkpoints/predictions") \
        .start()
    
    query.awaitTermination()

def run_batch_predictions():
    """Run batch predictions on historical data"""
    spark = create_spark_session()
    model = load_model()
    
    # Load historical transactions
    transactions = spark.read \
        .format("org.apache.spark.sql.cassandra") \
        .option("keyspace", "fraud_detection") \
        .option("table", "transactions_by_time") \
        .load()
    
    # Load user data
    users_df = load_user_data(spark)
    history_df = load_user_history(spark)
    
    # Enrich transactions
    enriched_transactions = enrich_transaction_data(transactions, users_df, history_df)
    
    # Apply model
    predictions = apply_model_prediction(model, enriched_transactions)
    
    # Save predictions
    save_predictions(predictions)
    
    print(f"Batch predictions completed. Processed {predictions.count()} transactions.")
    print(f"Found {predictions.filter(col('fraud_probability') > 0.5).count()} high-risk transactions.")

if __name__ == "__main__":
    # Wait for services to be ready
    print("Waiting for services to be ready...")
    time.sleep(20)
    
    # Choose mode based on environment variable
    mode = os.getenv("PREDICTION_MODE", "streaming").lower()
    
    if mode == "batch":
        run_batch_predictions()
    else:
        process_streaming_predictions()