from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, to_json, struct, to_timestamp
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, BooleanType
import os
import uuid
import json
from pyspark.ml.pipeline import PipelineModel
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType, StringType
import time

def create_spark_session():
    """Create a Spark session for stream processing"""
    return SparkSession.builder \
        .appName("FraudDetectionStreamProcessor") \
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.3.2,com.datastax.spark:spark-cassandra-connector_2.12:3.2.0") \
        .config("spark.cassandra.connection.host", os.getenv("CASSANDRA_CONTACT_POINTS", "cassandra")) \
        .getOrCreate()

def define_schema():
    """Define the schema for the incoming transaction data"""
    return StructType([
        StructField("transaction_id", StringType(), True),
        StructField("user_id", StringType(), True),
        StructField("amount", DoubleType(), True),
        StructField("timestamp", StringType(), True),
        StructField("merchant", StringType(), True),
        StructField("location", StringType(), True),
        StructField("card_present", BooleanType(), True),
        StructField("card_type", StringType(), True)
    ])

def load_model():
    """Load the trained ML model for fraud detection"""
    try:
        # Wait for model to be available
        model_path = "/app/models/fraud_model"
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
        # Return a simple rule-based detection function as fallback
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# UDF for fallback fraud detection
def fallback_fraud_detection(amount, location):
    """Simple rule-based fraud detection as fallback"""
    if amount > 1000:
        return (0.7, "High amount transaction")
    if location == "FOREIGN":
        return (0.6, "Foreign transaction")
    return (0.1, "Normal transaction")

def process_stream():
    """Main stream processing function"""
    spark = create_spark_session()
    schema = define_schema()
    model = load_model()
    
    # Create UDF for fallback fraud detection
    fallback_udf = udf(fallback_fraud_detection, StructType([
        StructField("score", DoubleType(), True),
        StructField("reason", StringType(), True)
    ]))
    
    # Read from Kafka
    df = spark \
        .readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:29092")) \
        .option("subscribe", os.getenv("INPUT_TOPIC", "raw-transactions")) \
        .option("startingOffsets", "latest") \
        .load()
    
    # Parse JSON data
    parsed = df.select(
        from_json(col("value").cast("string"), schema).alias("data")
    ).select("data.*")
    
    # Convert timestamp string to timestamp type
    parsed = parsed.withColumn("timestamp", to_timestamp(col("timestamp")))
    
    # Apply fraud detection
    if model:
        # Use the ML model
        predictions = model.transform(parsed)
        results = predictions.select(
            col("transaction_id"),
            col("user_id"),
            col("timestamp"),
            col("prediction").alias("is_fraud"),
            col("probability").getItem(1).alias("score"),
            col("merchant"),
            col("amount")
        )
    else:
        # Use fallback method
        results = parsed.withColumn(
            "fraud_detection", 
            fallback_udf(col("amount"), col("location"))
        ).select(
            col("transaction_id"),
            col("user_id"),
            col("timestamp"),
            col("fraud_detection.score").alias("score"),
            col("fraud_detection.reason").alias("reason"),
            col("merchant"),
            col("amount")
        )
    
    # Write to Cassandra
    cassandra_query = results \
        .writeStream \
        .foreachBatch(write_to_cassandra) \
        .outputMode("update") \
        .option("checkpointLocation", "/app/checkpoints/cassandra") \
        .start()
    
    # Write alerts to Kafka
    kafka_query = results \
        .filter(col("score") > 0.5) \
        .select(
            col("transaction_id").alias("key"),
            to_json(struct("*")).alias("value")
        ) \
        .writeStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:29092")) \
        .option("topic", os.getenv("OUTPUT_TOPIC", "fraud-alerts")) \
        .option("checkpointLocation", "/app/checkpoints/kafka") \
        .start()
    
    spark.streams.awaitAnyTermination()

def write_to_cassandra(batch_df, batch_id):
    """Write batch data to Cassandra"""
    # Filter for possible fraud
    alerts = batch_df.filter(col("score") > 0.5)
    
    # Generate alert IDs
    alerts = alerts.withColumn("alert_id", col("transaction_id"))
    
    # Write transactions to Cassandra
    batch_df.write \
        .format("org.apache.spark.sql.cassandra") \
        .mode("append") \
        .option("keyspace", "fraud_detection") \
        .option("table", "transactions") \
        .save()
    
    # Write alerts to Cassandra if any
    if not alerts.isEmpty():
        alerts.write \
            .format("org.apache.spark.sql.cassandra") \
            .mode("append") \
            .option("keyspace", "fraud_detection") \
            .option("table", "fraud_alerts") \
            .save()

if __name__ == "__main__":
    # Wait a bit for Kafka and Cassandra to be ready
    print("Waiting for services to be ready...")
    time.sleep(20)
    
    process_stream()