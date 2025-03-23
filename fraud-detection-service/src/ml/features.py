from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, datediff, to_date, current_date, count, sum, 
    avg, stddev, min, max, expr, when, lit, lag, 
    window, row_number, collect_list, collect_set
)
from pyspark.sql.types import DoubleType, IntegerType
from pyspark.sql.window import Window
import os

def create_spark_session():
    """Create a Spark session for feature engineering"""
    return SparkSession.builder \
        .appName("FraudDetectionFeatureEngineering") \
        .config("spark.jars.packages", "com.datastax.spark:spark-cassandra-connector_2.12:3.2.0") \
        .config("spark.cassandra.connection.host", os.getenv("CASSANDRA_CONTACT_POINTS", "cassandra")) \
        .getOrCreate()

def load_transaction_data(spark):
    """Load transaction data from Cassandra"""
    return spark.read \
        .format("org.apache.spark.sql.cassandra") \
        .option("keyspace", "fraud_detection") \
        .option("table", "transactions_by_user") \
        .load()

def load_user_data(spark):
    """Load user data from Cassandra"""
    return spark.read \
        .format("org.apache.spark.sql.cassandra") \
        .option("keyspace", "fraud_detection") \
        .option("table", "users") \
        .load()

def engineer_transaction_features(transactions_df):
    """Engineer features based on transaction data"""
    # Create window specs for various time-based aggregations
    user_window = Window.partitionBy("user_id")
    user_7day_window = Window.partitionBy("user_id").orderBy("timestamp").rangeBetween(-7*86400, 0)
    user_30day_window = Window.partitionBy("user_id").orderBy("timestamp").rangeBetween(-30*86400, 0)
    user_order_window = Window.partitionBy("user_id").orderBy("timestamp")
    
    # Add user-based transaction features
    transactions_with_features = transactions_df \
        .withColumn("hour_of_day", expr("hour(timestamp)")) \
        .withColumn("day_of_week", expr("dayofweek(timestamp)")) \
        .withColumn("is_weekend", when(expr("dayofweek(timestamp) IN (1, 7)"), 1).otherwise(0)) \
        .withColumn("prev_transaction_amount", lag("amount", 1).over(user_order_window)) \
        .withColumn("amount_diff_from_prev", col("amount") - col("prev_transaction_amount")) \
        .withColumn("transactions_last_7days", count("transaction_id").over(user_7day_window)) \
        .withColumn("transactions_last_30days", count("transaction_id").over(user_30day_window)) \
        .withColumn("avg_amount_last_7days", avg("amount").over(user_7day_window)) \
        .withColumn("avg_amount_last_30days", avg("amount").over(user_30day_window)) \
        .withColumn("max_amount_last_7days", max("amount").over(user_7day_window)) \
        .withColumn("max_amount_last_30days", max("amount").over(user_30day_window)) \
        .withColumn("total_amount_last_7days", sum("amount").over(user_7day_window)) \
        .withColumn("total_amount_last_30days", sum("amount").over(user_30day_window)) \
        .withColumn("stddev_amount_last_30days", stddev("amount").over(user_30day_window)) \
        .withColumn("amount_vs_avg_ratio", col("amount") / col("avg_amount_last_30days")) \
        .withColumn("unique_merchants_last_30days", 
                    expr("size(collect_set(merchant) over (partition by user_id order by timestamp range between interval 30 days preceding and current row))"))
                    
    # Location-based features
    transactions_with_features = transactions_with_features \
        .withColumn("same_as_prev_location", 
                   when(lag("location", 1).over(user_order_window) == col("location"), 1).otherwise(0)) \
        .withColumn("unique_locations_last_30days", 
                   expr("size(collect_set(location) over (partition by user_id order by timestamp range between interval 30 days preceding and current row))"))
                   
    # Card present/not present features
    transactions_with_features = transactions_with_features \
        .withColumn("card_not_present_last_30days", 
                   sum(when(col("card_present") == False, 1).otherwise(0)).over(user_30day_window)) \
        .withColumn("card_not_present_ratio", 
                   col("card_not_present_last_30days") / col("transactions_last_30days"))
                   
    return transactions_with_features

def engineer_user_merchant_features(transactions_df):
    """Engineer features for user-merchant interaction"""
    # User-merchant interactions
    user_merchant_df = transactions_df.groupBy("user_id", "merchant") \
        .agg(
            count("transaction_id").alias("user_merchant_txn_count"),
            avg("amount").alias("user_merchant_avg_amount"),
            max("amount").alias("user_merchant_max_amount"),
            stddev("amount").alias("user_merchant_stddev_amount")
        )
    
    # Join to get user-merchant features
    transactions_with_merchant_features = transactions_df \
        .join(user_merchant_df, on=["user_id", "merchant"], how="left")
    
    return transactions_with_merchant_features

def create_training_dataset(transactions_df, users_df):
    """Create the final training dataset with all features"""
    # Engineer transaction features
    transaction_features = engineer_transaction_features(transactions_df)
    
    # Engineer user-merchant features
    user_merchant_features = engineer_user_merchant_features(transaction_features)
    
    # Join with user data
    dataset = user_merchant_features \
        .join(users_df, on="user_id", how="left") \
        .select(
            "transaction_id", 
            "user_id", 
            "timestamp", 
            "amount", 
            "merchant", 
            "location", 
            "card_present", 
            "card_type",
            "hour_of_day", 
            "day_of_week", 
            "is_weekend",
            "transactions_last_7days", 
            "transactions_last_30days",
            "avg_amount_last_7days", 
            "avg_amount_last_30days",
            "max_amount_last_7days", 
            "max_amount_last_30days",
            "total_amount_last_7days", 
            "total_amount_last_30days",
            "stddev_amount_last_30days",
            "amount_vs_avg_ratio",
            "unique_merchants_last_30days",
            "same_as_prev_location",
            "unique_locations_last_30days",
            "card_not_present_last_30days",
            "card_not_present_ratio",
            "user_merchant_txn_count",
            "user_merchant_avg_amount",
            "user_merchant_max_amount",
            "user_merchant_stddev_amount",
            "risk_score",
            "spending_pattern",
            "is_flagged_fraud" # This would be added from your historical data
        )
    
    # Handle null values
    for column in dataset.columns:
        if dataset.schema[column].dataType in [DoubleType(), IntegerType()]:
            dataset = dataset.withColumn(column, when(col(column).isNull(), 0).otherwise(col(column)))
    
    return dataset

def save_features(dataset):
    """Save the engineered features to Cassandra"""
    dataset.write \
        .format("org.apache.spark.sql.cassandra") \
        .mode("append") \
        .option("keyspace", "fraud_detection") \
        .option("table", "feature_store") \
        .save()

def run_feature_engineering():
    """Main function to run the feature engineering process"""
    spark = create_spark_session()
    
    print("Loading transaction data...")
    transactions_df = load_transaction_data(spark)
    
    print("Loading user data...")
    users_df = load_user_data(spark)
    
    print("Creating features...")
    dataset = create_training_dataset(transactions_df, users_df)
    
    print("Saving features to feature store...")
    save_features(dataset)
    
    print("Feature engineering completed successfully")
    
if __name__ == "__main__":
    run_feature_engineering()