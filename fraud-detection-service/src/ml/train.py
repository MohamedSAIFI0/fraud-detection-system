from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer, OneHotEncoder
from pyspark.ml.classification import GBTClassifier, RandomForestClassifier, LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.pipeline import Pipeline
import os
from pyspark.sql.functions import col, when, rand
import mlflow
import mlflow.spark

def create_spark_session():
    """Create a Spark session for model training"""
    return SparkSession.builder \
        .appName("FraudDetectionModelTraining") \
        .config("spark.jars.packages", "com.datastax.spark:spark-cassandra-connector_2.12:3.2.0") \
        .config("spark.cassandra.connection.host", os.getenv("CASSANDRA_CONTACT_POINTS", "cassandra")) \
        .config("spark.sql.shuffle.partitions", "10") \
        .getOrCreate()

def load_feature_data(spark):
    """Load feature data from Cassandra"""
    return spark.read \
        .format("org.apache.spark.sql.cassandra") \
        .option("keyspace", "fraud_detection") \
        .option("table", "feature_store") \
        .load()

def preprocess_data(df):
    """Preprocess data for model training"""
    # Create categorical features
    categorical_cols = ["card_type", "merchant", "location", "spending_pattern"]
    indexers = [StringIndexer(inputCol=col, outputCol=f"{col}_index", handleInvalid="keep") 
                for col in categorical_cols]
    encoders = [OneHotEncoder(inputCol=f"{col}_index", outputCol=f"{col}_vec") 
                for col in categorical_cols]
    
    # Select numerical features
    numerical_cols = [
        "amount", "hour_of_day", "day_of_week", "is_weekend",
        "transactions_last_7days", "transactions_last_30days",
        "avg_amount_last_7days", "avg_amount_last_30days",
        "max_amount_last_7days", "max_amount_last_30days",
        "total_amount_last_7days", "total_amount_last_30days",
        "stddev_amount_last_30days", "amount_vs_avg_ratio",
        "unique_merchants_last_30days", "same_as_prev_location",
        "unique_locations_last_30days", "card_not_present_last_30days",
        "card_not_present_ratio", "user_merchant_txn_count",
        "user_merchant_avg_amount", "user_merchant_max_amount",
        "user_merchant_stddev_amount", "risk_score"
    ]
    
    # Boolean features
    boolean_cols = ["card_present"]
    for bool_col in boolean_cols:
        df = df.withColumn(bool_col, col(bool_col).cast("integer"))
    
    # Combine all feature columns
    all_features = numerical_cols + boolean_cols + [f"{col}_vec" for col in categorical_cols]
    
    # Assemble features into a vector
    assembler = VectorAssembler(inputCols=all_features, outputCol="features", handleInvalid="keep")
    
    # Scale features
    scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=True)
    
    # Create label column (is_flagged_fraud)
    df = df.withColumn("label", when(col("is_flagged_fraud") == True, 1.0).otherwise(0.0))
    
    # Split data into training and testing sets
    train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)
    
    print(f"Training set size: {train_data.count()}")
    print(f"Test set size: {test_data.count()}")
    
    return train_data, test_data, indexers, encoders, assembler, scaler

def train_model(train_data, test_data, indexers, encoders, assembler, scaler, model_path="/app/models/fraud_model"):
    """Train fraud detection model"""
    # Initialize MLflow
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("fraud_detection")
    
    with mlflow.start_run(run_name="fraud_detection_model"):
        # Log parameters
        mlflow.log_param("training_data_size", train_data.count())
        mlflow.log_param("test_data_size", test_data.count())
        
        # Create classifier
        rf = RandomForestClassifier(labelCol="label", featuresCol="scaled_features", 
                                    numTrees=100, maxDepth=10, seed=42)
        
        # Create pipeline
        pipeline = Pipeline(stages=indexers + encoders + [assembler, scaler, rf])
        
        # Train model
        print("Training model...")
        model = pipeline.fit(train_data)
        
        # Make predictions on test data
        predictions = model.transform(test_data)
        
        # Evaluate model
        evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction")
        auc = evaluator.evaluate(predictions)
        print(f"AUC: {auc}")
        
        multi_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
        f1 = multi_evaluator.evaluate(predictions)
        print(f"F1 Score: {f1}")
        
        # Log metrics
        mlflow.log_metric("auc", auc)
        mlflow.log_metric("f1", f1)
        
        # Save model
        print(f"Saving model to {model_path}")
        model.write().overwrite().save(model_path)
        
        # Log model to MLflow
        mlflow.spark.log_model(model, "fraud_model")
        
        return model, predictions

def hyperparameter_tuning(train_data, test_data, indexers, encoders, assembler, scaler):
    """Perform hyperparameter tuning for the model"""
    # Create base classifier
    rf = RandomForestClassifier(labelCol="label", featuresCol="scaled_features")
    
    # Define parameter grid
    paramGrid = ParamGridBuilder() \
        .addGrid(rf.numTrees, [50, 100, 200]) \
        .addGrid(rf.maxDepth, [5, 10, 15]) \
        .addGrid(rf.impurity, ["gini", "entropy"]) \
        .build()
    
    # Create pipeline
    pipeline = Pipeline(stages=indexers + encoders + [assembler, scaler, rf])
    
    # Create cross-validator
    evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction")
    cv = CrossValidator(estimator=pipeline, estimatorParamMaps=paramGrid, 
                       evaluator=evaluator, numFolds=3, seed=42, parallelism=2)
    
    # Run cross-validation and choose best model
    print("Running cross-validation for hyperparameter tuning...")
    cv_model = cv.fit(train_data)
    
    # Get best model
    best_model = cv_model.bestModel
    
    # Print best parameters
    print("Best parameters:")
    print(f"numTrees: {best_model.stages[-1].getNumTrees}")
    print(f"maxDepth: {best_model.stages[-1].getMaxDepth}")
    print(f"impurity: {best_model.stages[-1].getImpurity}")
    
    return best_model

def run_training():
    """Main function to run model training"""
    spark = create_spark_session()
    
    print("Loading feature data...")
    feature_data = load_feature_data(spark)
    
    print("Preprocessing data...")
    train_data, test_data, indexers, encoders, assembler, scaler = preprocess_data(feature_data)
    
    print("Training model...")
    model, predictions = train_model(train_data, test_data, indexers, encoders, assembler, scaler)
    
    # Optionally run hyperparameter tuning
    # best_model = hyperparameter_tuning(train_data, test_data, indexers, encoders, assembler, scaler)
    
    print("Model training completed successfully")
    
    return model

if __name__ == "__main__":
    run_training()