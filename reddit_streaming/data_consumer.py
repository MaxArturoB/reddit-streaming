from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
    DoubleType,
)
from pyspark.sql.functions import (
    from_json,
    col,
    window,
    from_unixtime,
    regexp_extract,
)


def start_streaming(host="localhost", port=9998):
    # Create a Spark session with the necessary configurations
    spark = (
        SparkSession.builder.appName("reddit streaming app")
        .master("local[*]")
        .config(
            "spark.jars.packages",
            "org.apache.spark:spark-sql-kafka-0-10_2.12:3.0.1,com.johnsnowlabs.nlp:spark-nlp-spark32_2.12:3.4.2",
        )
        .config("spark.streaming.stopGracefullyOnShutdown", "true")
        .getOrCreate()
    )

    # spark = SparkSession.builder.appName("reddit streaming").getOrCreate()
    # Read data from the specified socket
    lines_df = (
        spark.readStream.format("socket")
        .option("host", host)
        .option("port", port)
        .load()
    )
    # Define the schema for the JSON data
    schema = StructType(
        [
            StructField("id", StringType(), True),
            StructField("author", StringType(), True),
            StructField("created_utc", DoubleType(), True),
            StructField("score", IntegerType(), True),
            StructField("parent_id", StringType(), True),
            StructField("subreddit", StringType(), True),
            StructField("permalink", StringType(), True),
            StructField("text", StringType(), True),
        ]
    )
    # Parse the JSON data and extract the relevant fields
    json_df = lines_df.select(from_json(col("value"), schema).alias("data")).select(
        "data.*"
    )

    # Convert the created_utc field to a timestamp
    json_df = json_df.withColumn("timestamp", from_unixtime(col("created_utc")))

    # Write the processed data to a Parquet file
    raw_query = (
        json_df.writeStream.outputMode("append")
        .format("parquet")
        .option("path", "data/raw")
        .option("checkpointLocation", "data/checkpoint/raw")
        .trigger(processingTime="5 seconds")
        .start()
        .awaitTermination()
    )
    # Keep the streaming job running
    spark.streams.awaitAnyTermination()


if __name__ == "__main__":
    start_streaming()
