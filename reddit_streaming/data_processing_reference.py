from pyspark.sql import SparkSession

from pyspark.sql.functions import col, window, regexp_extract


def start_reference(input_dir="data/raw/*.parquet", output_dir="data/reference"):
    # Create a Spark session
    spark = (
        SparkSession.builder.appName("Reddit sentiment Processing")
        .master("local[*]")
        .getOrCreate()
    )

    # Read all Parquet files
    df = spark.read.parquet(input_dir)

    user_ref = df.withColumn(
        "reference", regexp_extract(col("text"), r"/u/(\w+)", 0)
    ).filter(col("reference") != "")
    post_ref = df.withColumn(
        "reference", regexp_extract(col("text"), r"/r/(\w+)", 0)
    ).filter(col("reference") != "")
    url_ref = df.withColumn(
        "reference", regexp_extract(col("text"), r"(http\S+)", 0)
    ).filter(col("reference") != "")

    combined_df = (
        user_ref.select("timestamp", "reference")
        .union(post_ref.select("timestamp", "reference"))
        .union(url_ref.select("timestamp", "reference"))
    )

    windowed_counts = combined_df.groupBy(
        window(col("timestamp"), "60 seconds", "5 seconds"), col("reference")
    ).count()

    windowed_counts.write.mode("append").parquet(output_dir)


if __name__ == "__main__":
    start_reference()
