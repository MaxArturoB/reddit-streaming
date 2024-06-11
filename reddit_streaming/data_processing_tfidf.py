from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, CountVectorizer, IDF
from pyspark.sql.window import Window as W
from pyspark.ml import Pipeline
import os
import uuid
from pyspark.sql.types import ArrayType, DoubleType
from pyspark.sql.functions import (
    col,
    window,
    collect_list,
    from_unixtime,
    concat_ws,
    explode,
    udf,
    array,
    lit,
    struct,
    row_number,
)
from pyspark.sql import Row


def start_processing(input_dir="data/raw/*.parquet", output_dir="data/tfidf"):
    # Create a Spark session
    spark = (
        SparkSession.builder.appName("Reddit TF-IDF Processing")
        .master("local[*]")  # Use local[*] master
        .getOrCreate()
    )

    # Read all Parquet files
    df = spark.read.parquet(input_dir)

    tokenizer = Tokenizer(inputCol="text", outputCol="words")

    # Apply CountVectorizer to get term frequency
    cv = CountVectorizer(
        inputCol="words", outputCol="rawFeatures", vocabSize=1000, minDF=1.0
    )

    # Apply IDF to get TF-IDF
    idf = IDF(inputCol="rawFeatures", outputCol="features")

    # Define a pipeline
    pipeline = Pipeline(stages=[tokenizer, cv, idf])

    # Apply windowing
    windowed_df = df.groupBy(window(col("timestamp"), "60 seconds", "5 seconds")).agg(
        collect_list("text").alias("texts")
    )

    def process_window(row):
        window_start, window_end = row["window"]["start"], row["window"]["end"]
        texts = row["texts"]

        # Create a DataFrame for the texts in the window
        texts_df = spark.createDataFrame([(text,) for text in texts], ["text"])

        # Fit the pipeline to the data
        model = pipeline.fit(texts_df)

        # Transform the data
        tfidf_df = model.transform(texts_df)

        # Extract the vocabulary and TF-IDF features
        vocab = model.stages[1].vocabulary

        # Convert sparse vector to dense vector
        def to_array(v):
            if v is None:
                return None
            return v.toArray().tolist()

        to_array_udf = udf(to_array, ArrayType(DoubleType()))
        tfidf_df = tfidf_df.withColumn("tfidf_values", to_array_udf(col("features")))

        # Explode the features column to get individual words and their TF-IDF scores
        exploded_df = tfidf_df.select(
            explode(
                array(
                    [
                        struct(
                            lit(vocab[i]).alias("word"),
                            col("tfidf_values")[i].alias("tfidf"),
                        )
                        for i in range(len(vocab))
                    ]
                )
            ).alias("word_tfidf")
        )

        # Select word and tfidf score
        top_words_df = exploded_df.select("word_tfidf.word", "word_tfidf.tfidf")

        # Get top 10 words based on TF-IDF scores
        window_spec = W.orderBy(col("tfidf").desc())
        top_words_df = top_words_df.withColumn(
            "rank", row_number().over(window_spec)
        ).filter(col("rank") <= 10)

        # Add window information
        top_words_df = top_words_df.withColumn(
            "window_start", lit(window_start)
        ).withColumn("window_end", lit(window_end))

        top_words_df.write.mode("append").parquet(output_dir)

    for row in windowed_df.rdd.collect():
        process_window(row)


if __name__ == "__main__":
    start_processing()
