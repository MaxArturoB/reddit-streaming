# data_processing.py  Reddit_consumer

import socket
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    LongType,
    IntegerType,
    DoubleType,
)
from pyspark.sql.functions import (
    explode,
    split,
    from_json,
    col,
    array_join,
    udf,
    avg,
    window,
    to_timestamp,
    from_unixtime,
)
from pyspark.streaming import StreamingContext
from sparknlp.base import DocumentAssembler
from sparknlp.annotator import Tokenizer
from sparknlp.annotator import Normalizer
from sparknlp.annotator import LemmatizerModel
from pyspark.ml.feature import CountVectorizer
from nltk.corpus import stopwords
from sparknlp.annotator import StopWordsCleaner
from sparknlp.annotator import PerceptronModel
from sparknlp.annotator import Chunker
from sparknlp.base import Finisher
from pyspark.ml import Pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pyspark.mllib.feature import HashingTF, IDF

analyzer = SentimentIntensityAnalyzer()


def start_streaming(host="localhost", port=9998):
    # Create a local StreamingContext with two working threads and batch interval of 5 seconds
    # sc = SparkContext("local[2]", "DisplayLines")
    # ssc = StreamingContext(sc, 5)
    spark = (
        SparkSession.builder.appName("reddit streaming app")
        .master("local[*]")
        .config(
            "spark.jars.packages",
            "org.apache.spark:spark-sql-kafka-0-10_2.12:3.0.1,com.johnsnowlabs.nlp:spark-nlp-spark32_2.12:3.4.2",
        )
        .config("spark.streaming.stopGracefullyOnShutdown", "true")
        .config("spark.driver.memory", "8G")
        .config("spark.driver.maxResultSize", "0")
        .config("spark.kryoserializer.buffer.max", "2000M")
        .getOrCreate()
    )

    # spark = SparkSession.builder.appName("reddit streaming").getOrCreate()

    lines_df = (
        spark.readStream.format("socket")
        .option("host", host)
        .option("port", port)
        .load()
    )
    # Create DataFrame representing the stream of input lines from connection to host:port
    # .window(10, 5)

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

    json_df = lines_df.select(from_json(col("value"), schema).alias("data")).select(
        "data.*"
    )
    # Convert epoch time to timestamp
    json_df = json_df.withColumn("timestamp", from_unixtime(col("created_utc")))

    # Create a temporary view for raw data
    json_df.createOrReplaceTempView("raw")

    query = (
        json_df.writeStream.format("json")
        .option("path", "data/raw")
        .option("checkpointLocation", "data/checkpoint")
        .start()
    )

    """
    query.awaitTermination()
    """
    # Count occurrences in 60-second windows, updated every 5 seconds
    windowed_counts = json_df.groupBy(
        window(col("timestamp"), "60 seconds", "5 seconds"),
        "author",
        "subreddit",
        "permalink",
        "text",
    ).count()

    # query = windowed_counts.writeStream.outputMode("update").format("console").start()
    # query.awaitTermination()

    # prepare into spark format
    documentAssembler = DocumentAssembler().setInputCol("text").setOutputCol("document")

    # tokenisation
    tokenizer = Tokenizer().setInputCols(["document"]).setOutputCol("tokenized")

    # convert the text to lowercase, empty string and special character
    normalizer = (
        Normalizer()
        .setInputCols(["tokenized"])
        .setOutputCol("normalized")
        .setLowercase(True)
    )

    # lemmtizing tokens
    lemmatizer = (
        LemmatizerModel.pretrained()
        .setInputCols(["normalized"])
        .setOutputCol("lemmatized")
    )

    # remove stop words,
    eng_stopwords = stopwords.words("english")
    stopwords_cleaner = (
        StopWordsCleaner()
        .setInputCols(["lemmatized"])
        .setOutputCol("no_stop_lemmatized")
        .setStopWords(eng_stopwords)
    )

    pos_tagger = (
        PerceptronModel.pretrained("pos_anc")
        .setInputCols(["document", "lemmatized"])
        .setOutputCol("pos")
    )

    allowed_tags = ["<JJ>+<NN>", "<NN>+<NN>"]
    chunker = (
        Chunker()
        .setInputCols(["document", "pos"])
        .setOutputCol("ngrams")
        .setRegexParsers(allowed_tags)
    )

    """ 
        cv = CountVectorizer(inputCol="no_stop_lemmatized", outputCol="tf_features")
        IDF()
        .setInputCol("tf_features")
        .setOutputCol("idf_features")
        .setNumFeatures(1000)
    )
    """
    finisher = Finisher().setInputCols(
        ["no_stop_lemmatized", "normalized", "tokenized"]
    )

    pipeline = Pipeline().setStages(
        [
            documentAssembler,
            tokenizer,
            normalizer,
            lemmatizer,
            stopwords_cleaner,
            pos_tagger,
            chunker,
            finisher,
        ]
    )
    processed_df = pipeline.fit(json_df).transform(json_df)
    processed_df = processed_df.withColumn(
        "processed_text", array_join(processed_df["finished_no_stop_lemmatized"], " ")
    )
    processed_df = processed_df.filter(
        processed_df.finished_no_stop_lemmatized.isNotNull()
    )
    tf = (
        CountVectorizer(inputCol="finished_no_stop_lemmatized", outputCol="tf_features")
        .fit(processed_df)
        .transform(processed_df)
    )

    tf_idf = (
        IDF(inputCol="tf_features", outputCol="vectorised_features")
        .fit(tf)
        .transform(tf)
    )

    some = tf_idf.select("vectorised_features", "finished_no_stop_lemmatized")

    """
    # Assuming 'processed_df' is a static DataFrame for fitting models
    # Fit the CountVectorizer model
    tf = (
        CountVectorizer(inputCol="finished_no_stop_lemmatized", outputCol="tf_features")
        .fit(processed_df)
        .transform(processed_df)
    )
    tf_idf = (
        IDF(inputCol="tf_features", outputCol="vectorised_features")
        .fit(tf)
        .transform(tf)
    )
    tf_idf = (
        IDF(inputCol="tf_features", outputCol="vectorised_features")
        .fit(tf)
        .transform(tf)
    )


    cv = CountVectorizer(
        inputCol="finished_no_stop_lemmatized",
        outputCol="tf_features",
        vocabSize=1000,
        minDF=5,
    )
    cv_model = cv.fit(processed_df)  # Note: In a real scenario, fit on a static DF

    # Fit the IDF model
    idf = IDF(inputCol="tf_features", outputCol="idf_features")
    idf_model = idf.fit(
        cv_model.transform(processed_df)
    )  # Note: In a real scenario, fit on a static DF

    # Apply CountVectorizer model to streaming data
    tf_df = cv_model.transform(processed_df)

    # Apply IDF model to streaming data
    idf_df = idf_model.transform(tf_df)
    """
    """    # Calculate TF-IDF
    tfidf_df = idf_df.select("id", "idf_features")

    # Explode the TF-IDF features to get individual words
    tfidf_exploded_df = tfidf_df.select("id", "idf_features").withColumn(
        "word", explode(col("idf_features"))
    )"""
    query = some.writeStream.outputMode("update").format("console").start()

    query.awaitTermination()
    """
    # Group by window and compute top 10 important words based on TF-IDF scores
    top_words_df = (
        tfidf_exploded_df.groupBy(window(json_df.timestamp, "60 seconds", "5 seconds"))
        .agg(collect_list("word").alias("words"))
        .withColumn("top_words", udf_extract_top_words("words"))
        .select("window", "top_words")
    )

    # Group by window and compute top 10 important words based on TF-IDF scores
    top_words_df = (
        tfidf_exploded_df.groupBy(window(json_df.timestamp, "60 seconds", "5 seconds"))
        .agg(collect_list("word").alias("words"))
        .withColumn("top_words", udf_extract_top_words("words"))
        .select("window", "top_words")
    )
    """

    def get_sentiment(text):
        vs = analyzer.polarity_scores(text)
        return vs["compound"]

    sentiment_udf = udf(get_sentiment, DoubleType())

    # Apply sentiment analysis
    sentiment_df = processed_df.withColumn(
        "sentiment", sentiment_udf(col("processed_text"))
    )

    # Calculate average sentiment
    avg_sentiment_df = sentiment_df.agg(avg("sentiment").alias("average_sentiment"))

    # Write average sentiment to console
    """
    query = (
        processed_df.writeStream.outputMode("complete")
        .format("console")
        .option("truncate", "false")
        .start()
    )
    query.awaitTermination()
    """
    """
    tf = (
        CountVectorizer(inputCol="finished_no_stop_lemmatized", outputCol="tf_features")
        .fit(sentiment_udf)
        .transform(sentiment_udf)
    )

    hashingTF = HashingTF(
        inputCol="finished_no_stop_lemmatized", outputCol="tf_features", numFeatures=20
    ).transform(sentiment_udf)

    tfIdf_df = (
        IDF(inputCol="tf_features", outputCol="vectorised_features")
        .fit(hashingTF)
        .transform(hashingTF)
    )
    """
    query = sentiment_df.writeStream.outputMode("append").format("console").start()

    """
    query = (
        sentiment_df.writeStream.outputMode("append")
        .format("json")
        .trigger(processingTime="60 seconds")
        .option("path", "data/json_data")
        .option("checkpointLocation", "data/checkpoint")
        .start()
    )
    """

    # Wait for the computation to terminate
    query.awaitTermination()


if __name__ == "__main__":
    start_streaming()
