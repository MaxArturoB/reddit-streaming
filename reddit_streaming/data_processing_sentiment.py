from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType,
    StructField,
    DoubleType,
)
from pyspark.sql.functions import (
    col,
    array_join,
    udf,
)
import sparknlp

from sparknlp.base import DocumentAssembler
from nltk.corpus import stopwords
from sparknlp.annotator import (
    StopWordsCleaner,
    PerceptronModel,
    Chunker,
    LemmatizerModel,
    Normalizer,
    Tokenizer,
)
from sparknlp.base import Finisher
from pyspark.ml import Pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()


def start_sentiments(input_dir="data/raw/*.parquet", output_dir="data/sentiment"):
    # Create a Spark session
    # spark = sparknlp.start()
    """spark = (
        SparkSession.builder.appName("reddit streaming app")
        .master("local[*]")
        .config(
            "spark.jars.packages",
            "com.johnsnowlabs.nlp:spark-nlp-spark32_2.12:3.4.2",
        )
        .getOrCreate()
    )"""
    spark = (
        SparkSession.builder.appName("reddit sentiment app")
        .master("local[*]")
        .config(
            "spark.jars.packages",
            "org.apache.spark:spark-sql-kafka-0-10_2.12:3.0.1,com.johnsnowlabs.nlp:spark-nlp-spark32_2.12:3.4.2",
        )
        .config("spark.streaming.stopGracefullyOnShutdown", "true")
        .getOrCreate()
    )
    # Read all Parquet files
    df = spark.read.parquet(input_dir)

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
    processed_df = pipeline.fit(df).transform(df)

    processed_df = processed_df.withColumn(
        "processed_text", array_join(processed_df["finished_no_stop_lemmatized"], " ")
    )

    sentiment_schema = StructType(
        [
            StructField("compound", DoubleType()),
            StructField("positive", DoubleType()),
            StructField("neutral", DoubleType()),
            StructField("negative", DoubleType()),
        ]
    )

    def get_sentiment(text):
        vs = analyzer.polarity_scores(text)
        return (vs["compound"], vs["pos"], vs["neu"], vs["neg"])

    sentiment_udf = udf(get_sentiment, sentiment_schema)

    sentiment_df = processed_df.withColumn(
        "sentiment", sentiment_udf(col("processed_text"))
    )

    sentiment__df = (
        sentiment_df.withColumn("compound", col("sentiment.compound"))
        .withColumn("positive", col("sentiment.positive"))
        .withColumn("neutral", col("sentiment.neutral"))
        .withColumn("negative", col("sentiment.negative"))
        .drop("sentiment")
    )

    sentiment__df.write.mode("append").parquet(output_dir)


if __name__ == "__main__":
    start_sentiments()
