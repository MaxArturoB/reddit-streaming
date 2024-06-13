import subprocess
import time
# Group Members
#    Ciaran Fox
#    Arturo Butron
#    John Omole
#    Felix Agosto

if __name__ == "__main__":
    # Start the data producer in a new terminal
    subprocess.Popen(["python3", "data_collection.py"])

    # Wait for a few seconds before starting the data consumer
    time.sleep(5)

    # Start the data consumer in a new terminal
    subprocess.Popen(["spark-submit", "data_consumer.py"])

    # Wait for a few seconds before starting the file processor
    time.sleep(5)

    # Start the file processor in a new terminal
    subprocess.Popen(
        [
            "spark-submit",
            "data_processing_reference.py",
        ]
    )

    # Wait for a few seconds before starting the file processor
    time.sleep(1)

    # Start the file processor in a new terminal
    subprocess.Popen(
        [
            "python3",
            "data_processing_sentiment.py",
        ]
    )

    # Wait for a few seconds before starting the file processor
    time.sleep(1)

    # Start the file processor in a new terminal
    subprocess.Popen(["spark-submit", "data_processing_tfidf.py"])
