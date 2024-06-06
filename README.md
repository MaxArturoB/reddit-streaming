# reddit-streaming
Produces insights and metrics in real time for a specific topic

Summary

To run the producer script directly, use: python data_collection.py
To run the consumer script directly, use: python data_processing.py

The use of the main.py allows to switch between producer and consumer modes, 
set the mode in main.py and run:

        PYTHONPATH=reddit_streaming python main.py


In Detail:
How to Run the Scripts
This project contains three main scripts: data_collection.py, data_processing.py, and main.py. Below are instructions on how to run each script separately or using the main.py script.

Running data_collection.py
data_collection.py is the producer script that collects data from Reddit and streams it to a socket.

Navigate to the project directory:
    cd /home/max/reddit_streaming/reddit_streaming
    Run the producer script:
            python data_collection.py

Running data_processing.py
data_processing.py is the consumer script that receives the data from the socket and processes it using PySpark.

Navigate to the project directory:
    cd /home/max/reddit_streaming/reddit_streaming
    Run the consumer script:
        python data_processing.py

Running with main.py
main.py is the main script that can run either the producer or the consumer script based on the mode you set.
    Edit main.py:
    Open main.py and set the mode variable to either "producer" or "consumer" based on your need.

    # main.py
    from reddit_streaming.data_collection import stream_and_save_comments
    from reddit_streaming.data_processing import start_streaming
    def main():
    # Choose the mode you want to run (producer or consumer)

    mode = "producer"  # Change to "consumer" to start the consumer

    Navigate to the project directory:
    cd /home/max/reddit_streaming
    Run the main script:

        PYTHONPATH=reddit_streaming python main.py


