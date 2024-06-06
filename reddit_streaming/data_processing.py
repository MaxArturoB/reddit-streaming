# data_processing.py  Reddit_consumer

import socket
from pyspark import SparkContext
from pyspark.streaming import StreamingContext

def start_streaming(host='localhost', port=9998):
    # Create a local StreamingContext with two working threads and batch interval of 5 seconds
    sc = SparkContext("local[2]", "DisplayLines")
    ssc = StreamingContext(sc, 5)

    # Create a DStream that will connect to the hostname and port
    lines = ssc.socketTextStream(host, port)

    # Print the lines received
    lines.pprint()

    # Start the computation
    ssc.start()

    # Wait for the computation to terminate
    ssc.awaitTermination()

if __name__ == "__main__":
    start_streaming()
