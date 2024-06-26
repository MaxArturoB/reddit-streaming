# reddit-streaming
Produces insights and metrics in real time for a specific topic.

## Summary

To run the producer script directly, use:
```sh
python data_collection.py

To run the consumer script directly, use:
python data_processing.py

The use of the main.py allows switching between producer and consumer modes. Set the mode in main.py and run:
PYTHONPATH=reddit_streaming python main.py

In Detail: How to Run the Scripts

This project contains several key scripts: data_collection.py, streamlit_app.py, and main.py. Below are instructions on how to run each script separately or using the main.py script.

Running data_collection.py

data_collection.py is the producer script that collects data from Reddit and streams it to a socket.

Navigate to the project directory:
cd /path/to/reddit_streaming/reddit_streaming

Run the producer script:
python data_collection.py

Running streamlit_app.py

streamlit_app.py is the script that launches a Streamlit application for real-time visualizations.

Navigate to the project directory:
cd /path/to/reddit_streaming/reddit_streaming

Run the Streamlit application:
streamlit run streamlit_app.py

Running with main.py

main.py is the main script that can run either the producer or the consumer script based on the mode you set.

Edit main.py:
Open main.py and set the mode variable to either “producer” or “consumer” based on your need.

# main.py
from reddit_streaming.data_collection import stream_and_save_comments
from reddit_streaming.data_processing import start_streaming

def main():
    # Choose the mode you want to run (producer or consumer)
    mode = "producer"  # Change to "consumer" to start the consumer

Navigate to the project directory:
cd /path/to/reddit_streaming

Run the main script:
PYTHONPATH=reddit_streaming python main.py

Additional Information

Ensure you have all the dependencies installed as specified in the pyproject.toml file. Use poetry install to set up your environment correctly.

For real-time visualizations, run streamlit_app.py after starting the data collection to see the live data updates in your browser.

This README provides a clear guide on how to run the scripts individually or through the main.py script for switching between producer and consumer modes. Make sure to adjust the paths according to your project directory structure.

Replace /path/to/reddit_streaming with the actual path to your project directory before copying this content into your README file.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
