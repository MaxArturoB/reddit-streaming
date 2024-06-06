# main.py

from reddit_streaming.data_collection import stream_and_save_comments
from reddit_streaming.data_processing import start_streaming

def main():
    # Choose the mode you want to run (producer or consumer)
    mode = "producer"  # Change to "producer" to start the producer

    if mode == "consumer":
        # Example post ID
        post_id = "1d7cpts"  # Update with the post ID you want to stream and save comments for
        stream_and_save_comments(post_id)
    elif mode == "consumer":
        start_streaming()

if __name__ == "__main__":
    main()
