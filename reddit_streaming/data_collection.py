# data_collection.py   -- producer
import os
import subprocess
from dotenv import load_dotenv
import praw
import json
import socket
import time
from requests.exceptions import RequestException

# Load environment variables from the creds.sh file
env_file = os.path.join(os.path.dirname(__file__), "creds.sh")
# env_file = "creds.sh"
if os.path.exists(env_file):
    load_dotenv(env_file, override=True)
else:
    raise FileNotFoundError(f"The environment file {env_file} does not exist.")
# Access the environment variables
CLIENT_ID = os.getenv("CLIENT_ID")
SECRET_TOKEN = os.getenv("SECRET_TOKEN")
USER_AGENT = os.getenv("USER_AGENT")

# Ensure the environment variables are correctly loaded
if not all([CLIENT_ID, SECRET_TOKEN, USER_AGENT]):
    raise ValueError(
        "Missing one or more environment variables: CLIENT_ID, SECRET_TOKEN, USER_AGENT"
    )

# Initialize the Reddit client with your credentials
reddit = praw.Reddit(
    client_id=CLIENT_ID, client_secret=SECRET_TOKEN, user_agent=USER_AGENT
)


def save_to_json(data, filename):
    filepath = os.path.join("data", "json", filename)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)
        print(f"Data saved to {filepath}")


def send_data_to_socket(data, conn):
    json_encoded = json.dumps(data)
    conn.sendall(
        json_encoded.encode("utf-8") + b"\n"
    )  # Add newline to signal end of message


def get_post_info(post_id, retries=3):
    for attempt in range(retries):
        try:
            submission = reddit.submission(id=post_id)
            info = {
                "id": submission.id,
                "title": submission.title,
                "author": submission.author.name if submission.author else "deleted",
                "subreddit": submission.subreddit.display_name,
                "score": submission.score,
                "created_utc": submission.created_utc,
                "url": submission.url,
                "num_comments": submission.num_comments,
                "text": submission.selftext,
            }
            return info
        except praw.exceptions.ClientException as e:
            print(f"Client error: {e}")
            return None
        except RequestException as e:
            print(f"Network error: {e}")
            if attempt < retries - 1:
                print("Retrying...")
                time.sleep(2)
            else:
                return None
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None


def stream_comments(post_id):
    # Socket Preparation
    host = "localhost"
    port = 9998  # Use a port for sending live comments
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((host, port))
    s.listen(1)  # Listen with a backlog of 1
    print(f"Listening on port: {port}")

    conn, addr = s.accept()
    print(f"Connection from {addr}")

    try:
        while True:
            post_info = get_post_info(post_id)
            if post_info:
                print("Post Information:")
                print(post_info)
                # save_to_json(post_info, f"{post_id}_info.json")
                send_data_to_socket(post_info, conn)
                submission = reddit.submission(id=post_id)
                submission.comments.replace_more(limit=None)
                all_comments = submission.comments.list()

                for comment in all_comments:
                    comment_data = {
                        "id": comment.id,
                        "author": (
                            comment.author.name if comment.author else "deleted"
                        ),
                        "created_utc": comment.created_utc,
                        "score": comment.score,
                        "parent_id": comment.parent_id,
                        "subreddit": comment.subreddit.display_name,
                        "permalink": comment.permalink,
                        "text": comment.body,
                    }
                    print(comment_data)
                    send_data_to_socket(comment_data, conn)
                    time.sleep(1)  # Adjust as needed to control rate of sending data
    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    # Example post ID
    post_id = (
        "1d7cpts"  # Update with the post ID you want to stream and save comments for
    )

    # Start streaming and saving comments
    stream_comments(post_id)
