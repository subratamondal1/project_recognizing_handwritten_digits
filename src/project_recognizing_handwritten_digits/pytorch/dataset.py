from pathlib import Path # Import the Path class from pathlib module
import requests # Import the requests module for making HTTP requests

def myData():
    ### Download Dataset ###
    data_path = Path("data") # Create a Path object for the data directory
    actual_path = data_path/"mnist" # Create a Path object for the mnist subdirectory

    # Create the directory if it does not exist
    actual_path.mkdir(parents=True, exist_ok=True)

    url = "https://github.com/pytorch/tutorials/raw/main/_static/" # The base URL for the dataset
    file_name = "mnist.pkl.gz" # The file name of the dataset

    # Check if the file already exists in the local directory
    if not (actual_path/file_name).exists():
        # If not, download the file from the URL and save it to the local directory
        content = requests.get(url+file_name).content # Get the content of the file as bytes
        (actual_path/file_name).open("wb").write(content) # Open the file in write mode and write the content

    ### Extract Dataset ###
    import pickle # Import the pickle module for loading and saving Python objects
    import gzip # Import the gzip module for working with compressed files

    # Open the compressed file in read mode
    with gzip.open((actual_path/file_name).as_posix(), "rb") as f:
        # Load the pickle object from the file, which contains three tuples: 
        # (x_train,y_train), (x_valid,y_valid), (x_test,y_test)
        ((x_train,y_train),(x_valid,y_valid),_) = pickle.load(file=f, encoding="latin-1")
        # We only need the first two tuples, so we ignore the third one with an underscore

    return (x_train,y_train), (x_valid,y_valid) # Return the two tuples as the output of the function



(x_train,y_train), (x_valid,y_valid) = myData()
print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
print(f"x_valid shape: {x_valid.shape}, y_valid shape: {y_valid.shape}")