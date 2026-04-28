import os
from kaggle.api.kaggle_api_extended import KaggleApi

def main():
    # Define directories
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_data_dir = os.path.join(base_dir, 'data', 'raw')
    
    # Ensure data directory exists
    os.makedirs(raw_data_dir, exist_ok=True)
    
    dataset = "nasa/kepler-exoplanet-search-results"
    
    print("Authenticating with Kaggle API...")
    try:
        api = KaggleApi()
        api.authenticate()
    except Exception as e:
        print("Failed to authenticate with Kaggle.")
        print("Ensure kaggle.json is placed in your ~/.kaggle/ directory.")
        print(f"Error: {e}")
        return

    print(f"Downloading dataset '{dataset}' to '{raw_data_dir}'...")
    try:
        api.dataset_download_files(dataset, path=raw_data_dir, unzip=True)
        print("Download and extraction complete.")
    except Exception as e:
        print(f"Failed to download the dataset. Error: {e}")

if __name__ == "__main__":
    main()