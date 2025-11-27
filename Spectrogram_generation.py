import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from tqdm import tqdm
import pandas as pd
import warnings

# Suppress librosa warning about large audio files
warnings.filterwarnings('ignore', category=UserWarning)


AUDIO_DIR = 'fma_small' 
OUTPUT_DIR = 'fma_spectrograms'
METADATA_PATH = 'fma_metadata/tracks.csv' 

# Hyperparameters for Spectrogram Generation
DURATION = 3        
TARGET_SR = 22050   
N_FFT = 2048
HOP_LENGTH = 512
MAX_SAMPLES = 0 

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Helper Function to Load Metadata and Paths
def get_track_metadata(metadata_path, audio_dir):
    """Loads track IDs, genres, and maps them to their file paths."""
    
    # Load the FMA metadata CSV
    tracks = pd.read_csv(metadata_path, index_col=0, header=[0, 1])
    
    # Clean up column names for easier access
    tracks = tracks['track'][['genre_top']]
    
    # Map track ID to file path
    all_files = glob.glob(os.path.join(audio_dir, '*/*.mp3'))
    
    data_list = []
    for fpath in all_files:
        track_id = int(os.path.basename(fpath).split('.')[0])
        
        if track_id in tracks.index:
            genre_label = tracks.loc[track_id, 'genre_top']
            data_list.append({
                'track_id': track_id,
                'file_path': fpath,
                'genre_label': genre_label
            })
            
    return pd.DataFrame(data_list)


# Spectrogram Processing Function 
def create_spectrogram_image(audio_path, genre_label, track_id):
    """Loads audio, computes Mel Spectrogram, and saves as a standardized image."""
    
    # Load the audio file
    y, sr = librosa.load(audio_path, duration=DURATION, sr=TARGET_SR)

    # Calculate the Mel Spectrogram (2D feature grid)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)

    # Convert magnitude to Decibels (dB) for standardization
    S_db = librosa.power_to_db(S, ref=np.max)

    # Naming and Saving
    output_filename = f"{track_id}_{genre_label.replace(' ', '_')}.png"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    # Plot and Save as an image
    plt.figure(figsize=(4, 4), frameon=False) 
    librosa.display.specshow(S_db, cmap='inferno')
    
    # Essential for CNN input: remove axes, padding, and title
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    return True


# 3. Main Processing Loop
if __name__ == '__main__':
    
    # Load and prepare data frame
    metadata_df = get_track_metadata(METADATA_PATH, AUDIO_DIR)
    
    # Apply the test limit
    if MAX_SAMPLES > 0 and MAX_SAMPLES < len(metadata_df):
        df_to_process = metadata_df.head(MAX_SAMPLES)
        print(f"Processing a test batch of {MAX_SAMPLES} tracks...")
    else:
        df_to_process = metadata_df
        print(f"Processing all {len(metadata_df)} tracks...")


    processed_count = 0
    # Iterate over the prepared DataFrame
    for index, row in tqdm(df_to_process.iterrows(), total=len(df_to_process), desc="Generating Spectrograms"):
        try:
            success = create_spectrogram_image(
                audio_path=row['file_path'],
                genre_label=row['genre_label'],
                track_id=row['track_id']
            )
            if success:
                processed_count += 1
                
        except Exception as e:
            print(f"\nSkipping Track ID {row['track_id']} due to error: {e}")
            
    print(f"\n✅ Processing complete. Total spectrograms saved: {processed_count} / {len(df_to_process)}")