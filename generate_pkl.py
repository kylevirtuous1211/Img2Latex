import os
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms

def preprocess(image_path, save_path, excel_path, max_len=150):
    """
    Creates a tensor-formula pair for each formula in the dataset
    and saves the tensor-formula pairs in a pickle file using torch.save().
    
    Args:
        image_path (str): Directory containing the processed images.
        save_path (str): Path to save the serialized tensor-formula pairs.
        excel_path (str): Path to the CSV file containing image names and formulas.
        max_len (int): Maximum number of tokens to include in each formula.
    """
    # Read the CSV file
    try:
        data = pd.read_csv(excel_path)
        print(f"Loaded {len(data)} records from {excel_path}")
    except FileNotFoundError:
        print(f"Error: The file {excel_path} does not exist.")
        return
    except pd.errors.EmptyDataError:
        print(f"Error: The file {excel_path} is empty.")
        return
    except Exception as e:
        print(f"An unexpected error occurred while reading {excel_path}: {e}")
        return

    tensor_image_pair = []
    transform = transforms.ToTensor()

    for index, row in data.iterrows():
        image_name = row.get('image')
        formula_row = row.get('formula')

        if pd.isna(image_name) or pd.isna(formula_row):
            print(f"Warning: Missing data at row {index}. Skipping.")
            continue

        full_image_path = os.path.join(image_path, image_name)

        if not os.path.isfile(full_image_path):
            print(f"Warning: Image file {full_image_path} does not exist. Skipping.")
            continue

        try:
            # Open the image and convert to RGB
            image = Image.open(full_image_path).convert('RGB')
        except Exception as e:
            print(f"Error opening image {full_image_path}: {e}. Skipping.")
            continue

        try:
            # Apply transformations to convert the image to a tensor
            image_tensor = transform(image)
        except Exception as e:
            print(f"Error transforming image {full_image_path}: {e}. Skipping.")
            continue

        # Truncate the formula to the maximum length
        processed_formula = " ".join(formula_row.split()[:max_len])

        tensor_image_pair.append((image_tensor, processed_formula))

        if (index + 1) % 1000 == 0:
            print(f"Processed {index + 1} / {len(data)} records.")

    # Ensure the save directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    try:
        # Save using torch.save() for compatibility with torch.load()
        torch.save(tensor_image_pair, save_path)
        print(f"Successfully saved {len(tensor_image_pair)} tensor-formula pairs to {save_path}")
    except Exception as e:
        print(f"Error saving tensor-formula pairs to {save_path}: {e}")

if __name__ == "__main__":
    # Define paths
    image_path = './formula_images_processed'  # Directory containing processed images
    save_dir = './'  # Directory where the pickle file will be saved
    excel_dir = 'data'  # Directory containing CSV files

    # Define splits
    splits = ['train', 'validate', 'test']

    for split in splits:
        save_filename = f"tensor_formula_pairs_filter_{split}.pkl"
        save_path = os.path.join(save_dir, save_filename)
        excel_filename = f"im2latex_{split}.csv"
        excel_path = os.path.join(excel_dir, excel_filename)
        
        print(f"\n--- Processing {split.capitalize()} Split ---")
        preprocess(image_path, save_path, excel_path, max_len=150)
