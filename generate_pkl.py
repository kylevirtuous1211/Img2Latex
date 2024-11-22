import os
import pandas as pd
from PIL import Image
import pickle
import torch
from torchvision import transforms


def preprocess(image_path, save_path, excel_path):
    # create a tensor, formula pair for each formula in the dataset
    # and save the tensor, formula pairs in a pickle file
    data = pd.read_csv(excel_path)

    tensor_image_pair = []
    transform = transforms.ToTensor()

    for index, row in data.iterrows():
        image_name = row['image']
        formula_row = row['formula']

        full_image_path = os.path.join(image_path, image_name)
        image = Image.open(full_image_path)

        image_tensor = transform(image)
        tensor_image_pair.append((image_tensor, formula_row))

    with open(save_path, 'wb') as f:
        pickle.dump(tensor_image_pair, f)


if __name__ == "__main__":
    image_path = './formula_images_processed'
    save_path = './tensor_formula_pairs_test.pkl'  # Specify the file name
    excel_path = '100k_excel_test_train_validate/im2latex_test.csv'
    preprocess(image_path, save_path, excel_path)
