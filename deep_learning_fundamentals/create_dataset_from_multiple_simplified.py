# new additions to the dataset
# https://www.kaggle.com/datasets/alistairking/recyclable-and-household-waste-classification
# https://www.kaggle.com/datasets/mostafaabla/garbage-classification
# https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2
# https://www.kaggle.com/datasets/zlatan599/garbage-dataset-classification
# https://www.kaggle.com/datasets/vencerlanz09/plastic-paper-garbage-bag-synthetic-images

# Ones that are already downloaded (previous scripts), but not combined fully yet
# https://www.kaggle.com/datasets/techsash/waste-classification-data
# https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification


from os import listdir, makedirs
import os
from os.path import join
import shutil
from tqdm import tqdm


final_classes = {
    'organic': 0,
    'battery': 1,
    'glass': 2,
    'metal': 3,
    'paper': 4,
    'cardboard': 5,
    'plastic': 6,
    'textiles': 7,
    'trash': 8,
}

mappings = {
    'recyclable-and-household-waste-classification': {
        'aerosol_cans': 'metal',
        'aluminum_food_cans': 'metal',
        'aluminum_soda_cans': 'metal',
        'cardboard_boxes': 'cardboard',
        'cardboard_packaging': 'cardboard',
        'clothing': 'textiles',
        'coffee_grounds': 'organic',
        'disposable_plastic_cutlery': 'plastic',
        'eggshells': 'organic',
        'food_waste': 'organic',
        'glass_beverage_bottles': 'glass',
        'glass_cosmetic_containers': 'glass',
        'glass_food_jars': 'glass',
        'magazines': 'paper',
        'newspaper': 'paper',
        'office_paper': 'paper',
        'paper_cups': 'paper',
        'plastic_cup_lids': 'plastic',
        'plastic_detergent_bottles': 'plastic',
        'plastic_food_containers': 'plastic',
        'plastic_shopping_bags': 'plastic',
        'plastic_soda_bottles': 'plastic',
        'plastic_straws': 'plastic',
        'plastic_trash_bags': 'plastic',
        'plastic_water_bottles': 'plastic',
        'shoes': 'textiles',
        'steel_food_cans': 'metal',
        'styrofoam_cups': 'plastic',
        'styrofoam_food_containers': 'plastic',
        'tea_bags': 'organic',
    },
    'waste-classification-data': {
        'O': 'organic',
    },
    'plastic-paper-garbage-bag-synthetic-images': {
        'Garbage Bag Images': 'plastic',
        'Paper Bag Images': 'paper',
        'Plastic Bag Images': 'plastic'
    },
    'garbage-dataset-classification': {
        'cardboard': 'cardboard',
        'glass': 'glass',
        'metal': 'metal',
        'paper': 'paper',
        'plastic': 'plastic',
        'trash': 'trash',
    },
    'garbage-classification-v2': {
        'battery': 'battery',
        'biological': 'organic',
        'cardboard': 'cardboard',
        'clothes': 'textiles',
        'glass': 'glass',
        'metal': 'metal',
        'paper': 'paper',
        'plastic': 'plastic',
        'shoes': 'textiles',
        'trash': 'trash',
    },
    'garbage-classification-12-classes': {
        'battery': 'battery',
        'biological': 'organic',
        'brown-glass': 'glass',
        'cardboard': 'cardboard',
        'clothes': 'textiles',
        'green-glass': 'glass',
        'metal': 'metal',
        'paper': 'paper',
        'plastic': 'plastic',
        'shoes': 'textiles',
        'trash': 'trash',
        'white-glass': 'glass',
    },
    'garbage_classification': {
        'cardboard': 'cardboard',
        'glass': 'glass',
        'metal': 'metal',
        'paper': 'paper',
        'plastic': 'plastic',
        'trash': 'trash',
    },
}

output_root = './datasets/combined_waste_dataset'

# step 1 - create directory structure for the combined dataset
# params - root_dir (output directory, main path), classes_dict (final classes dictionary)
# just main directories, no splitting and subdirectories

def create_directory_structure(root_dir, classes_dict): 
    for class_name in classes_dict.keys():
        class_dir = join(root_dir, class_name)
        makedirs(class_dir, exist_ok=True)
    print(f"Created directory structure in {root_dir}")


create_directory_structure(output_root, final_classes)

# step 2 - move images from each dataset to the combined directory structure
# create a separate function for each dataset (separate processing for clarity)
# functions are named according to the dataset they process (no "general function")

# first is the waste_classification_data dataset
# folder structure: two folders TRAIN and TEST, each with two subfolders O and R
# we will ignore the split and just move all images to the combined dataset
# moreover, we will map according to the "mappings" dictionary above (so ignore the R class, it is not split into recyclable types)

def process_waste_classification_data(source_dir, dest_dir_root, mapping):
    for split in ['TRAIN', 'TEST']:
        split_dir = join(source_dir, split)
        for class_folder in tqdm(listdir(split_dir), desc=f"Processing {split} of waste_classification_data"):
            class_dir = join(split_dir, class_folder)
            if class_folder in mapping:
                final_class = mapping[class_folder]
                final_class_dir = join(dest_dir_root, final_class)
                for img_file in tqdm(listdir(class_dir), desc=f"Processing images in {class_folder}"):
                    src_path = join(class_dir, img_file)
                    dest_path = join(final_class_dir, img_file)
                    if os.path.exists(dest_path):
                        # If file already exists, create a unique name
                        base, ext = os.path.splitext(img_file)
                        counter = 1
                        new_name = f"{base}_{counter}{ext}"
                        new_dest_path = join(final_class_dir, new_name)
                        while os.path.exists(new_dest_path):
                            counter += 1
                            new_name = f"{base}_{counter}{ext}"
                            new_dest_path = join(final_class_dir, new_name)
                        shutil.copy(src_path, new_dest_path)
                    else:
                        shutil.copy(src_path, dest_path)
    print(f"Processed waste_classification_data from {source_dir} "
          f"to {dest_dir_root}")


process_waste_classification_data(
    source_dir='./datasets/waste_classification_data',
    dest_dir_root=output_root,
    mapping=mappings['waste-classification-data']
)


# we can now do the same with the garbage_classification dataset (second of the original ones)
# the structure is easy, inside the folder we have named subfolders, full of images
# we can use the mapping dictionary again

def process_garbage_classification(source_dir, dest_dir_root, mapping):
    for class_folder in tqdm(listdir(source_dir), desc="Processing garbage_classification"):
        class_dir = join(source_dir, class_folder)
        if class_folder in mapping:
            final_class = mapping[class_folder]
            final_class_dir = join(dest_dir_root, final_class)
            for img_file in tqdm(listdir(class_dir), desc=f"Processing images in {class_folder}"):
                src_path = join(class_dir, img_file)
                dest_path = join(final_class_dir, img_file)
                if os.path.exists(dest_path):
                    # If file already exists, create a unique name
                    base, ext = os.path.splitext(img_file)
                    counter = 1
                    new_name = f"{base}_{counter}{ext}"
                    new_dest_path = join(final_class_dir, new_name)
                    while os.path.exists(new_dest_path):
                        counter += 1
                        new_name = f"{base}_{counter}{ext}"
                        new_dest_path = join(final_class_dir, new_name)
                    shutil.copy(src_path, new_dest_path)
                else:
                    shutil.copy(src_path, dest_path)
    print(f"Processed garbage_classification from {source_dir} "
          f"to {dest_dir_root}")


process_garbage_classification(
    source_dir='./datasets/garbage_classification',
    dest_dir_root=output_root,
    mapping=mappings['garbage_classification']
)

# # Process garbage-v2 dataset which contains battery and textiles categories
# def process_garbage_v2(source_dir, dest_dir_root, mapping):
#     for class_folder in tqdm(listdir(source_dir), desc="Processing garbage-v2"):
#         class_dir = join(source_dir, class_folder)
#         if os.path.isdir(class_dir) and class_folder in mapping:
#             final_class = mapping[class_folder]
#             final_class_dir = join(dest_dir_root, final_class)
#             for img_file in tqdm(listdir(class_dir), desc=f"Processing images in {class_folder}"):
#                 if img_file.endswith('.jpg'):
#                     src_path = join(class_dir, img_file)
#                     dest_path = join(final_class_dir, f"v2_{img_file}")  # dodajemy prefiks v2_ żeby uniknąć konfliktów nazw
#                     shutil.copy(src_path, dest_path)
#     print(f"Processed garbage-v2 from {source_dir} to {dest_dir_root}")

# # Process garbage-v2 dataset to get battery and textiles
# process_garbage_v2(
#     source_dir='./datasets/garbage-v2',
#     dest_dir_root=output_root,
#     mapping=mappings['garbage-classification-v2']
# )

# now new datasets
# first one is the recyclable-and-household-waste-classification
# it has a simple structure, in the root folder there is an  images subfolder, with another images subfolder
# after that there are subfolders named according to the waste types
# inside that there are two folders - "default" and "real_world". We can combine images from both folders
# again, we will use the mapping dictionary
