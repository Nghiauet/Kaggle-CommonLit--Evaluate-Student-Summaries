import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi
import subprocess
import json
import random
# Kaggle stores token at this path as default
KAGGLE_TOKEN_PATH = os.path.join(os.path.expanduser("~"), ".kaggle/kaggle.json")

with open(KAGGLE_TOKEN_PATH) as token_file:
    token = json.load(token_file)

username = token['username']
print('username: ', username)
def zip_dir(directory_path, zip_path):
    # Use a system "zip -r" command to create a zip file
    subprocess.run(["zip", "-r", zip_path, directory_path])

def create_metadata_file(dataset_folder, dataset_id, username, title):
    dataset_id = dataset_id.replace(' ', '-')
    dataset_id = dataset_id.replace('_', '-')
    dataset_id = dataset_id.replace('.', '-')
    title = title.replace('_', '-')
    title = title.replace('.', '-')
    # remove special characters
    metadata = {
        "licenses": [{"name": "CC0-1.0"}],
        # replace all _ by - in dataset_id
        "id": f"{username}/{dataset_id.replace('_', '-')}",
        # add rundom number to title
        "title": f"{title}",
    }
    
    with open(os.path.join(dataset_folder, 'dataset-metadata.json'), 'w') as f:
        json.dump(metadata, f)

def push_to_kaggle(zip_directory, dataset_folder, description, dataset_name, username):
    # Initialize Kaggle API
    api = KaggleApi()
    api.authenticate()
    full_dataset_name = f"{username}/{dataset_name}"
    
    # Create metadata file
    create_metadata_file(zip_directory, dataset_name, username, description)

    # check if dataset have existed
    try :
        command = f"kaggle datasets create -p {zip_directory}/ "
        os.system(command)
    except Exception as e:
        print(e)
        print(f"Dataset {full_dataset_name} already exists.")


list_folders = [
                # "1.debertav3base_lr2e-05_save_bs19", DONE
                # 1.debertav3base_lr16e-05_save_bs18 DONE
                # "1.debertav3base_lr16e-05_save_bs19", DONE
                # "debertav3base_lr18e-05" , DONE
                # "debertav3base_lr17e-05", DONE
                # "debertav3base_lr15e-05", DONE
                # "debertav3base_lr21e-05", DONE
                # "debertav3base_lr22e-05",DONE
                # "debertav3base_lr5e-05", DONE
                
                # "debertav3large_lr12e-05", #wating uploading 
                # "debertav3large_lr13e-05", #watting uploading
                # "debertav3large_lr1e-05", #watting uploading
                # "debertav3large_lr15e-05" , #uploading
                # "1.debertav3large_lr16e-05_save_bs_19" DONE error when mcrse = 51
                # 'debertav3large_lr8e-06_att_0007', # upload 
                # 'debertav3large_lr9e-06_att_0007', # upload 
                # 'debertav3large_lr1e-05_att_0007', # upload 
                # 'debertav3large_lr11e-05_att_0007',
                # 'debertav3large_lr12e-05_att_0007',
                # 'debertav3large_lr13e-05_att_0007',
                # 'debertav3large_lr14e-05_att_0007',
                # 'debertav3large_lr15e-05_att_0007', # upload 
                # 'debertav3large_lr16e-05_att_0007', # upload 
                # 'debertav3large_lr17e-05_att_0007', # upload 
                # 'debertav3large_lr18e-05_att_0007' # upload 
                # 'debertav3large_lr6e-06_att_0006',
                # 'debertav3large_lr7e-06_att_0006',
                # 'debertav3large_lr8e-06_att_0006',
                'debertav3large_lr12e-05_clean_text',
                'debertav3large_lr7e-06_clean_text',
                'debertav3large_lr8e-06_clean_text',
                ]  

for folder in list_folders:
    zip_directory = os.path.abspath(f"zip-of-{folder}")  # Folder to save zip files
    if not os.path.exists(zip_directory):
        os.makedirs(zip_directory)

    zip_path = os.path.join(zip_directory, f"{folder}.zip")
    print(f"Zipping {folder} to {zip_path}")
    # zip_dir(folder, zip_path) 
    print(f"Pushing {zip_directory} to Kaggle")
    push_to_kaggle(zip_directory, folder, f"CL_{folder}", f"CL_{folder}", username)