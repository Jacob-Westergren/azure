from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.storage.blob import BlobServiceClient
import os
from pathlib import Path
import glob
from dotenv import load_dotenv

load_dotenv()

def test_uploaded_data():
    credential = DefaultAzureCredential()

    # Initialize MLClient
    ml_client = MLClient(
        credential=credential,
        subscription_id=os.getenv("SUBSCRIPTION_ID"),
        resource_group_name=os.getenv("RESOURCE_GROUP"),
        workspace_name=os.getenv("WORKSPACE_NAME")
    )
    # Download the registered data asset
    dataset_path = ml_client.data.download(
        name="npz_dataset",
        download_path="downloaded_npz_data",  # Local path to save the files
    )

def upload_to_datastore():
    credential = DefaultAzureCredential()
    
    # Initialize MLClient
    ml_client = MLClient(
        credential=credential,
        subscription_id=os.getenv("SUBSCRIPTION_ID"),
        resource_group_name=os.getenv("RESOURCE_GROUP"),
        workspace_name=os.getenv("WORKSPACE_NAME")
    )

    # Retrieve datastore (which by default is the blob container)
    datastore = ml_client.datastores.get_default()  # accesses datastore mlsldev323963's container azureml-blobstore-7380dda4-19e4-4992-9ad6-d6b5ed3fdebb

    # Initialize BlobServiceClient and create client for interacting with blob container
    blob_service_client = BlobServiceClient(
        account_url=f"https://{datastore.account_name}.blob.core.windows.net",  # datastore.account_name= terraform's storage_account_name = "mlsldev323963"
        credential=credential
    )
    container_client = blob_service_client.get_container_client(datastore.container_name)
    
    # Local directory containing NPZ files
    local_data_path = "data/"  # Update this path to your local NPZ files directory
    remote_path = "npz_data"  # Path in blob storage where files will be uploaded
    
    # Upload each NPZ file
    # npz_files = glob.glob(local_data_path/*)
    npz_files = glob.glob(os.path.join(local_data_path, "*poseheader"))

    progress_bar = tqdm(npz_files, desc="Uploading")
    
    for npz_file in progress_bar:
        progress_bar.set_description(f"Uploading {os.path.basename(npz_file)}")
    
        blob_name = os.path.join(remote_path, os.path.basename(npz_file))
        with open(npz_file, "rb") as data:
            container_client.upload_blob(name=blob_name, data=data, overwrite=True)

    # Register the data asset
    data_asset = Data(
        name="npz_dataset",
        path=f"azureml://datastores/{datastore.name}/paths/{remote_path}",
        type="uri_folder",
        description="Dataset containing NPZ files for model training"
    )
    
    ml_client.data.create_or_update(data_asset)
    print("Successfully registered data asset")

# found this, can maybe be used to compute velocity easier, as i can get the distances using this and just handle outliers like fredrik did with threshold
# https://github.com/sign-language-processing/pose/blob/master/src/python/pose_format/numpy/representation/distance.py

from pose_format.utils.generic import fake_pose
from pathlib import Path
import numpy as np
import io
from tqdm import tqdm
def save_header(output_directory: str= "data"):
    pose = fake_pose(num_frames=(1))
    header_path = Path(output_directory) / "poseheader"
    with open(header_path, "wb") as f:
        pose.header.write(f)

def save_poses_to_folder(output_directory: str = "data"):
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    poses = [fake_pose(num_frames=20000) for _ in range(10)]

    for idx, pose in enumerate(tqdm(poses, desc="Processing poses")):
        # Only take the first person's data
        data = pose.body.data[:, 0, :, :]
        float16_data = data.filled(0).astype(np.float16)

        # Create corresponding .npz filename
        npz_filename = output_path / f"pose_{idx}.npz"

        # Save as compressed npz
        np.savez_compressed(npz_filename, data=float16_data, mask=data.mask)

    print(f"Saved {len(poses)} files to {output_path}")


if __name__ == "__main__":
    #save_poses_to_folder()
    #save_header()
    upload_to_datastore() 