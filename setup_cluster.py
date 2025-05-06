from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import AmlCompute, Data
from azure.ai.ml.constants import AssetTypes
import os
from dotenv import load_dotenv

def setup_compute():
    # Initialize MLClient, allowing us to communicate with azure
    load_dotenv()
    ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=os.getenv("SUBSCRIPTION_ID"),
        resource_group_name=os.getenv("RESOURCE_GROUP"),
        workspace_name=os.getenv("WORKSPACE_NAME")
    )

    # Define the compute cluster
    cluster = AmlCompute(
        name="gpu-cluster",
        type="amlcompute",
        size="Standard_NC4as_T4_v3",  
        min_instances=0,  # Scale to 0 when not in use to save costs
        max_instances=1,  # Only need 1 for debugging
        idle_time_before_scale_down=1800,  # 30 minutes
    )

    # Create or update the compute cluster
    try:
        ml_client.compute.begin_create_or_update(cluster)
        print("Successfully created/updated cluster")
    except Exception as e:
        print(f"Error creating compute cluster: {e}")
    return 


if __name__ == "__main__":
    setup_compute() 