from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.ai.ml import MLClient, load_component, Input
from azure.ai.ml.entities import Data, Environment
from azure.ai.ml.dsl import pipeline
import os
from dotenv import load_dotenv

# Good source: https://github.com/Azure/azureml-examples/blob/main/cli/jobs/pipelines-with-components/nyc_taxi_data_regression/env_train/Dockerfile

load_dotenv()

@pipeline()
def pose_pipeline(
    data_dir: Input,
    max_pose_length: int = 512,
    batch_size: int = 16,
    codebook_size: int = 1024,
    num_codebooks: int = 4,
    num_layers: int = 4,
    lr: float = 0.0001,
    steps: int = 10,  # Small number for debugging
    device: str = "cpu",
    loss_hand_weight: float = 1.0
):  
    train_step = train_component(
        data_dir=data_dir,
        max_pose_length=max_pose_length,
        batch_size=batch_size,
        codebook_size=codebook_size,
        num_codebooks=num_codebooks,
        num_layers=num_layers,
        lr=lr,
        steps=steps,
        device=device,
        loss_hand_weight=loss_hand_weight
    )
    train_step.compute = "cpu-cluster"

    eval_step = eval_component(
        model_dir=train_step.outputs.model_dir,
        data_dir=data_dir,
        batch_size=batch_size,
        device=device
    )
    eval_step.compute = "cpu-cluster"
    
    return {
        "model_dir": train_step.outputs.model_dir
    }

# TODO - Check if mlflow stored model is truly the same as the pytorch model, i.e. that it has loss weights, learning rate, etc. 

if __name__ == "__main__":
    # Load component    
    print(os.getenv("WORKSPACE_NAME"))
    print(os.getenv("RESOURCE_GROUP"))
    print(os.getenv("SUBSCRIPTION_ID"))
    
    ml_client = MLClient(
            credential=DefaultAzureCredential(),
            subscription_id=os.getenv("SUBSCRIPTION_ID"),
            resource_group_name=os.getenv("RESOURCE_GROUP"),
            workspace_name=os.getenv("WORKSPACE_NAME")
    )
    print("Connected")
    print(ml_client.workspace_name)
    # mcr.microsoft.com/azureml/curated/acpt-pytorch-1.13-py38-cuda11.7-gpu:10
    # mcr.microsoft.com/azureml/curated/acpt-pytorch-2.2-cuda12.1:31
    #image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
    # Create new environment
    # add     - vector-quantize-pytorch later
    # future not for me, if it cant find defautl credentials even though everything in the env works well adn it can connect to ml client, just relogin
    env = Environment(  
        name="pose-training-env",
        image="mcr.microsoft.com/azureml/curated/acpt-pytorch-2.2-cuda12.1:31",
        conda_file="conda.yml",
        description="Curated PyTorch environment with small additions."
    )
    print(f"Creating environment...")
    train_component = ml_client.environments.create_or_update(env)
    #if False:
    # Creating/Loading the components
    print("Loading components...")
    train_component = load_component(source="train-model.yml")
    eval_component = load_component(source="eval-model.yml")
    print("Components loaded, registering...")
    train_component = ml_client.components.create_or_update(train_component)
    eval_component = ml_client.components.create_or_update(eval_component)
    print("Components registered successfully")

    # Instantiate pipeline
    print("Creating pipeline...")
    pipeline_job = pose_pipeline(
        data_dir=Input(
            type="uri_folder",
            path="azureml:npz_dataset:2",
            mode="mount"
        ),
        max_pose_length=512,
        batch_size=16,
        codebook_size=1024,
        num_codebooks=1,
        num_layers=6,
        lr=0.0001,
        steps=15,  # Small number for debugging
        device="cpu",
        loss_hand_weight=1.0
    )

    # Submit pipeline
    print("Submitting pipeline...")
    job = ml_client.jobs.create_or_update(pipeline_job, experiment_name="pose_training_pipeline")
    
    # Print the job name for reference
    print(f"Job submitted with name: {job.name}")
    
    # After the job completes, you can check the output using:
    # check_model_output(job.name)
