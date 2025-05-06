from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential

from azure.ai.ml import MLClient
from azure.ai.ml.dsl import pipeline
from azure.ai.ml import load_component

import os
from dotenv import load_dotenv

# Pipeline
from data_prep.prep_component import prepare_data_component
from train.train_component import train_component
keras_score_component = load_component(source="./score/score.yaml")
load_dotenv()


@pipeline()
def sl_vqvae_pipeline(pipeline_input_data, compute_cluster):
    """E2E image classification pipeline with keras using python sdk."""
    prepare_data_node = prepare_data_component(input_data=pipeline_input_data)

    train_node = train_component(
        input_data=prepare_data_node.outputs.training_data
    )
    train_node.compute = compute_cluster

    score_node = keras_score_component(
        input_data=prepare_data_node.outputs.test_data,
        input_model=train_node.outputs.output_model,
    )


if __name__ == "__main__":
    # Create azure ML client
    ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=os.getenv("SUBSCRIPTION_ID"),
        resource_group_name=os.getenv("RESOURCE_GROUP"),
        workspace_name=os.getenv("WORKSPACE_NAME")
    )

    # create a pipeline
    pipeline_job = sl_vqvae_pipeline(pipeline_input_data="pose_dataset.npz", compute_cluster="cpu_cluster")
    # run pipeline
    pipeline_job = ml_client.jobs.create_or_update(
        pipeline_job, experiment_name="pipeline_samples"
    )