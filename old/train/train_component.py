import os
from pathlib import Path
from mldesigner import command_component, Input, Output

@command_component(
    name="train_sl_vqvae",
    version="1",
    display_name="Train SL VQ-VAE",
    description="train sign language vq-vae with pytorch lightning",
    environment=dict(
        conda_file=Path(__file__).parent / "conda.yaml",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
    ),
)
def train_component(
    input_data: Input(type="uri_folder"),   # or uri_file if zip file
    output_model: Output(type="uri_folder"),# guess this should still be folder though, since all models will be stored in a folder not a file
    epochs=10,
):
    # avoid dependency issue, execution logic is in train() func in train.py file
    from train import train
    # data: "/mnt/azureml/cr/j/da02c2086499434183577b4768bd46fc/cap/data-capability/wd/INPUT_npz_input/pose_0.npz"

    train(input_data, output_model, epochs)