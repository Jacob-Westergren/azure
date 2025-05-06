import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer
import numpy as np
import os
from tqdm import tqdm
import logging
import argparse

from mymodifiedhubert import WavLM, WavLMConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LayerWeightedFeatures(nn.Module):
    """
    Module to compute weighted sum of features from all encoder layers.
    """
    def __init__(self, num_layers, hidden_dim):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, layer_features):
        """
        Computes weighted sum of features from all layers.
        
        Args:
            layer_features: List of tensors containing layer outputs
                            Each tensor has shape [batch_size, seq_len, hidden_dim]
        
        Returns:
            Weighted sum of layer features with shape [batch_size, seq_len, hidden_dim]
        """
        # Normalize weights with softmax for better stability
        norm_weights = torch.softmax(self.weights, dim=0)
        
        # Weighted sum of layer outputs
        weighted_sum = torch.zeros_like(layer_features[0])
        for i, features in enumerate(layer_features):
            weighted_sum += norm_weights[i] * features
        
        # Apply layer normalization
        return self.layer_norm(weighted_sum)


class SignLanguageTranslationModel(nn.Module):
    """
    End-to-end model for sign language translation using a pretrained WavLM encoder
    and ByteT5 decoder.
    """
    def __init__(self, wavlm_model, bytet5_model, freeze_encoder=True, use_layer_aggregation=False):
        super().__init__()
        self.encoder = wavlm_model
        self.decoder = bytet5_model
        self.use_layer_aggregation = use_layer_aggregation
        
        if use_layer_aggregation:
            # Add weighted layer aggregation
            self.layer_aggregation = LayerWeightedFeatures(
                num_layers=wavlm_model.cfg.encoder_layers + 1,  # +1 for the input embedding
                hidden_dim=wavlm_model.cfg.encoder_embed_dim * len(wavlm_model.cfg.group_dims)
            )
        
        self.projection = nn.Linear(
            wavlm_model.cfg.encoder_embed_dim * len(wavlm_model.cfg.group_dims),
            bytet5_model.config.d_model
        )
        
        # Freeze encoder parameters if specified
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
    
    def forward(self, sign_input, padding_mask=None, text_input_ids=None, text_labels=None):
        # Extract features from the sign language encoder
        if self.use_layer_aggregation:
            # Get outputs from all layers
            encoder_output, layer_results = self.encoder.extract_features(
                source=sign_input,
                padding_mask=padding_mask,
                mask=False,
                ret_conv=False,
                ret_layer_results=True
            )
            
            # Extract layer features
            layer_features = [result[0] for result in layer_results]
            
            # Handle the encoder output based on its type
            if isinstance(encoder_output, dict):
                weighted_features = self.layer_aggregation(layer_features)
            else:
                layer_features.append(encoder_output)
                weighted_features = self.layer_aggregation(layer_features)
        else:
            # Use only final layer output
            encoder_output, _ = self.encoder.extract_features(
                source=sign_input,
                padding_mask=padding_mask,
                mask=False,
                ret_conv=False,
                output_layer=None,
                ret_layer_results=False
            )
            
            # Handle the case where extract_features returns a dict with loss info
            if isinstance(encoder_output, dict):
                encoder_output = torch.cat([
                    encoder_output.get("x", torch.zeros(0))  # Fallback to empty tensor if x doesn't exist
                ], dim=-1)
            
            weighted_features = encoder_output
        
        # Project to the decoder dimension
        projected_features = self.projection(weighted_features)
        
        # Pass to the ByteT5 decoder
        outputs = self.decoder(
            encoder_outputs=projected_features.unsqueeze(0),  # ByteT5 expects [batch_size, seq_len, hidden_dim]
            input_ids=text_input_ids,
            labels=text_labels
        )
        
        return outputs


class StockholmSignLanguageDataset(Dataset):
    """
    Dataset for loading sign language data from Stockholm University.
    """
    def __init__(self, data_dir, tokenizer, max_length=128):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data entries (will be specific to the Stockholm dataset format)
        self.data_entries = self._load_data_entries()
        
    def _load_data_entries(self):
        """
        Load data entries from the Stockholm dataset.
        Modify this according to the actual structure of your dataset.
        """
        # This is a placeholder - you'll need to modify based on actual data format
        entries = []
        
        # Example structure (modify based on your actual data):
        # Walk through the dataset directory and collect data
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith('.npy'):  # Assuming keypoint data is stored as numpy arrays
                    # Find corresponding annotation file
                    base_name = os.path.splitext(file)[0]
                    annotation_file = os.path.join(root, f"{base_name}.txt")
                    
                    if os.path.exists(annotation_file):
                        # Read the annotation (text translation)
                        with open(annotation_file, 'r', encoding='utf-8') as f:
                            text = f.read().strip()
                        
                        entries.append({
                            'keypoint_file': os.path.join(root, file),
                            'text': text
                        })
        
        logger.info(f"Loaded {len(entries)} entries from the dataset")
        return entries
    
    def __len__(self):
        return len(self.data_entries)
    
    def __getitem__(self, idx):
        entry = self.data_entries[idx]
        
        # Load keypoint data
        keypoints = np.load(entry['keypoint_file'])
        
        # Convert to proper format (adjust based on your model's input requirements)
        # The input should be structured based on group_dims in your WavLM config
        # Example: [batch, channels, sequence_length]
        keypoints = torch.tensor(keypoints, dtype=torch.float32)
        
        # Tokenize the text
        text = entry['text']
        tokenized_text = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            'keypoints': keypoints,
            'input_ids': tokenized_text.input_ids.squeeze(),
            'attention_mask': tokenized_text.attention_mask.squeeze(),
            'labels': tokenized_text.input_ids.squeeze()  # Use input_ids as labels for sequence-to-sequence tasks
        }


def save_checkpoint(model, optimizer, epoch, save_path):
    """Save model checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, save_path)
    logger.info(f"Checkpoint saved to {save_path}")


def train(args):
    # Load pretrained models
    logger.info("Loading pretrained models...")
    
    # Load your custom WavLM model
    config = WavLMConfig()
    # Update config with any fine-tuning specific parameters
    config.update({
        "mask_prob": 0.0,  # No masking during fine-tuning
        "dropout": 0.1,    # Adjust dropout as needed
    })
    
    # Load the pretrained WavLM model
    wavlm_model = WavLM(config)
    if args.wavlm_checkpoint:
        checkpoint = torch.load(args.wavlm_checkpoint, map_location='cpu')
        wavlm_model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded WavLM checkpoint from {args.wavlm_checkpoint}")
    
    # Load ByteT5 model with label smoothing
    bytet5_tokenizer = T5Tokenizer.from_pretrained("google/bytet5-small")
    bytet5_model = T5ForConditionalGeneration.from_pretrained(
        "google/bytet5-small", 
        label_smoothing=args.label_smoothing  # Add label smoothing
    )
    
    # Create the translation model
    model = SignLanguageTranslationModel(
        wavlm_model=wavlm_model,
        bytet5_model=bytet5_model,
        freeze_encoder=args.freeze_encoder,
        use_layer_aggregation=args.use_layer_aggregation
    )
    
    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Create dataset and dataloader
    train_dataset = StockholmSignLanguageDataset(
        data_dir=args.train_data,
        tokenizer=bytet5_tokenizer,
        max_length=args.max_length
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    # Setup optimizer with different learning rates
    # Separate parameters into encoder and decoder groups
    encoder_params = []
    decoder_params = []
    
    # Include only unfrozen encoder parameters (if any)
    if not args.freeze_encoder:
        encoder_params.extend([p for p in model.encoder.parameters() if p.requires_grad])
    
    # Add projection layer to encoder parameters
    encoder_params.extend([p for p in model.projection.parameters()])
    
    # If using layer aggregation, add those parameters to encoder group
    if args.use_layer_aggregation:
        encoder_params.extend([p for p in model.layer_aggregation.parameters()])
    
    # Add decoder parameters
    decoder_params.extend([p for p in model.decoder.parameters()])
    
    # Create optimizer with parameter groups
    optimizer = optim.AdamW([
        {'params': encoder_params, 'lr': args.encoder_lr, 'weight_decay': args.weight_decay},
        {'params': decoder_params, 'lr': args.decoder_lr, 'weight_decay': args.weight_decay}
    ])
    
    # Setup learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.num_epochs * len(train_loader),
        eta_min=0
    )
    
    # Training loop
    logger.info("Starting training...")
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for batch in progress_bar:
            # Move batch to device
            keypoints = batch['keypoints'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(
                sign_input=keypoints,
                text_input_ids=input_ids,
                text_labels=labels
            )
            
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            
            # Update progress bar
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
        
        # Log epoch results
        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1}/{args.num_epochs}, Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                save_path=os.path.join(args.output_dir, f"checkpoint-epoch-{epoch+1}.pt")
            )
    
    # Save final model
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=args.num_epochs,
        save_path=os.path.join(args.output_dir, "final_model.pt")
    )
    
    logger.info("Training completed!")


def translate(model, tokenizer, keypoints, device, max_length=128, beam_width=4):
    """
    Perform translation with a trained model.
    """
    model.eval()
    keypoints = keypoints.to(device)
    
    with torch.no_grad():
        # Get encoder features (handling layer aggregation if needed)
        if model.use_layer_aggregation:
            # Get outputs from all layers
            encoder_output, layer_results = model.encoder.extract_features(
                source=keypoints,
                padding_mask=None,
                mask=False,
                ret_conv=False,
                ret_layer_results=True
            )
            
            # Extract layer features
            layer_features = [result[0] for result in layer_results]
            
            # Handle the encoder output based on its type
            if isinstance(encoder_output, dict):
                weighted_features = model.layer_aggregation(layer_features)
            else:
                layer_features.append(encoder_output)
                weighted_features = model.layer_aggregation(layer_features)
        else:
            # Use only final layer output
            encoder_output, _ = model.encoder.extract_features(
                source=keypoints,
                padding_mask=None,
                mask=False,
                ret_conv=False
            )
            
            # Handle the case where extract_features returns a dict
            if isinstance(encoder_output, dict):
                encoder_output = torch.cat([
                    encoder_output.get("x", torch.zeros(0, device=device))
                ], dim=-1)
                
            weighted_features = encoder_output
        
        # Project features
        projected_features = model.projection(weighted_features)
        
        # Generate text using the ByteT5 model with specified beam width
        outputs = model.decoder.generate(
            encoder_outputs=projected_features.unsqueeze(0),
            max_length=max_length,
            num_beams=beam_width,
            early_stopping=True
        )
        
        # Decode the generated ids
        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
    return translated_text


def inference(args):
    """Run inference with a trained model."""
    # Load models and tokenizer
    logger.info("Loading model for inference...")
    
    # Load your custom WavLM model
    config = WavLMConfig()
    wavlm_model = WavLM(config)
    
    # Load ByteT5 model and tokenizer
    bytet5_tokenizer = T5Tokenizer.from_pretrained("google/bytet5-small")
    bytet5_model = T5ForConditionalGeneration.from_pretrained("google/bytet5-small")
    
    # Create the translation model
    model = SignLanguageTranslationModel(
        wavlm_model=wavlm_model,
        bytet5_model=bytet5_model,
        use_layer_aggregation=args.use_layer_aggregation
    )
    
    # Load trained model checkpoint
    checkpoint = torch.load(args.model_checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded model checkpoint from {args.model_checkpoint}")
    
    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Load test data
    test_dataset = StockholmSignLanguageDataset(
        data_dir=args.test_data,
        tokenizer=bytet5_tokenizer,
        max_length=args.max_length
    )
    
    # Run inference on test data
    results = []
    for i, sample in enumerate(tqdm(test_dataset, desc="Translating")):
        keypoints = sample['keypoints'].unsqueeze(0)  # Add batch dimension
        translated_text = translate(
            model=model,
            tokenizer=bytet5_tokenizer,
            keypoints=keypoints,
            device=device,
            max_length=args.max_length,
            beam_width=args.beam_width
        )
        
        ground_truth = bytet5_tokenizer.decode(sample['labels'], skip_special_tokens=True)
        
        results.append({
            'sample_id': i,
            'prediction': translated_text,
            'ground_truth': ground_truth
        })
        
        if i < 5:  # Print a few examples
            logger.info(f"Sample {i}:")
            logger.info(f"  Prediction: {translated_text}")
            logger.info(f"  Ground truth: {ground_truth}")
    
    # Save results
    import json
    with open(os.path.join(args.output_dir, "translation_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved results to {os.path.join(args.output_dir, 'translation_results.json')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune WavLM model for sign language translation")
    
    # Common arguments
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Directory to save outputs")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length for tokenization")
    
    # Model configuration
    parser.add_argument("--use_layer_aggregation", action="store_true", help="Use weighted sum of all encoder layers")
    parser.add_argument("--label_smoothing", type=float, default=0.2, help="Label smoothing factor (0.0 to disable)")
    parser.add_argument("--beam_width", type=int, default=5, help="Beam width for decoding")
    
    # Training specific arguments
    parser.add_argument("--train", action="store_true", help="Run training")
    parser.add_argument("--train_data", type=str, help="Path to training data directory")
    parser.add_argument("--wavlm_checkpoint", type=str, help="Path to pretrained WavLM checkpoint")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--encoder_lr", type=float, default=5e-5, help="Learning rate for encoder")
    parser.add_argument("--decoder_lr", type=float, default=5e-4, help="Learning rate for decoder")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm for clipping")
    parser.add_argument("--save_interval", type=int, default=1, help="Save checkpoint every N epochs")
    parser.add_argument("--freeze_encoder", action="store_true", help="Freeze the encoder parameters")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    
    # Inference specific arguments
    parser.add_argument("--inference", action="store_true", help="Run inference")
    parser.add_argument("--test_data", type=str, help="Path to test data directory")
    parser.add_argument("--model_checkpoint", type=str, help="Path to finetuned model checkpoint")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.train:
        train(args)
    elif args.inference:
        inference(args)
    else:
        logger.error("Please specify either --train or --inference") 