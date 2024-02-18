from dotenv import load_dotenv
import os
load_dotenv()

import torch 

class CFG:
    debug = os.getenv("DEBUG")
    image_path = os.getenv("IMAGE_PATH")
    captions_path = os.getenv("CAPTIONS_PATH")
    batch_size = int(os.getenv("BATCH_SIZE"))
    num_workers = int(os.getenv("NUM_WORKERS"))
    head_lr = float(os.getenv("HEAD_LR"))
    image_encoder_lr = float(os.getenv("IMAGE_ENCODER_LR"))
    text_encoder_lr = float(os.getenv("TEXT_ENCODER_LR"))
    weight_decay = float(os.getenv("WEIGHT_DECAY"))
    patience = int(os.getenv("PATIENCE"))
    factor = float(os.getenv("FACTOR"))
    epochs = int(os.getenv("EPOCHS"))
    device = torch.device(os.getenv("DEVICE"))
    model_name = os.getenv("MODEL_NAME")
    image_embedding = int(os.getenv("IMAGE_EMBEDDING"))
    text_encoder_model = os.getenv("TEXT_ENCODER_MODEL")
    text_embedding = int(os.getenv("TEXT_EMBEDDING"))
    text_tokenizer = os.getenv("TEXT_TOKENIZER")
    max_length = int(os.getenv("MAX_LENGTH"))
    pretrained = bool(os.getenv("PRETRAINED"))
    trainable = bool(os.getenv("TRAINABLE"))
    temperature = float(os.getenv("TEMPERATURE"))
    size = int(os.getenv("SIZE"))
    num_projection_layers = int(os.getenv("NUM_PROJECTION_LAYERS"))
    projection_dim = int(os.getenv("PROJECTION_DIM"))
    dropout = float(os.getenv("DROPOUT"))
