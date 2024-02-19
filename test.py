import os
import itertools
import numpy as np
import pandas as pd
import torch
from Clip.configs import CFG
from Clip.component import CLIPDataset, get_transforms
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer
from Clip.component import AvgMeter
from Clip.model import CLIPModel
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as F
dataset = "8k" 
config_info = CFG().get_config()


def build_loaders(dataframe, tokenizer, mode):
    transforms = get_transforms(mode=mode, config_info = config_info)
    dataset = CLIPDataset(
        dataframe["image"].values,
        dataframe["caption"].values,
        tokenizer=tokenizer,
        transforms=transforms,
        config_info = config_info
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config_info.batch_size,
        # num_workers=config_info.num_workers,
        shuffle=True if mode == "train" else False,
    )
    return dataloader


def get_image_embeddings(valid_df, model_path):
    tokenizer = DistilBertTokenizer.from_pretrained(config_info.text_tokenizer)
    valid_loader = build_loaders(valid_df, tokenizer, mode="valid")

    model = CLIPModel().to(config_info.device)
    model.load_state_dict(torch.load(model_path, map_location=config_info.device))
    model.eval()

    valid_image_embeddings = []
    with torch.no_grad():
        for batch in tqdm(valid_loader):
            image_features = model.image_encoder(batch["image"].to(config_info.device))
            image_embeddings = model.image_projection(image_features)
            valid_image_embeddings.append(image_embeddings)
    return model, torch.cat(valid_image_embeddings)




def make_train_valid_dfs():
    dataframe = pd.read_csv(f"./{config_info.captions_path}/captions.csv")
    max_id = dataframe["id"].max() + 1 if not config_info.debug else 100
    image_ids = np.arange(0, max_id)
    np.random.seed(42)
    valid_ids = np.random.choice(
        image_ids, size=int(0.2 * len(image_ids)), replace=False
    )
    train_ids = [id_ for id_ in image_ids if id_ not in valid_ids]
    train_dataframe = dataframe[dataframe["id"].isin(train_ids)].reset_index(drop=True)
    valid_dataframe = dataframe[dataframe["id"].isin(valid_ids)].reset_index(drop=True)
    return train_dataframe, valid_dataframe


def find_matches(model, image_embeddings, query, image_filenames, n=9):
    tokenizer = DistilBertTokenizer.from_pretrained(config_info.text_tokenizer)
    encoded_query = tokenizer([query])
    batch = {
        key: torch.tensor(values).to(config_info.device)
        for key, values in encoded_query.items()
    }
    with torch.no_grad():
        text_features = model.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        text_embeddings = model.text_projection(text_features)

    image_embeddings_n = F.normalize(image_embeddings, p=2, dim=-1)
    text_embeddings_n = F.normalize(text_embeddings, p=2, dim=-1)
    dot_similarity = text_embeddings_n @ image_embeddings_n.T

    values, indices = torch.topk(dot_similarity.squeeze(0), n * 5)
    matches = [image_filenames[idx] for idx in indices[::5]]

    _, axes = plt.subplots(3, 3, figsize=(10, 10))
    for match, ax in zip(matches, axes.flatten()):
        image = cv2.imread(f"{config_info.image_path}/{match}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax.imshow(image)
        ax.axis("off")

    plt.show()
   
        
def main():
    _, valid_df = make_train_valid_dfs()
    model, image_embeddings = get_image_embeddings(valid_df, "content/best.pt")  
    find_matches(model,
             image_embeddings,
             query="boy playing with dog",
             image_filenames=valid_df['image'].values,
             n=9)
if __name__ == "__main__":
    main()