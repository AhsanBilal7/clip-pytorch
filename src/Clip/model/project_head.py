from torch import nn
from Clip.configs import CFG

config_info = CFG().get_config()
class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=config_info.projection_dim,
        dropout=config_info.dropout
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x