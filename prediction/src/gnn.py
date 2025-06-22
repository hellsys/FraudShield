import torch
import torch.nn.functional as F
from torch_geometric.nn import GINEConv


class EdgeGNNClassifier(torch.nn.Module):
    """GNN → MLP, предсказывающий вероятность «фрод» для рёбер."""

    def __init__(self, in_feats: int, edge_feats: int, hidden: int = 64) -> None:
        super().__init__()

        self.conv1 = GINEConv(
            torch.nn.Sequential(
                torch.nn.Linear(in_feats, hidden),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden, hidden),
            ),
            edge_dim=edge_feats,
        )
        self.conv2 = GINEConv(
            torch.nn.Sequential(
                torch.nn.Linear(hidden, hidden),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden, hidden),
            ),
            edge_dim=edge_feats,
        )

        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden + edge_feats, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1),
        )

    def forward(  # noqa: D401
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        src, _ = edge_index
        h_edge = torch.cat([x[src], edge_attr], dim=1)
        return self.edge_mlp(h_edge).view(-1)
