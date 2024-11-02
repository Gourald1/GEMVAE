import torch
import torch.nn.functional as F

def contrastive_loss(z_i, z_j, temperature=0.5):
    z_i = F.normalize(z_i, dim=1)
    z_j = F.normalize(z_j, dim=1)

    representations = torch.cat([z_i, z_j], dim=0)
    similarity_matrix = torch.mm(representations, representations.T)

    labels = torch.cat([torch.arange(z_i.size(0))] * 2).to(z_i.device)
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(z_i.device)
    similarity_matrix = similarity_matrix[~mask].view(labels.shape[0], -1)
    similarity_matrix /= temperature

    loss = F.cross_entropy(similarity_matrix, labels)
    return loss
