import torch
from torch.nn import Module
import torch.nn.functional as F

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class ContrastiveLoss(Module):
    """
    Implements the NT-Xent loss function used in SimCLR. The reference for the implementation of this
    class are the following:
    https://github.com/google-research/simclr/blob/bfe07eed7f101ab51f3360100a28690e1bfbf6ec/objective.py#L38
    https://zablo.net/blog/post/understanding-implementing-simclr-guide-eli5-pytorch/
    """

    def __init__(self, temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.temperature = torch.tensor(temperature).to(DEVICE)

    def forward(self, emb_1, emb_2):
        batch_size = emb_1.shape[0]
        # matrix where each element of the diagonal are set to 0. All other elements are set to 1.
        # This is necessary because in the denominator you don't want to consider the positive pairs i.e. (i,i).
        mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float().to(DEVICE)

        # normalize the data to get more numerically stable results
        emb_1 = F.normalize(emb_1)
        emb_2 = F.normalize(emb_2)
        # concatenate the two representations to implement vectorization algorithm
        representations = torch.cat((emb_1, emb_2), dim=0)
        # compute the matrix that stores the similarity between each elements of the two representations
        # It is necessary to unsqueeze the dimensions to be able to perform the matrix multiplication (rows*columns)
        sim_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2).to(DEVICE)
        # As stated in the paper, the final loss is calculated considering both (i,j) and (j,i)
        sim_ij = torch.diag(sim_matrix, batch_size)
        sim_ji = torch.diag(sim_matrix, -batch_size)
        # concatenate for vectorization
        positive_pairs = torch.cat((sim_ij, sim_ji), dim=0).to(DEVICE)
        numerator = torch.exp(positive_pairs / self.temperature)
        denominator = mask * torch.exp(sim_matrix / self.temperature)
        loss = -torch.log(numerator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss) / batch_size * 2

        return loss