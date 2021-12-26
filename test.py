import numpy as np
import torch

if __name__ == '__main__':
    q_vectors = torch.from_numpy(np.random.rand(5, 10))
    p_vectors = torch.from_numpy(np.random.rand(5, 10))
    n_vectors = torch.from_numpy(np.random.rand(5, 10))

    p_cos_dis = torch.mm(q_vectors, p_vectors.t())
    p_cos_dis, p_cos_dis_rank = torch.sort(p_cos_dis, dim=1, descending=True)

    n_cos_dis = torch.mm(q_vectors, n_vectors.t())
    n_cos_dis, n_cos_dis_rank = torch.sort(n_cos_dis, dim=1, descending=True)

    p_cos_dis, p_cos_dis_rank = p_cos_dis.cpu().numpy(), p_cos_dis_rank.cpu().numpy()
    n_cos_dis, n_cos_dis_rank = n_cos_dis.cpu().numpy(), n_cos_dis_rank.cpu().numpy()

    a = p_cos_dis_rank[0, 2]

    print('xxx')
