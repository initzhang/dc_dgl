import dgl
import torch
import random
import numpy as np

def set_random_seeds(seed):
    dgl.seed(seed)
    dgl.random.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

print('==================')
print('test installation')
set_random_seeds(0)
rowptr, col, _ = dgl.rand_graph(1000,10000).adj_sparse(fmt='csc')
test_g = dgl.graph(('csc', (rowptr, col, torch.Tensor())))

try:
    test_g.pin_memory_()
    sep = 100
    cached_indptr = rowptr[:sep].cuda()
    cached_indices = col[:rowptr[sep]+1].cuda()
    seed_nodes = torch.arange(10).cuda()
    fanout = 10
    set_random_seeds(0)
    sg_ducati = dgl.sampling.sample_neighbors_with_cache(test_g, cached_indptr, cached_indices, seed_nodes, fanout)
    set_random_seeds(0)
    test_g_cuda = test_g.to(torch.device('cuda:0'))
    sg_dgl = dgl.sampling.sample_neighbors(test_g_cuda, seed_nodes, fanout)
    torch.cuda.synchronize()
    assert torch.equal(sg_ducati.edata['_ID'], sg_dgl.edata['_ID']), "inconsistent behaviors between DUCATI and DGL"
    print('test successful')
except Exception as error:
    print(error)
    print('test failed, please check installation again!')

print('==================')
