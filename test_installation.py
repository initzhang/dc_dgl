import dgl
import torch
print('==================')
print('test installation')
g = dgl.rand_graph(1000,10000).formats(['csc'])
g.pin_memory_()
assert g.is_pinned()
rowptr, col, _ = g.adj_sparse(fmt='csc')
cached_indptr = rowptr[:100].cuda()
cached_indices = col[:100].cuda()
seed_nodes = torch.arange(10).cuda()
fanout = 10
try:
    sg = dgl.sampling.sample_neighbors_with_cache(g, cached_indptr, cached_indices, seed_nodes, fanout)
    print('test successful') 
except Exception as error:
    print(error)
    print('test failed, please check installation again!')
print('==================')
