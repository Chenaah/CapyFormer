
import copy
import pdb
import numpy as np


num_modules = 5
cfg_table = {}
cfg_table_inverse = {}
ctr = 1 # 0: empty pipeline
for parent_id in range(num_modules):
    for posa in range(18):
        for posb in range(9):
            for r in range(3):
                pipeline = [parent_id, posa, posb, r]
                cfg_table[tuple(pipeline)] = ctr
                cfg_table_inverse[ctr] = pipeline
                ctr += 1
cfg_table_inverse[0] = []
# Max idx: 3430

def pipeidx_encode(pipeline):
    pipeidx = []
    for step in np.reshape(pipeline, (-1, 4)):
        # pdb.set_trace()
        pipeidx.append(cfg_table[tuple(step)])
    if len(pipeidx) < num_modules-1:
        pipeidx += [0] * (num_modules-1 - len(pipeidx))
    return np.array(pipeidx)

def pipeidx_decode(tree):
    pipeline = []
    for node_id in tree:
        pipeline.extend(cfg_table_inverse[node_id])
    return np.array(pipeline)

# def sort_tree_nodes(tree: list):
#     # pipelines = pipeidx_decode(tree).reshape(-1, 4)
#     tree = copy.deepcopy(tree)
#     cfg_table_inverse_ = copy.deepcopy(cfg_table_inverse)
#     for n in range(len(tree)-1):
#         for k in range(len(tree)-2, -1, -1):
#             node_id_1 = tree[k]
#             node_id_2 = tree[k+1]
#             pipe_1 = cfg_table_inverse_[node_id_1]
#             pipe_2 = cfg_table_inverse_[node_id_2]
#             if pipe_1[0] > pipe_2[0] or (pipe_1[0] == pipe_2[0] and pipe_1[1] > pipe_2[1]):
#                 # print("! Swapping ", tree[k], tree[k+1])
#                 tree[k], tree[k+1] = tree[k+1], tree[k]
#                 for i in range(k+1, len(tree)):
#                     node_id = tree[i]
#                     pipe = cfg_table_inverse_[node_id]
#                     if pipe[0] == k+1:
#                         pipe[0] = k+2
#                     elif pipe[0] == k+2:
#                         pipe[0] = k+1

def sort_tree_nodes(tree: list):
    pipelines = pipeidx_decode(tree).reshape(-1, 4)
    # print("!Pipelines: ", pipelines)
    for n in range(len(pipelines)-1):
        for k in range(len(pipelines)-2, -1, -1):
            pipe_1 = pipelines[k]
            pipe_2 = pipelines[k+1]
            if pipe_1[0] > pipe_2[0] or (pipe_1[0] == pipe_2[0] and pipe_1[1] > pipe_2[1]):
                # print("! Swapping ", pipelines[k], pipelines[k+1])
                # pdb.set_trace()
                # tmp = pipelines[k]
                # pipelines[k] = pipelines[k+1]
                # pipelines[k+1] = tmp
                pipelines[[k, k+1]] = pipelines[[k+1, k]]
                # pipelines[k], pipelines[k+1] = pipelines[k+1], pipelines[k]
                for i in range(k+1, len(pipelines)):
                    # pipe = pipelines[i]
                    if pipelines[i][0] == k+1:
                        pipelines[i][0] = k+2
                    elif pipelines[i][0] == k+2:
                        pipelines[i][0] = k+1
    print("Sorted pipelines: ", pipelines)
    return pipeidx_encode(pipelines)

def compare_pipelines(pipeline1, pipeline2):
    for i in range(len(pipeline1)):
        if pipeline1[i] < pipeline2[i]:
            return False
        if pipeline1[i] > pipeline2[i]:
            return True
        if pipeline1[i] == pipeline2[i]:
            continue
    return True

if __name__ == "__main__":
    pipelines = [[ 0,3,0,0, 0,1,0,0, 2,13,0,0, 1,15,0,0, 0,0,0,0 ]]
    # [   1   28   82 1324  892]

    # pipelines = [[ 0,1,0,0, 0,3,0,0, 0,13,0,0, 0,15,0,0 ],
    #              [ 0,0,0,2 ],
    #              [ 0,1,0,0, 0,3,0,0 ],
    #              [ 0,1,0,0, 0,3,0,0, 0,17,0,2 ],
    #              []]
    

    tree = pipeidx_encode(pipelines[0])
    print("Tree: ", tree)
    tree = sort_tree_nodes(tree)
    print("Sorted Tree: ", tree)
    print("Decoded Tree: \n", pipeidx_decode(tree).reshape(-1, 4))