import numpy as np
import scipy.sparse as sps
import time

import variational_space as vs

def set_singlet_triplet_matrix_element(VS, state_idx, hole_idx1, hole_idx2,
                                       row, col, data, S_double_val, Sz_double_val, count_list):
    """
    设置单态三重态的变换矩阵元
    """
    state = VS.get_state(VS.lookup_tbl[state_idx])
    orb1, s1 = state[hole_idx1][-2:]
    orb2, s2 = state[hole_idx2][-2:]

    # 当两个空穴自旋相同时
    if s1 == s2:
        row.append(state_idx)
        col.append(state_idx)
        data.append(np.sqrt(2))
        S_double_val[state_idx] = 1
        if s1 == 'up':
            Sz_double_val[state_idx] = 1
        else:
            Sz_double_val[state_idx] = -1

    # 当两个空穴自旋不同是, 分为轨道相同和不同两个部分
    else:
        if orb1 == orb2:
            data.append(np.sqrt(2))
            row.append(state_idx)
            col.append(state_idx)
            S_double_val[state_idx] = 0
            Sz_double_val[state_idx] = 0
        else:
            # 交换两个自旋
            partner_state = [list(hole) for hole in state]
            partner_state[hole_idx1][-1], partner_state[hole_idx2][-1] = \
                partner_state[hole_idx2][-1], partner_state[hole_idx1][-1]
            partner_state = [tuple(hole) for hole in partner_state]
            partner_state, _ = vs.make_state_canonical(partner_state)
            # 找到对应的索引
            partner_idx = VS.get_index(partner_state)
            count_list.append(partner_idx)

            # 将state_idx设为单态 = 1/sqrt(2)(|up, dn> - |dn, up>)
            # 注意在这里也有可能是1/sqrt(2)(|dn, up> - |up, dn>)
            data.append(1.)
            row.append(state_idx)
            col.append(state_idx)

            data.append(-1.)
            row.append(partner_idx)
            col.append(state_idx)

            S_double_val[state_idx] = 0
            Sz_double_val[state_idx] = 0

            # 将partner_idx设为三重态 = 1/sqrt(2)(|up, dn> + |dn, up>)
            data.append(1.)
            row.append(state_idx)
            col.append(partner_idx)

            data.append(1.)
            row.append(partner_idx)
            col.append(partner_idx)

            S_double_val[partner_idx] = 1
            Sz_double_val[partner_idx] = 0


def create_singlet_triplet_basis_change_matrix_d8(VS, d_state_idx, d_hole_idx):
    """
    对d8的单态三重态变换矩阵
    :param VS:
    :param d_state_idx:[state_idx1, state_idx2, ...]
    :param d_hole_idx:[hole_idx1, hole_idx2, ...]
    :return:
    """
    t0 = time.time()
    dim = VS.dim
    row = []
    col = []
    data = []

    # 存储partner state的索引, 避免重复
    count_list = []
    # 标记该态是单态或者三重态
    S_d8_val = {}
    Sz_d8_val = {}

    # 遍历所有的态空间
    for i in range(dim):
        # 不是d8的态, 变换矩阵的对角元设置为sqrt(2)(最后会除以sqrt(2))
        if i not in d_state_idx:
            data.append(np.sqrt(2))
            row.append(i)
            col.append(i)

    # 遍历所有d8态
    for i, state_idx in enumerate(d_state_idx):
        if state_idx in count_list:
            continue
        # d8态中两个空穴索引
        hole_idx1, hole_idx2 = d_hole_idx[i]
        set_singlet_triplet_matrix_element(VS, state_idx, hole_idx1, hole_idx2,
                                           row, col, data, S_d8_val, Sz_d8_val, count_list)

    t1 = time.time()
    print(f'singlet_triplet_d8 basis change time {(t1-t0)//60//60}h {(t1-t0)//60%60}min, {(t1-t0)%60}s')

    return sps.coo_matrix((data, (row, col)), shape=(dim, dim)) / np.sqrt(2), S_d8_val, Sz_d8_val
