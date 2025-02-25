import lattice as lat
import variational_space as vs
import hamiltonian as ham
import numpy as np
import scipy.sparse as sps


def create_singlet_triplet_basis_change_matrix_d8(VS, d_state_idx, d_hole_idx):
    """
    生成单态三重态变换矩阵
    :param VS:
    :param d_state_idx:[state_idx1, state_idx2, ...]
    :param d_hole_idx:[hole_idx1, hole_idx2, ...]
    :return:
    """
    dim = VS.dim
    data = []
    row = []
    col = []

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
        state = VS.get_state(VS.lookup_tbl[state_idx])
        # d8态中两个空穴索引
        hole_idx1, hole_idx2 = d_hole_idx[i]
        orb1, s1 = state[hole_idx1][-2:]
        orb2, s2 = state[hole_idx2][-2:]

        # 当d8中两个空穴的自旋相同时
        if s1 == s2:
            data.append(np.sqrt(2))
            row.append(state_idx)
            col.append(state_idx)
            S_d8_val[state_idx] = 1
            if s1 == 'up':
                Sz_d8_val[state_idx] = 1
            else:
                Sz_d8_val[state_idx] = -1

        # 当d8两个空穴的自旋不同时, 分为orbital1 == orbital2和orbital1 != orbital2两种情况
        else:
            if orb1 == orb2:
                data.append(np.sqrt(2))
                row.append(state_idx)
                col.append(state_idx)
                S_d8_val[state_idx] = 0
                Sz_d8_val[state_idx] = 0
            else:
                # 找partner_state之间要将state列表化, 方便修改
                partner_state = [list(hole) for hole in state]
                # 交换自旋
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

                S_d8_val[state_idx] = 0
                Sz_d8_val[state_idx] = 0

                # 将partner_idx设为三重态 = 1/sqrt(2)(|up, dn> + |dn, up>)
                data.append(1.)
                row.append(state_idx)
                col.append(partner_idx)

                data.append(1.)
                row.append(partner_idx)
                col.append(partner_idx)

                S_d8_val[partner_idx] = 1
                Sz_d8_val[partner_idx] = 0

    out = sps.coo_matrix((data, (row, col)), shape=(dim, dim)) / np.sqrt(2)

    return out, S_d8_val, Sz_d8_val
