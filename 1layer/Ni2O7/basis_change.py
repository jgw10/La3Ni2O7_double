import time
import numpy as np
import scipy.sparse as sps
from itertools import product
from sympy import Rational
from sympy.physics.quantum.cg import CG

import variational_space as vs
import lattice as lat


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


def create_singlet_triplet_basis_change_matrix(VS, d_state_idx, d_hole_idx, position):
    """
    单边的单态三重态变换矩阵
    :param VS:
    :param d_state_idx:[state_idx1, state_idx2, ...]
    :param d_hole_idx:[hole_idx1, hole_idx2, ...]
    :param position: Ni的位置
    :return:
    """
    t0 = time.time()
    dim = VS.dim
    row = []
    col = []
    data = []

    # 标记该态是单态或者三重态
    S_Ni_val = {}
    Sz_Ni_val = {}
    # 存储partner state的索引, 避免重复
    count_list = []

    # 遍历态空间
    for state_idx in range(dim):
        if state_idx in count_list:
            continue

        # d8
        if state_idx in d_state_idx:
            # 找到d8对应的空穴索引
            idx = d_state_idx.index(state_idx)
            hole_idx1, hole_idx2 = d_hole_idx[idx]
            set_singlet_triplet_matrix_element(VS, state_idx, hole_idx1, hole_idx2,
                                               row, col, data, S_Ni_val, Sz_Ni_val, count_list)
        else:
            state = VS.get_state(VS.lookup_tbl[state_idx])
            side1, side2 = lat.get_Ni_side_num(state)
            # 一边是两个空穴的情况. 为防止重复, 先只做一边的变换
            if (len(side1) == 2 and position[0] < 0) or (len(side2) == 2 and position[0] > 0):
                # 要交换的两个空穴索引
                if position[0] < 0:
                    hole_idx1, hole_idx2 = side1
                else:
                    hole_idx1, hole_idx2 = side2
                set_singlet_triplet_matrix_element(VS, state_idx, hole_idx1, hole_idx2,
                                                   row, col, data, S_Ni_val, Sz_Ni_val, count_list)
            else:
                data.append(np.sqrt(2))
                row.append(state_idx)
                col.append(state_idx)

    t1 = time.time()
    print(f'singlet_triplet_double basis change time {t1 - t0}')

    return sps.coo_matrix((data, (row, col)), shape=(dim, dim)) / np.sqrt(2), S_Ni_val, Sz_Ni_val


def coupling_representation(j1_list, j2_list, j1m1_list, j2m2_list, expand1_list, expand2_list):
    """
    This function is used to expand two states in the coupled representation into the uncoupled representation.
    :param j1_list:particle1's spin quantum numbers
    :param j2_list:
    :param j1m1_list:particle1's spin quantum numbers and its corresponding magnetic quantum number
    :param j2m2_list:
    :param expand1_list:
    :param expand2_list:
    :return:
    """
    # j1m1_list和j2m2_list中的(j1, m1)转化为sympy.Rational类型, 方便符号运算和查找
    j1m1_list = [(Rational(j1), Rational(m1)) for j1, m1 in j1m1_list]
    j2m2_list = [(Rational(j2), Rational(m2)) for j2, m2 in j2m2_list]

    # 1. 计算耦合自旋量子数j
    cou_j_list = []
    jm_list = []
    expand_list = []
    start_idx1 = 0  # 用于j1m1_list的起始索引
    for j1 in j1_list:
        start_idx2 = 0  # 用于j2m2_list的起始索引
        for j2 in j2_list:
            j_list = np.arange(abs(j1 - j2), j1 + j2 + 1)
            for j in j_list:
                cou_j_list.append(j)

                # 2. 计算耦合磁量子数m
                for m in np.arange(-j, j + 1):
                    expand = {}

                    # 3. 计算耦合磁量子数m1和m2
                    for m1 in np.arange(-j1, j1 + 1):
                        m2 = m - m1
                        if m2 < -j2 or m2 > j2:
                            continue

                        # 4. 计算CG系数并将|j1, m1> = expand1和|j2, m2> = expand2代入cg|j1, m1>|j2, m2> + ...中
                        # 计算结果存储在expand中
                        j1, m1, j2, m2 = Rational(j1), Rational(m1), Rational(j2), Rational(m2)
                        j, m = Rational(j), Rational(m)
                        cg = CG(j1, m1, j2, m2, j, m).doit()
                        # (j1, m1)有重复出现，需要加入索引范围start_idx1到start_idx1+2*j1+1，在2*j1+1(m1的个数)个里面找
                        idx1 = j1m1_list.index((j1, m1), start_idx1, start_idx1 + 2 * j1 + 1)
                        idx2 = j2m2_list.index((j2, m2), start_idx2, start_idx2 + 2 * j2 + 1)
                        for factor1, coef1 in expand1_list[idx1].items():
                            for factor2, coef2 in expand2_list[idx2].items():
                                factor = factor1 + factor2
                                if factor in expand:
                                    expand[factor] += cg * coef1 * coef2  # 如果项因子factor已经出现过，则累加
                                else:
                                    expand[factor] = cg * coef1 * coef2

                    # 将|j, m>存储在jm_list，将展开式右边存储在expand_list
                    jm_list.append((j, m))
                    expand_list.append(expand)

            start_idx2 += 2 * j2 + 1  # 更新j2m2_list的起始索引
        start_idx1 += 2 * j1 + 1  # 更新j1m1_list的起始索引

    return cou_j_list, jm_list, expand_list


def create_coupled_representation_matrix(VS):
    """
    Construct the transformation matrix to the coupled representation.
    :param VS:all state
    :return:sps.coc_matrix, 变换矩阵; S_val, 总自旋; Sz_val, 总自旋的z分量
    """
    t0 = time.time()
    # 单个空穴
    half = Rational(1, 2)
    j1_list = [half]
    j1m1_list = [(half, -half), (half, half)]
    expand1_list = [{(-half,): 1}, {(half,): 1}]

    row = []
    col = []
    data = []
    S_val = {}
    Sz_val = {}
    count_list = [] # 找到对应的2 ** n态, 记录下来, 当遍历到这2 ** n态后, 直接跳过

    # 遍历态空间
    dim = VS.dim
    for istate in range(dim):
        if istate in count_list:
            continue
        state = VS.get_state(VS.lookup_tbl[istate])

        # 1.遍历态, 找到(x, y, z, orb)不同的部分, 记录空穴索引
        idx_dict = {}
        for hole_idx, hole in enumerate(state):
            x, y, z, orb, _ = hole
            if (x, y, z, orb) in idx_dict:
                del idx_dict[(x, y, z, orb)]
            else:
                idx_dict[(x, y, z, orb)] = hole_idx
        idx_list = [value for value in idx_dict.values()]
        idx_list.sort()

        # 2.确定要耦合的空穴数目, 生成耦合表象在非耦合表象下的展开式
        coupled_num = len(idx_list)
        # 若要耦合的空穴数目小于2, 则不需要耦合
        if coupled_num < 2:
            data.append(1.)
            row.append(istate)
            col.append(istate)
            continue

        state = [list(hole) for hole in state]  # 将元组改为列表, 方便修改

        j_list = j1_list
        jm_list = j1m1_list
        expand_list = expand1_list
        for _ in range(coupled_num-1):
            j_list, jm_list, expand_list = coupling_representation(j_list, j1_list, jm_list, j1m1_list,
                                                                       expand_list, expand1_list)
        # 调整展开式的顺序, 按照m, j升序排列
        sorted_jm_idx = sorted(range(len(jm_list)), key=lambda i: (jm_list[i][1], jm_list[i][0]))
        jm_list = [jm_list[i] for i in sorted_jm_idx]
        expand_list = [expand_list[i] for i in sorted_jm_idx]

        # 3.根据耦合的空穴数目n, 找到固定(x, y, z, orb)所对应的2 ** n种态以及相位
        partner_dict = {}
        for sz_tuple in product(['up', 'dn'], repeat=coupled_num):
            partner_state = state
            sz_tot = 0
            sz_val = []
            for s_idx, sz in enumerate(sz_tuple):
                if sz == 'up':
                    sz_tot += 1/2
                    sz_val.append(half)
                else:
                    sz_tot -= 1/2
                    sz_val.append(-half)
                hole_idx = idx_list[s_idx]
                partner_state[hole_idx][-1] = sz
            partner_state = [tuple(hole) for hole in partner_state]
            partner_state, ph = vs.make_state_canonical(partner_state)
            i_partner = VS.get_index(partner_state)
            count_list.append(i_partner)
            sz_val = tuple(sz_val)
            partner_dict[sz_val] = (sz_tot, i_partner, ph)

        partner_list = [partner for partner in partner_dict.values()]
        partner_list.sort()

        # 4.设置耦合变换的矩阵元
        for jm_idx, expand in enumerate(expand_list):
            j, m = jm_list[jm_idx]
            col_idx = partner_list[jm_idx][1]
            S_val[col_idx] = j
            Sz_val[col_idx] = m

            for factor, coef in expand.items():
                coef = float(coef)
                row_idx, ph = partner_dict[factor][1:]

                row.append(row_idx)
                col.append(col_idx)
                data.append(ph*coef)

    t1 = time.time()
    print(f'coupled representation time {t1-t0}')

    return sps.coo_matrix((data, (row, col)), shape=(dim, dim)), S_val, Sz_val
