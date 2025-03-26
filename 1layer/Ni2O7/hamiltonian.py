import time
import numpy as np
import scipy.sparse as sps
from itertools import product
import parameters as pam
import lattice as lat
import variational_space as vs

directions_to_vecs = {'UR': (1, 1, 0), 'UL': (-1, 1, 0), 'DL': (-1, -1, 0), 'DR': (1, -1, 0),
                      'L': (-1, 0, 0), 'R': (1, 0, 0), 'U': (0, 1, 0), 'D': (0, -1, 0),
                      'T': (0, 0, 1), 'B': (0, 0, -1),
                      'L2': (-2, 0, 0), 'R2': (2, 0, 0), 'U2': (0, 2, 0), 'D2': (0, -2, 0), 'T2': (0, 0, 2), 'B2': (0, 0, -2),
                      'pzL': (-1, 0, 1), 'pzR': (1, 0, 1), 'pzU': (0, 1, 1), 'pzD': (0, -1, 1),
                      'mzL': (-1, 0, -1), 'mzR': (1, 0, -1), 'mzU': (0, 1, -1), 'mzD': (0, -1, -1)}


def set_tpd_tpp(tpd, tpp):
    """
    设置通过tpd, tpp跳跃的轨道和对应的方向, 以及对应的值
    :param tpd: p, d轨道的跳跃值
    :param tpp: p, p轨道的跳跃值
    :return: tpd_nn_hop_dir, tpd_nn_hop_fac, tpp_nn_hop_dir, tpp_nn_hop_fac
    """
    if pam.Norb == 5 or pam.Norb == 8:
        tpd_nn_hop_dir = {'d3z2r2': ['L', 'R', 'U', 'D'],
                          'dx2y2': ['L', 'R', 'U', 'D']}
        tpd_nn_hop_fac = {('d3z2r2', 'L', 'px'): -tpd/np.sqrt(3),
                          ('d3z2r2', 'R', 'px'): tpd/np.sqrt(3),
                          ('d3z2r2', 'U', 'py'): tpd/np.sqrt(3),
                          ('d3z2r2', 'D', 'py'): -tpd/np.sqrt(3),
                          ('dx2y2', 'L', 'px'): tpd,
                          ('dx2y2', 'R', 'px'): -tpd,
                          ('dx2y2', 'U', 'py'): tpd,
                          ('dx2y2', 'D', 'py'): -tpd}

        tpp_nn_hop_dir = ['UR', 'UL', 'DL', 'DR']
        # 注意字典顺序
        tpp_nn_hop_fac = {('UR', 'px', 'py'): -tpp,
                          ('UL', 'px', 'py'): tpp,
                          ('DL', 'px', 'py'): -tpp,
                          ('DR', 'px', 'py'): tpp}
    else:
        tpd_nn_hop_dir = None
        tpd_nn_hop_fac = None

        tpp_nn_hop_dir = None
        tpp_nn_hop_fac = None

    return tpd_nn_hop_dir, tpd_nn_hop_fac, tpp_nn_hop_dir, tpp_nn_hop_fac


def set_tdo_tpo(tdo, tpo):
    """
    设置通过tdo, tpo跳跃的轨道和对应的方向, 以及跳跃值
    :param tdo: d, pz轨道的跳跃值
    :param tpo: p, pz轨道的跳跃值
    :return:
    """
    if pam.Norb == 5 or pam.Norb == 8:
        # 设置tdo的跳跃轨道和方向, 跳跃值
        tdo_nn_hop_dir = {'apz': ['T', 'B']}
        tdo_nn_hop_fac = {('apz', 'B', 'd3z2r2'): -tdo,
                          ('apz', 'T', 'd3z2r2'): tdo}

        # 设置tpo的跳跃轨道和方向, 和跳跃值
        tpo_nn_hop_dir = {'apz': ['pzL', 'pzR', 'mzL', 'mzR', 'pzU', 'pzD', 'mzU', 'mzD']}
        tpo_nn_hop_fac = {('apz', 'mzR', 'px'): tpo,
                          ('apz', 'mzL', 'px'): -tpo,
                          ('apz', 'pzR', 'px'): -tpo,
                          ('apz', 'pzL', 'px'): tpo,
                          ('apz', 'mzD', 'py'): -tpo,
                          ('apz', 'mzU', 'py'): tpo,
                          ('apz', 'pzD', 'py'): tpo,
                          ('apz', 'pzU', 'py'): -tpo}
    else:
        tdo_nn_hop_dir = None
        tdo_nn_hop_fac = None

        tpo_nn_hop_dir = None
        tpo_nn_hop_fac = None

    return tdo_nn_hop_dir, tdo_nn_hop_fac, tpo_nn_hop_dir, tpo_nn_hop_fac


def set_tz(if_tz_exist, tz_a1a1, tz_b1b1):
    """

    :param if_tz_exist:
    :param tz_a1a1:
    :param tz_b1b1:
    :return:
    """
    if pam.Norb ==5:
        if if_tz_exist == 0:
            tz_fac = {('px', 'px'): tz_b1b1,
                      ('py', 'py'): tz_b1b1,
                      ('d3z2r2', 'd3z2r2'): tz_a1a1,
                      ('dx2y2', 'dx2y2'): tz_b1b1}
        elif if_tz_exist == 1:
            tz_fac = {('d3z2r2', 'd3z2r2'): tz_a1a1,
                      ('dx2y2', 'dx2y2'): tz_b1b1}
        elif if_tz_exist == 2:
            tz_fac = {('d3z2r2', 'd3z2r2'): tz_a1a1}
        else:
            tz_fac = None
    else:
        tz_fac = None

    return tz_fac


def get_interaction_mat(A, sym):
    """
    根据对称性, 设置d8相互作用的矩阵元
    :param A:相互作用的一个参数
    :param sym: 对称性
    :return: stot, 总自旋
    Sz_set, 总自旋的z分量
    state_order,  {(orb1, orb2): 0, ...), state和interaction_mat的索引关系
    特别注意(orb1, orb2)按照字典序排列
    interaction_mat, [[值1, ...], ...]
    """
    B = pam.B
    C = pam.C
    if sym == '1A1':
        Stot = 0
        Sz_set = [0]
        state_order = {('d3z2r2', 'd3z2r2'): 0,
                       ('dx2y2', 'dx2y2'): 1}
        interaction_mat = [[A+4.*B+3.*C, 4.*B+C],
                           [4.*B+C, A+4.*B+3.*C]]

    elif sym == '1B1':
        Stot = 0
        Sz_set = [0]
        state_order = {('d3z2r2', 'dx2y2'): 0}
        interaction_mat = [[A+2.*C]]

    elif sym == '3B1':
        Stot = 1
        Sz_set = [-1, 0, 1]
        state_order = {('d3z2r2', 'dx2y2'): 0}
        interaction_mat = [[A-8.*B]]

    else:
        Stot = None
        Sz_set = None
        state_order = None
        interaction_mat = None

    return Stot, Sz_set, state_order, interaction_mat


def create_tpd_nn_matrix(VS, tpd_nn_hop_dir, tpd_nn_hop_fac):
    """
    创建Tpd哈密顿矩阵, 只用遍历d到p轨道的跳跃,
    而p轨道到d轨道的跳跃只需将行列交换, 值不变
    :param VS:类, 含有lookup_tbl(存储要计算的态), 函数get_state_uid, get_state, get_index
    :param tpd_nn_hop_dir: 跳跃轨道和方向
    :param tpd_nn_hop_fac: 跳跃的值
    :return: out(coo_matrix), Tpp哈密顿矩阵
    """
    t0 = time.time()
    dim = VS.dim
    # tpd_orbs = [orbital 1, ...], tpd_keys = (orbital 1, direction, orbital 2)...
    tpd_orbs = tpd_nn_hop_dir.keys()
    tpd_keys = tpd_nn_hop_fac.keys()
    data = []
    row = []
    col = []

    # 遍历整个态空间
    for row_idx in range(dim):
        state = VS.get_state(VS.lookup_tbl[row_idx])
        hole_num = len(state)

        # 其中一个空穴跳跃, 其他空穴不动
        for hole_idx in range(hole_num):
            hole = state[hole_idx]
            x, y, z, orb, s = hole
            # 根据轨道决定是否跳跃, 求出跳跃后的坐标
            if orb in tpd_orbs:
                for direction in tpd_nn_hop_dir[orb]:
                    vx, vy, vz = directions_to_vecs[direction]
                    hop_x, hop_y, hop_z = x + vx, y + vy, z + vz
                    # 由跳跃后的坐标得出跳跃后的轨道, 自旋不变
                    hop_orbs = lat.get_unit_cell_rep(hop_x, hop_y, hop_z)
                    if hop_orbs == ['NotOnSublattice']:
                        continue
                    for hop_orb in hop_orbs:
                        orb12 = (orb, direction, hop_orb)
                        if orb12 in tpd_keys:

                            # 跳跃后的空穴
                            hop_hole = (hop_x, hop_y, hop_z, hop_orb, s)
                            if hop_hole not in state:       # 检验是否满足Pauli不相容原理
                                # 将其中的一个空穴换成是跳跃后的空穴
                                hop_state = list(state)
                                hop_state[hole_idx] = hop_hole
                                hop_state, ph = vs.make_state_canonical(hop_state)
                                col_idx = VS.get_index(hop_state)
                                if col_idx is not None:
                                    value = tpd_nn_hop_fac[orb12] * ph
                                    data.extend((value, value))
                                    row.extend((row_idx, col_idx))
                                    col.extend((col_idx, row_idx))

    out = sps.coo_matrix((data, (row, col)), shape=(dim, dim))
    t1 = time.time()
    print('Tpd cost time', t1-t0)

    return out


def create_tpp_nn_matrix(VS, tpp_nn_hop_dir, tpp_nn_hop_fac):
    """
    设置Tpp哈密顿矩阵
    :param VS: 态空间
    :param tpp_nn_hop_dir: p轨道之间的跳跃方向
    :param tpp_nn_hop_fac: p轨道之间的跳跃值
    :return:out(coo_matrix), Tpp哈密顿矩阵
    """
    t0 = time.time()
    dim = VS.dim
    # tpp_keys = (direction, orb1, orb2)...
    tpp_keys = tpp_nn_hop_fac.keys()
    data = []
    row = []
    col = []

    # 遍历整个态空间
    for row_idx in range(dim):
        state = VS.get_state(VS.lookup_tbl[row_idx])
        hole_num = len(state)

        # 其中一个空穴跳跃, 其他空穴不动
        for hole_idx in range(hole_num):
            hole = state[hole_idx]
            x, y, z, orb, s = hole
            # 根据轨道决定是否跳跃, 求出跳跃后的坐标
            if orb in pam.O_orbs:
                for direction in tpp_nn_hop_dir:
                    vx, vy, vz = directions_to_vecs[direction]
                    hop_x, hop_y, hop_z = x+vx, y+vy, z+vz
                    # 由跳跃后的坐标得出跳跃后的轨道, 自旋不变
                    hop_orbs = lat.get_unit_cell_rep(hop_x, hop_y, hop_z)
                    if hop_orbs == ['NotOnSublattice']:
                        continue
                    if hop_orbs != pam.O1_orbs and hop_orbs != pam.O2_orbs:
                        continue
                    for hop_orb in hop_orbs:
                        # 注意字典顺序
                        orb12 = sorted([orb, direction, hop_orb])
                        orb12 = tuple(orb12)
                        if orb12 in tpp_keys:

                            # 跳跃后的空穴
                            hop_hole = (hop_x, hop_y, hop_z, hop_orb, s)
                            if hop_hole not in state:       # 检验是否满足Pauli不相容原理
                                # 将其中的一个空穴换成是跳跃后的空穴
                                hop_state = list(state)
                                hop_state[hole_idx] = hop_hole
                                hop_state, ph = vs.make_state_canonical(hop_state)
                                col_idx = VS.get_index(hop_state)
                                if col_idx is not None:
                                    value = tpp_nn_hop_fac[orb12] * ph
                                    data.append(value)
                                    row.append(row_idx)
                                    col.append(col_idx)

    out = sps.coo_matrix((data, (row, col)), shape=(dim, dim))
    t1 = time.time()
    print('Tpp cost time', t1 - t0)

    return out


def create_tdo_nn_matrix(VS, tdo_nn_hop_dir, tdo_nn_hop_fac):
    """
    设置Tdo哈密顿矩阵
    :param VS: 态空间
    :param tdo_nn_hop_dir: apz的跳跃方向
    :param tdo_nn_hop_fac: apz往不同方向对应的跳跃值
    :return: out(coo_matrix), Tdo哈密顿矩阵
    """
    t0 = time.time()
    dim = VS.dim
    tdo_orbs = tdo_nn_hop_dir.keys()
    tdo_keys = tdo_nn_hop_fac.keys()
    data = []
    row = []
    col = []

    # 遍历整个态空间
    for row_idx in range(dim):
        state = VS.get_state(VS.lookup_tbl[row_idx])
        hole_num = len(state)

        # 其中一个空穴跳跃, 其他空穴不动
        for hole_idx in range(hole_num):
            hole = state[hole_idx]
            x, y, z, orb, s = hole
            # 根据轨道决定是否跳跃, 求出跳跃后的坐标
            if orb in tdo_orbs:
                for direction in tdo_nn_hop_dir[orb]:
                    vx, vy, vz = directions_to_vecs[direction]
                    hop_x, hop_y, hop_z = x + vx, y + vy, z + vz
                    # 由跳跃后的坐标得出跳跃后的轨道, 自旋不变
                    hop_orbs = lat.get_unit_cell_rep(hop_x, hop_y, hop_z)
                    if hop_orbs == ['NotOnSublattice']:
                        continue
                    for hop_orb in hop_orbs:
                        orb12 = (orb, direction, hop_orb)
                        if orb12 in tdo_keys:

                            # 跳跃后的空穴
                            hop_hole = (hop_x, hop_y, hop_z, hop_orb, s)
                            if hop_hole not in state:  # 检验是否满足Pauli不相容原理
                                # 将其中的一个空穴换成是跳跃后的空穴
                                hop_state = list(state)
                                hop_state[hole_idx] = hop_hole
                                hop_state, ph = vs.make_state_canonical(hop_state)
                                col_idx = VS.get_index(hop_state)
                                if col_idx is not None:
                                    value = tdo_nn_hop_fac[orb12] * ph
                                    data.extend((value, value))
                                    row.extend((row_idx, col_idx))
                                    col.extend((col_idx, row_idx))

    out = sps.coo_matrix((data, (row, col)), shape=(dim, dim))
    t1 = time.time()
    print('Tdo cost time', t1 - t0)

    return out


def create_tpo_nn_matrix(VS, tpo_nn_hop_dir, tpo_nn_hop_fac):
    """
    设置Tpo哈密顿矩阵
    :param VS: 态空间
    :param tpo_nn_hop_dir: apz的跳跃方向
    :param tpo_nn_hop_fac: apz往不同方向对应的跳跃值
    :return: out(coo_matrix), Tpo哈密顿矩阵
    """
    t0 = time.time()
    dim = VS.dim
    tpo_orbs = tpo_nn_hop_dir.keys()
    tpo_keys = tpo_nn_hop_fac.keys()
    data = []
    row = []
    col = []

    # 遍历整个态空间
    for row_idx in range(dim):
        state = VS.get_state(VS.lookup_tbl[row_idx])
        hole_num = len(state)

        # 其中一个空穴跳跃, 其他空穴不动
        for hole_idx in range(hole_num):
            hole = state[hole_idx]
            x, y, z, orb, s = hole
            # 根据轨道决定是否跳跃, 求出跳跃后的坐标
            if orb in tpo_orbs:
                for direction in tpo_nn_hop_dir[orb]:
                    vx, vy, vz = directions_to_vecs[direction]
                    hop_x, hop_y, hop_z = x + vx, y + vy, z + vz
                    # 由跳跃后的坐标得出跳跃后的轨道, 自旋不变
                    hop_orbs = lat.get_unit_cell_rep(hop_x, hop_y, hop_z)
                    if hop_orbs == ['NotOnSublattice']:
                        continue
                    for hop_orb in hop_orbs:
                        orb12 = (orb, direction, hop_orb)
                        if orb12 in tpo_keys:

                            # 跳跃后的空穴
                            hop_hole = (hop_x, hop_y, hop_z, hop_orb, s)
                            if hop_hole not in state:  # 检验是否满足Pauli不相容原理
                                # 将其中的一个空穴换成是跳跃后的空穴
                                hop_state = list(state)
                                hop_state[hole_idx] = hop_hole
                                hop_state, ph = vs.make_state_canonical(hop_state)
                                col_idx = VS.get_index(hop_state)
                                if col_idx is not None:
                                    value = tpo_nn_hop_fac[orb12] * ph
                                    data.extend((value, value))
                                    row.extend((row_idx, col_idx))
                                    col.extend((col_idx, row_idx))

    out = sps.coo_matrix((data, (row, col)), shape=(dim, dim))
    t1 = time.time()
    print('Tpo cost time', t1 - t0)

    return out


def create_Esite_matrix(VS, A, ed, ep, eo):
    """
    创建Onsite_energy哈密顿矩阵, 并计算dn, 能量设为A + abs(n - 8) * A / 2
    :param VS:
    :param A:
    :param ed
    :param ep:
    :param eo:
    :return:
    """
    t0 = time.time()
    dim = VS.dim
    data = []
    row = []
    col = []
    for row_idx in range(dim):
        state = VS.get_state(VS.lookup_tbl[row_idx])
        diag_el = 0.
        Ni_num = {position: 0 for position in lat.Ni_position}
        for x, y, z, orb, _ in state:
            # 计算d, p, apz轨道上的在位能
            if orb in pam.Ni_orbs:
                diag_el += ed[orb]
            elif orb in pam.O_orbs:
                diag_el += ep
            elif orb in pam.Oap_orbs:
                diag_el += eo

            # 统计在相同Ni上的个数
            if orb in pam.Ni_orbs:
                Ni_num[(x, y, z)] += 1

        # dn的能量, 比d8高abs(n - 2) * A / 2., d8估计为A
        for num in Ni_num.values():
            if num != 2:
                diag_el += A + abs(num - 2) * A / 2.

        data.append(diag_el)
        row.append(row_idx)
        col.append(row_idx)

    out = sps.coo_matrix((data, (row, col)), shape=(dim, dim))
    t1 = time.time()
    print('Esite cost time', t1 - t0)

    return out


def create_tz_matrix(VS, tz_fac):
    """
    设置d轨道之间的杂化
    :param VS:
    :param tz_fac:
    :return: out(coo_matrix)
    """
    t0 = time.time()
    dim = VS.dim
    data = []
    row = []
    col = []
    # tz_keys = (orbital1, orbital2)
    tz_keys = tz_fac.keys()
    # 只选择被夹的那一层(含有Ni)
    z_list = range(2, 2*pam.layer_num-1, 2)
    z_list = list(z_list)

    # 遍历整个态空间
    for row_idx in range(dim):
        state = VS.get_state(VS.lookup_tbl[row_idx])
        hole_num = len(state)

        # 其中一个空穴跳跃, 其他空穴不动
        for hole_idx in range(hole_num):
            hole = state[hole_idx]
            x, y, z, orb, s = hole
            if z in z_list and (orb, orb) in tz_keys:     # 夹层并且满足选定的轨道
                hop_z = z - 2   # 往下一层
                hop_hole = (x, y, hop_z, orb, s)
                if hop_hole not in state:       # 是否符合Pauli不相容原理
                    # 将其中的一个空穴换成是跳跃后的空穴
                    hop_state = list(state)
                    hop_state[hole_idx] = hop_hole
                    hop_state, ph = vs.make_state_canonical(hop_state)
                    col_idx = VS.get_index(hop_state)
                    if col_idx is not None:
                        value = tz_fac[(orb, orb)] * ph
                        data.extend((value, value))
                        row.extend((row_idx, col_idx))
                        col.extend((col_idx, row_idx))

    out = sps.coo_matrix((data, (row, col)), shape=(dim, dim))
    t1 = time.time()
    print('Tz cost time', t1 - t0)

    return out


def get_double_occ_list(VS):
    """
    找出态中有两个空穴是在同一位置
    :param VS: 态空间
    :return: multi_d_state_idx = {Ni0位置: [state_idx1, state_idx2, ....]...}
    multi_d_hole_idx, {Ni0位置: [state_idx1, state_idx2, ...]...}
    p_idx_pair, [(p_idx, p_pair)...]
    apz_idx_pair, [(apz_idx, apz_pair)...]
    """
    t0 = time.time()
    multi_d_state_idx = {}
    multi_d_hole_idx = {}
    p_idx_pair = []
    apz_idx_pair = []

    dim = VS.dim
    # 遍历整个态空间
    for i in range(dim):
        state = VS.get_state(VS.lookup_tbl[i])
        # 统计在相同Ni上空穴的索引和在相同O上空穴数目
        Ni_idx = {}
        O_num = {}      # 相同层间O上空穴数目
        Oap_num = {}        #相同层内O上空穴数目
        hole_num = len(state)
        for hole_idx in range(hole_num):
            x, y, z, orb, s = state[hole_idx]
            if orb in pam.Ni_orbs:
                if (x, y, z) in Ni_idx:
                    Ni_idx[(x, y, z)] += [hole_idx]
                else:
                    Ni_idx[(x, y, z)] = [hole_idx]
            elif orb in pam.O_orbs:
                if (x, y, z) in O_num:
                    O_num[(x, y, z)] += 1
                else:
                    O_num[(x, y, z)] = 1
            else:
                if (x, y, z) in Oap_num:
                    Oap_num[(x, y, z)] += 1
                else:
                    Oap_num[(x, y, z)] = 1

        # 记录在Ni上的空穴数目是2的态索引和空穴索引
        for position, idx_list in Ni_idx.items():
            Ni_num = len(idx_list)
            if Ni_num == 2:
                if position in multi_d_state_idx:
                    multi_d_state_idx[position] += [i]
                    multi_d_hole_idx[position] += [idx_list]
                else:
                    multi_d_state_idx[position] = [i]
                    multi_d_hole_idx[position] = [idx_list]

        # 记录在层内O上空穴数目大于1的态索引和空穴对
        p_pair = 0
        for num in O_num.values():
            if num > 1:
                p_pair += num * (num - 1) / 2
        if p_pair > 0:
            p_idx_pair.append((i, p_pair))

        # 记录在层间O上空穴数目大于1的
        apz_pair = 0
        for num in Oap_num.values():
            if num > 1:
                apz_pair += num * (num - 1) / 2
        if apz_pair > 0:
            apz_idx_pair.append((i, apz_pair))

    t1 = time.time()
    print(f'double_occ time {t1-t0}')

    return multi_d_state_idx, multi_d_hole_idx, p_idx_pair, apz_idx_pair


def create_interaction_matrix_d8(VS, d_state_idx, d_hole_idx, S_val, Sz_val, A):
    """
    设置d8相互作用矩阵
    :param VS:
    :param d_state_idx: (state_idx1, state_idx2, ...)
    :param d_hole_idx: ([hole_idx1, hole_idx2, ...], ....)
    :param S_val: {state_idx1: S1, ...}
    :param Sz_val: {state_idx1: Sz1, ...}
    :param A:
    :return: out
    """
    t0 = time.time()
    data = []
    row = []
    col = []
    dim = VS.dim
    # Ni轨道的所有可能组合
    exist_orb34 = product(pam.Ni_orbs, repeat=2)
    exist_orb34 = tuple(exist_orb34)

    # 遍历所求对称性
    channels = ('1A1', '1B1', '3B1')
    for sym in channels:
        Stot, Sz_set, state_order, interaction_mat = get_interaction_mat(A, sym)
        for i, state_idx in enumerate(d_state_idx):
            count = []      # 避免重复计算
            state = VS.get_state(VS.lookup_tbl[state_idx])
            # 提取d8的两个空穴轨道, S12和Sz12
            hole_idx1, hole_idx2 = d_hole_idx[i]
            orb1 = state[hole_idx1][-2]
            orb2 = state[hole_idx2][-2]
            orb1, orb2 = sorted([orb1, orb2])
            orb12 = (orb1, orb2)
            S12 = S_val[state_idx]
            Sz12 = Sz_val[state_idx]

            # 判断轨道, S12, Sz12是否满足要求
            if orb12 not in state_order.keys() or S12 != Stot or Sz12 not in Sz_set:
                continue

            # 得出interaction_mat的行索引
            mat_idx1 = state_order[orb12]
            # 遍历interaction_mat的列, 同时对应state_order.keys()
            for mat_idx2, orb34 in enumerate(state_order.keys()):
                if orb34 not in exist_orb34:
                    continue

                # 生成相互作用的另一个态, 并找出对应的索引
                # 先生成新的d8对应的两个空穴
                for s1 in ['dn', 'up']:
                    for s2 in ['dn', 'up']:
                        if (orb34[0], s1) == (orb34[1], s2):      # 检查是否满足Pauli不相容原理
                            continue
                        hole1 = state[hole_idx1][:3] + (orb34[0], s1)
                        hole2 = state[hole_idx2][:3] + (orb34[1], s2)

                        # 将state列表化, 并将其中的两个空穴替换成新的d8
                        inter_state = list(state)
                        inter_state[hole_idx1], inter_state[hole_idx2] = hole1, hole2
                        inter_state, _ = vs.make_state_canonical(inter_state)

                        # 找到相互作用态的索引
                        inter_idx = VS.get_index(inter_state)
                        if inter_idx is None or inter_idx in count:
                            continue
                        # 判断新的d8对应的S34, Sz34是否满足要求
                        S34, Sz34 = S_val[inter_idx], Sz_val[inter_idx]
                        if S34 != S12 or Sz34 != Sz12:
                            continue

                        # 利用mat_idx1和mat_idx找出矩阵值
                        value = interaction_mat[mat_idx1][mat_idx2]
                        data.append(value)
                        row.append(state_idx)
                        col.append(inter_idx)
                        count.append(inter_idx)

    out = sps.coo_matrix((data, (row, col)), shape=(dim, dim))
    t1 = time.time()
    print(f'interaction_d8 time {t1 - t0}')

    return out


def create_interaction_matrix_po(VS, p_idx_pair, apz_idx_pair, Upp, Uoo):
    """
    设置p, pz轨道的相互作用
    :param VS:
    :param p_idx_pair: [(p_idx1, p_pair1), ...]
    :param apz_idx_pair: [(apz_idx1, apz_idx2), ...]
    :param Upp: p,p轨道的相互作用
    :param Uoo: pz, pz轨道的相互作用
    :return:
    """
    t0 = time.time()
    dim = VS.dim
    data = []
    row = []
    col = []

    # p, p轨道相互作用矩阵
    if Upp != 0:
        for state_idx, p_pair in p_idx_pair:
            data.append(Upp*p_pair)
            row.append(state_idx)
            col.append(state_idx)

    # pz, pz轨道相互作用矩阵
    if Uoo != 0:
        for state_idx, apz_pair in apz_idx_pair:
            data.append(Uoo*apz_pair)
            row.append(state_idx)
            col.append(state_idx)

    out = sps.coo_matrix((data, (row, col)), shape=(dim, dim))
    t1 = time.time()
    print(f'interaction_po time {t1-t0}')

    return out
