import parameters as pam

# 将轨道, 自旋与数字一一对应, 用来生成每个态的数字uid
if pam.Norb == 5:
    orb_int = {'d3z2r2': 0,
               'dx2y2': 1,
               'px': 2,
               'py': 3,
               'apz': 4}
    int_orb = {value: key for key, value in orb_int.items()}
spin_int = {'up': 1, 'dn': 0}
int_spin = {value: key for key, value in spin_int.items()}

# Ni, 层内O和层间O的位置
Ni_xy = ((-1, 0), (1, 0))
O1_xy = ((-2, 0), (0, 0), (2, 0))
O2_xy = ((-1, 1), (-1, -1), (1, 1), (1, -1))
Oap_xy = ((-1, 0), (1, 0))

layer_num = pam.layer_num
Ni_position = [(x, y, z) for x, y in Ni_xy for z in range(0, 2*layer_num-1, 2)]
O1_position = [(x, y, z) for x, y in O1_xy for z in range(0, 2*layer_num-1, 2)]
O2_position = [(x, y, z) for x, y in O2_xy for z in range(0, 2*layer_num-1, 2)]
O_position = O1_position + O2_position
Oap_position = [(x, y, z) for x, y in Oap_xy for z in range(1, 2*layer_num-2, 2)]

b_x = 5
delta_x = 2
b_y = 3
delta_y = 1
b_z = 2*layer_num-1

def get_unit_cell_rep(x, y, z):
    """
    确定需要计算的晶格, 根据坐标确定轨道
    :return:orbs
    """
    # 确定轨道
    if (x, y, z) in Ni_position:
        return pam.Ni_orbs
    elif (x, y, z) in O1_position:
        return pam.O1_orbs
    elif (x, y, z) in O2_position:
        return pam.O2_orbs
    elif (x, y, z) in Oap_position:
        return pam.Oap_orbs
    else:
        return ['NotOnSublattice']


def get_state_type(state):
    """
    按照Ni, 层内O, 层间O每层的数量给态分类
    :param state: state = ((x1, y1, z1, orb1, s1), ...)
    :return:state_type
    """
    # 统计每一层Ni, 层内O, 层间O的数量
    Ni_num = {}
    O_num = {}
    Oap_num = {}
    for hole in state:
        x, _, z, orb, _ = hole
        # Ni
        if orb in pam.Ni_orbs:      # 这里尽量用轨道来判断, 方便以后修改lattice.py
            if (x, z) in Ni_num:
                Ni_num[(x, z)] += 1
            else:
                Ni_num[(x, z)] = 1
        # 层内O
        if orb in pam.O_orbs:
            if x < 0:
                x = -1
            elif x > 0:
                x = 1
            if (x, z) in O_num:
                O_num[(x, z)] += 1
            else:
                O_num[(x, z)] = 1
        # 层间O
        if orb in pam.Oap_orbs:
            if (x, z) in Oap_num:
                Oap_num[(x, z)] += 1
            else:
                Oap_num[(x, z)] = 1

    # 根据每一层Ni, 层内O和层间O的空穴数量, 生成态的类型
    state_type = {}
    # Ni
    for xz, num in Ni_num.items():
        state_type[xz] = f'd{10-num}'
    # 层内O
    for xz, num in O_num.items():
        if num == 1:
            if xz in state_type:
                state_type[xz] += 'L'
            else:
                state_type[xz] = 'L'
        else:
            if xz in state_type:
                state_type[xz] += f'L{num}'
            else:
                state_type[xz] = f'L{num}'
    # 层间O
    for xz, num in Oap_num.items():
        if num == 1:
            state_type[xz] = 'O'
        else:
            state_type[xz] = f'O{num}'

    sorted_xz = sorted(state_type.keys())
    layer_string = {}
    for xz in sorted_xz:
        layer_part = state_type[xz]
        x, z = xz
        if z in layer_string:
            layer_string[z] += [layer_part]
        else:
            layer_string[z] = [layer_part]
    state_type = []
    sorted_z = sorted(layer_string.keys())
    for z in sorted_z:
        state_type.append('_'.join(layer_string[z]))
    state_type = '-'.join(state_type)

    return state_type


def get_orb_type(state):
    """
    将具体的state_type细化, 输出具体的d轨道
    :param state:
    :return:
    """
    simple_orbs = {'d3z2r2': 'dz2', 'dx2y2': 'dx2'} # 简化坐标表示
    Ni_orb = {}
    Oap_orb = {}    # 收集层间O的轨道数目
    L_orb = {}  # 收集层内O的轨道数目
    for hole in state:
        x, _, z, orb, _ = hole
        if orb in pam.Ni_orbs:
            if orb in simple_orbs:
                orb = simple_orbs[orb]
            if (x, z) in Ni_orb:
                Ni_orb[(x, z)] += [orb]
            else:
                Ni_orb[(x, z)] = [orb]
        elif orb in pam.Oap_orbs:
            if (x, z) in Oap_orb:
                Oap_orb[(x, z)] += 1
            else:
                Oap_orb[(x, z)] = 1
        else:
            if x < 0:
                x = -1
            elif x > 0:
                x = 1
            if (x, z) in L_orb:
                L_orb[(x, z)] += 1
            else:
                L_orb[(x, z)] = 1

    orb_type = {}
    for xz, orbs in Ni_orb.items():
        orbs.sort()
        orb_type[xz] = ''.join(orbs)
    for xz, num in L_orb.items():
        if xz in orb_type:
            if num == 1:
                orb_type[xz] += f'L'
            else:
                orb_type[xz] += f'L{num}'
        else:
            if num == 1:
                orb_type[xz] = f'L'
            else:
                orb_type[xz] = f'L{num}'
    for xz, num in Oap_orb.items():
        if xz in orb_type:
            if num == 1:
                orb_type[xz] += f'apz'
            else:
                orb_type[xz] += f'apz{num}'
        else:
            if num == 1:
                orb_type[xz] = f'apz'
            else:
                orb_type[xz] = f'apz{num}'
    sorted_xz = sorted(orb_type.keys())
    layer_string = {}
    for xz in sorted_xz:
        layer_part = orb_type[xz]
        x, z = xz
        if z in layer_string:
            layer_string[z] += [layer_part]
        else:
            layer_string[z] = [layer_part]
    orb_type = []
    sorted_z = sorted(layer_string.keys())
    for z in sorted_z:
        orb_type.append('_'.join(layer_string[z]))
    orb_type = '-'.join(orb_type)

    return orb_type
