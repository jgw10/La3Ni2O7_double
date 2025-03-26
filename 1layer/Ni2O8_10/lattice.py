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
Ni_xy = ((-1, -1), (1, 1))
O1_xy = ((-2, -1), (0, -1), (0, 1), (2, 1))
O2_xy = ((-1, -2), (-1, 0), (1, 0), (1, 2))
Oap_xy = ((-1, -1), (1, 1))

if pam.if_change_lattice == 1:
    Ni_xy = ((-1, 1), (1, -1))
    O1_xy = ((-2, 1), (0, 1), (0, -1), (2, -1))
    O2_xy = ((-1, 2), (-1, 0), (1, 0), (1, -2))
    Oap_xy = ((-1, 1), (1, -1))

layer_num = pam.layer_num
Ni_position = [(x, y, z) for x, y in Ni_xy for z in range(0, 2*layer_num-1, 2)]
O1_position = [(x, y, z) for x, y in O1_xy for z in range(0, 2*layer_num-1, 2)]
O2_position = [(x, y, z) for x, y in O2_xy for z in range(0, 2*layer_num-1, 2)]
O_position = O1_position + O2_position

if pam.if_Oap == 1:
    print('Ni2O10')
    # Oap_position = [(x, y, z) for x, y in Oap_xy for z in range(1, 2*layer_num-2, 2)]
    Oap_position = [(x, y, 1) for x, y in Oap_xy]
else:
    print('Ni2O8')
    Oap_position = []
b_x = 5
delta_x = 2
b_y = 5
delta_y = 2
# b_z = 2*layer_num-1
b_z = 2

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
    每层(z)划分为三个区间
    region1: x < 0 and y < 0, 标记region = (-1, z)
    region2: x == 0 or y == 0, 标记region = (0, z)
    region3: x > 0 and y > 0, 标记region = (1, z)
    层间Oap的z减去1
    :param state: state = ((x1, y1, z1, orb1, s1), ...)
    :return:state_type
    """
    # 统计每一层Ni, 层内O, 层间O的数量
    Ni_num = {}
    O_num = {}
    Oap_num = {}
    for hole in state:
        x, y, z, orb, _ = hole
        # Ni
        if orb in pam.Ni_orbs:
            region = (x, z)
            if region in Ni_num:
                Ni_num[region] += 1
            else:
                Ni_num[region] = 1
        # 层内O
        if orb in pam.O_orbs:
            if pam.if_change_lattice == 1:
                if x > 0 and y < 0:
                    region = (1, z)
                elif x < 0 and y > 0:
                    region = (-1, z)
                else:
                    region = (0, z)
            else:
                if x > 0 and y > 0:
                    region = (1, z)
                elif x < 0 and y < 0:
                    region = (-1, z)
                else:
                    region = (0, z)
            if region in O_num:
                O_num[region] += 1
            else:
                O_num[region] = 1
        # 层间O
        if orb in pam.Oap_orbs:
            region = (x, z-1)
            if region in Oap_num:
                Oap_num[region] += 1
            else:
                Oap_num[region] = 1

    # 根据每一层Ni, 层内O和层间O的空穴数量, 生成态的类型
    state_type = {}
    # Ni
    for region, num in Ni_num.items():
        state_type[region] = f'd{10-num}'
    # 层内O
    for region, num in O_num.items():
        if num == 1:
            if region in state_type:
                state_type[region] += 'L'
            else:
                state_type[region] = 'L'
        else:
            if region in state_type:
                state_type[region] += f'L{num}'
            else:
                state_type[region] = f'L{num}'
    # 层间O
    for region, num in Oap_num.items():
        if num == 1:
            if region in state_type:
                state_type[region] += 'O'
            else:
                state_type[region] = 'O'
        else:
            if region in state_type:
                state_type[region] += f'O{num}'
            else:
                state_type[region] = f'O{num}'

    # 先对每层进行处理
    sorted_region = sorted(state_type.keys())
    layer_string = {}
    for region in sorted_region:
        layer_part = state_type[region]
        z = region[-1]
        if z in layer_string:
            layer_string[z] += [layer_part]
        else:
            layer_string[z] = [layer_part]
    # 将每层的字符串按照z升序排列, 并用'-'连接
    state_type = []
    sorted_z = sorted(layer_string.keys())
    for z in sorted_z:
        state_type.append('_'.join(layer_string[z]))
    state_type = '-'.join(state_type)

    return state_type


def get_orb_type(state):
    """
    将具体的state_type细化, 输出具体的d轨道
    层内用'_'连接, 将层内划为两部分, 每层Ni用'-'连接
    每层(z)划分为三个区间
    region1: x < 0 and y < 0, 标记region = (-1, z)
    region2: x == 0 or y == 0, 标记region = (0, z)
    region3: x > 0 and y > 0, 标记region = (1, z)
    层间Oap的z减去1
    :param state:
    :return:
    """
    simple_orbs = {'d3z2r2': 'dz2', 'dx2y2': 'dx2'} # 简化坐标表示
    Ni_orb = {}
    Oap_orb = {}    # 收集层间O的轨道数目
    L_orb = {}  # 收集层内O的轨道数目
    for hole in state:
        x, y, z, orb, _ = hole
        # Ni
        if orb in pam.Ni_orbs:
            if orb in simple_orbs:
                orb = simple_orbs[orb]
            region = (x, z)
            if region in Ni_orb:
                Ni_orb[region] += [orb]
            else:
                Ni_orb[region] = [orb]
        # Oap
        elif orb in pam.Oap_orbs:
            region = (x, z-1)
            if region in Oap_orb:
                Oap_orb[region] += 1
            else:
                Oap_orb[region] = 1
        # 层内O
        else:
            # 若层内O在右上角, 则将x坐标改为和右上角Ni的x相同, 便于合并
            if pam.if_change_lattice == 1:
                if x > 0 and y < 0:
                    region = (1, z)
                elif x < 0 and y > 0:
                    region = (-1, z)
                else:
                    region = (0, z)
            else:
                if x > 0 and y > 0:
                    region = (1, z)
                elif x < 0 and y < 0:
                    region = (-1, z)
                else:
                    region = (0, z)
            if region in L_orb:
                L_orb[region] += 1
            else:
                L_orb[region] = 1

    orb_type = {}
    for region, orbs in Ni_orb.items():
        orbs.sort()
        orb_type[region] = ''.join(orbs)
    for region, num in L_orb.items():
        if region in orb_type:
            if num == 1:
                orb_type[region] += f'L'
            else:
                orb_type[region] += f'L{num}'
        else:
            if num == 1:
                orb_type[region] = f'L'
            else:
                orb_type[region] = f'L{num}'
    for region, num in Oap_orb.items():
        if region in orb_type:
            if num == 1:
                orb_type[region] += f'apz'
            else:
                orb_type[region] += f'apz{num}'
        else:
            if num == 1:
                orb_type[region] = f'apz'
            else:
                orb_type[region] = f'apz{num}'
    sorted_region = sorted(orb_type.keys())
    layer_string = {}
    for region in sorted_region:
        layer_part = orb_type[region]
        z = region[-1]
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


def get_Ni_side_num(state):
    """
    得到和Ni同一边的空穴个数
    改动该函数, 对应basis_change.py中的
    create_singlet_triplet_basis_change_matrix也要一起修改
    :param state:
    :return:
    """
    side1_idx = []
    side2_idx = []
    for idx, hole in enumerate(state):
        x, y = hole[0], hole[1]
        if x < 0 and y < 0:
            side1_idx.append(idx)
        elif x > 0 and y > 0:
            side2_idx.append(idx)
    return side1_idx, side2_idx
