import pandas as pd
import time
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg

import parameters as pam
import lattice as lat


def get_ground_state(matrix, VS, multi_S_val, multi_Sz_val, **kwargs):
    """
    求解矩阵的本征值和本征向量, 并对求解结果进行整理
    :param multi_Sz_val:
    :param multi_S_val:
    :param matrix: 哈密顿矩阵
    :param VS: 要求解的所有态
    :return:
    """
    t0 = time.time()
    dim = VS.dim
    Neval = pam.Neval
    vals, vecs = sps.linalg.eigsh(matrix, k=Neval, which='SA')
    print('lowest eigenvalue of H from np.linalg.eigsh = ')
    print(vals)

    # 计算不同的本征值第一次出现的索引，并存储在degen_idx中
    # 对应的简并度即为degen_idx[i+1] - degen_idx[i]
    val_num = pam.val_num
    degen_idx = [0]
    for _ in range(val_num):
        for idx in range(degen_idx[-1] + 1, pam.Neval):
            if abs(vals[idx] - vals[degen_idx[-1]]) > 1e-4:
                degen_idx.append(idx)
                break

    for i in range(val_num):
        print(f'Degeneracy of {i}th state is {degen_idx[i + 1] - degen_idx[i]}')
        print('val = ', vals[degen_idx[i]])
        weight_average = np.average(abs(vecs[:, degen_idx[i]:degen_idx[i + 1]]) ** 2, axis=1)

        # 创建MultiIndex DataFrame, 类似excel格式
        data = {'istate': [], 'state_type': [], 'orb_type': [], 'vec': [], 'weight': []}
        for istate in range(dim):
            weight = weight_average[istate]
            if weight < 1e-4:
                continue
            state = VS.get_state(VS.lookup_tbl[istate])
            state_type = lat.get_state_type(state)
            orb_type = lat.get_orb_type(state)

            data['istate'].append(istate)
            data['state_type'].append(state_type)
            data['orb_type'].append(orb_type)
            data['weight'].append(weight)
            data['vec'].append(vecs[istate, degen_idx[i]:degen_idx[i + 1]])

        df = pd.DataFrame(data)
        df.set_index('istate', inplace=True)
        # 计算state_type总的weight
        df['type_weight'] = df.groupby('state_type')['weight'].transform('sum').round(6)
        df['orb_type_weight'] = df.groupby('orb_type')['weight'].transform('sum').round(6)

        # 依次按type_weight, orb_type_weight降序排列, 并修改原本的df, inplace = True
        df.sort_values(by=['type_weight', 'state_type', 'orb_type_weight', 'orb_type', 'weight'],
                       ascending=[False, True, False, True, False], inplace=True)

        # 只返回基态的数据
        if i == 0:
            select_df = df[['state_type', 'orb_type', 'type_weight', 'orb_type_weight', 'weight']]

        # 先输出state_type: type_weight, 再层次化输出orb_typeP orb_type_weight, 最后输出state和weight
        current_type = None
        current_orb_type = None
        for istate, row in df.iterrows():
            if row['type_weight'] < 0.02:
                continue
            if row['state_type'] != current_type:
                current_type = row['state_type']
                print(f"{current_type} = {row['type_weight']}\n")

            if row['orb_type'] != current_orb_type and row['orb_type_weight'] > 1e-3:
                current_orb_type = row['orb_type']
                print(f"{current_orb_type}: {row['orb_type_weight']}\n")

            state = VS.get_state(VS.lookup_tbl[istate])
            weight = row['weight']
            vec = row['vec']
            if weight < 1e-3:
                continue

            # 将态转为字符串
            state_string = []
            for hole in state:
                x, y, z, orb, s = hole
                hole_string = f'({x}, {y}, {z}, {orb}, {s})'
                state_string.append(hole_string)

            # 将字符串列表分成四个一组
            chunks = [state_string[i: i+4] for i in range(0, len(state_string), 4)]
            # 每个组内用', '连接, 并把组与组之间用'\n\t'连接
            state_string = '\n\t'.join([', '.join(chunk) for chunk in chunks])

            other_string = []
            # 自旋信息转为字符串
            for position in lat.Ni_position:
                if istate in multi_S_val[position]:
                    other_string.append(f'S,Sz{position} = {multi_S_val[position][istate]},{multi_Sz_val[position][istate]}')
            if 'S_val' in kwargs:
                if istate in kwargs['S_val']:
                    other_string.append(f"S_tot, Sz_tot = {kwargs['S_val'][istate]}, {kwargs['Sz_val'][istate]}")
            # 串联字符串
            other_string = '; '.join(other_string)

            # 打印输出
            print(f"\t{state_string}\n\t{other_string}\n\tweight = {weight}\n\tvec = {vec}\n")

    t1 = time.time()
    print(f'gs time {(t1-t0)//60//60}h, {(t1-t0)//60%60}min, {(t1-t0)%60}s\n')

    return vals, select_df
