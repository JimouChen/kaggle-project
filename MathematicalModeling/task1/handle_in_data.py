"""
# @Time    :  2020/9/10
# @Author  :  Jimou Chen
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def handle_en_msg():
    data = pd.read_excel('企业信息.xlsx')
    '''求D等级的企业'''
    d_level = []
    c = 1  # 企业代号
    for i in data.iloc[:, 2]:
        # print(i)
        if i == 4:
            d_level.append(c)
        c += 1
    print('D等级的企业: ', d_level)

    '''求企业对应的等级'''
    all_level = []
    for i in data.iloc[:, 2]:
        all_level.append(i)

    return all_level


def handle_in_ticket():
    data = pd.read_excel('in_data.xlsx')
    all_level = handle_en_msg()
    data['信誉评级'] = 0
    c = 0  # 当前data的索引
    num = list(range(0, 124))  # 序号
    for i in data.iloc[:, 0]:
        for j in all_level:
            if i == j:
                # 修改信誉的值
                data.loc[c, '信誉评级'] = all_level[num.index(j)]
                break
        c += 1
    # 保存
    save = pd.DataFrame(data)
    save.to_csv('new_in_data.csv', encoding='utf_8_sig')
    print('save finished')


def handle_out_ticket():
    data = pd.read_excel('销项发票信息.xlsx')

    # 保存
    # save = pd.DataFrame(x_data)
    # save.to_csv('new_out_ticket.csv', encoding='utf_8_sig')
    # print('save finished')


if __name__ == '__main__':
    # handle_en_msg()
    handle_in_ticket()
