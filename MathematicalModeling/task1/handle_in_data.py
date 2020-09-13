import pandas as pd


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
    all_lay = handle_lay_data()
    print(all_lay)
    data['信誉评级'] = 0
    data['是否违约'] = 0

    c = 0  # 当前data的索引
    num = list(range(1, 124))  # 序号
    for i in data.iloc[:, 0]:
        for j in num:
            if i == j:
                data.loc[c, '信誉评级'] = all_level[num.index(j)]
                data.loc[c, '是否违约'] = all_lay[num.index(j)]
                break
        c += 1

    save = pd.DataFrame(data)
    save.to_csv('new_in_data.csv', encoding='utf_8_sig')
    print('save finished')


def handle_out_ticket():
    data = pd.read_excel('销项发票信息.xlsx')
    all_level = handle_en_msg()
    all_lay = handle_lay_data()
    print(all_lay)
    data['信誉评级'] = 0
    data['是否违约'] = 0

    c = 0  # 当前data的索引
    num = list(range(1, 124))  # 序号
    for i in data.iloc[:, 0]:
        for j in num:
            if i == j:
                data.loc[c, '信誉评级'] = all_level[num.index(j)]
                data.loc[c, '是否违约'] = all_lay[num.index(j)]
                break
        c += 1

    save = pd.DataFrame(data)
    save.to_csv('new_out_data.csv', encoding='utf_8_sig')
    print('save finished')


def add_in_and_out():
    in_data = pd.read_csv('new_in_data.csv')
    out_data = pd.read_csv('new_out_data.csv')
    df = [in_data, out_data]
    data = pd.concat(df, ignore_index=True)
    # 保存
    save = pd.DataFrame(data)
    save.to_csv('in_out_data.csv', encoding='utf_8_sig')
    print('save finished')


'''合并附件2'''


def handle_file2_data():
    in_data = pd.read_excel('in2_data.xlsx')
    out_data = pd.read_excel('out2_data.xlsx')
    df = [in_data, out_data]
    data = pd.concat(df)
    # 保存
    save = pd.DataFrame(data)
    save.to_csv('file2_data.csv', encoding='utf_8_sig')
    print('save finished')


def handle_lay_data():
    data = pd.read_excel('企业信息.xlsx')
    lay = []
    for i in data.iloc[:, -1]:
        lay.append(i)

    return lay


if __name__ == '__main__':
    # handle_en_msg()
    # handle_out_ticket()
    # add_in_and_out()
    handle_file2_data()
    # handle_lay_data()
    # handle_in_ticket()
