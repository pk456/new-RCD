# 读取data/ASSIST/graph目录下的e_to_u.txt和u_from_e.txt文件，并输出行数

if __name__ == '__main__':
    u_from_e = []
    with open('../data/ASSIST/graph/k_from_e.txt', 'r') as f:
        u_from_e = f.readlines()
    print(len(u_from_e))
    u_from_e = set(u_from_e)
    e_to_u = set()
    with open('../data/ASSIST/graph/e_to_k.txt', 'r') as f:
        e_to_u = f.readlines()
    e_to_u = set(e_to_u)
    print(len(e_to_u))
    # 利用set运算，输出仅在一个set中的元素
    print(u_from_e - e_to_u)
    print(e_to_u - u_from_e)
    # print(u_from_e)
    # e_to_u = open('../data/ASSIST/graph/e_to_u.txt', 'r')
    # u_from_e = open('../data/ASSIST/graph/u_from_e.txt', 'r')
    # print(len(e_to_u.readlines()))
    # print(len(u_from_e.readlines()))
    # e_to_u.close()
    # u_from_e.close()

