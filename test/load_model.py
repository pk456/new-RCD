import torch

from RCD.model import Net
from ctl import CommonArgParser
from main import construct_local_map

if __name__=='__main__':

    args = CommonArgParser().parse_args()
    net = Net(args, construct_local_map(args))

    net.load_state_dict(torch.load(r'E:\kt\new-RCD\model3\model_epoch_100.pth'))

    stu = torch.LongTensor([1])
    exe = torch.LongTensor([1])
    knowledge_emb = [0.] * 835
    knowledge_emb[1] = 1
    k = torch.Tensor([knowledge_emb])

    y = net(stu,exe,k)
    print(y)