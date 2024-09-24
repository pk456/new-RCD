import torch

from RCD.model import Net
from ctl import CommonArgParser
from main import construct_local_map, adjust_args

if __name__ == '__main__':
    args = CommonArgParser().parse_args()
    adjust_args(args)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = Net(args, construct_local_map(args))
    net = net.to(device)
    net.eval()
    net.load_state_dict(torch.load(r'model/c-20/model_epoch_20.pth'))

    stu = torch.LongTensor([1]).to(device)
    exe = torch.LongTensor([1]).to(device)
    knowledge_emb = [0.] * args.knowledge_n
    knowledge_emb[1] = 1
    k = torch.Tensor([knowledge_emb]).to(device)

    y = net(stu, exe, k)
    edge = net.FusionLayer1.k_to_e.g.edges(form='all')
    print(edge)

    # print(y)

