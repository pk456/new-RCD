import torch
import torch.nn as nn
from .fusion import Fusion


class Net(nn.Module):
    def __init__(self, args, local_map):
        super(Net, self).__init__()

        self.device = torch.device(('cuda:%d' % (args.gpu)) if torch.cuda.is_available() else 'cpu')

        self.knowledge_num = args.knowledge_n
        self.exer_num = args.exer_n
        self.stu_num = args.student_n

        self.emb = args.emb

        self.local_map = {
            'directed_g': local_map['directed_g'].to(self.device),
            'undirected_g': local_map['undirected_g'].to(self.device),
            'e_to_k': local_map['e_to_k'].to(self.device),
            'k_to_e': local_map['k_to_e'].to(self.device),
            'e_to_u': local_map['e_to_u'].to(self.device),
            'u_to_e': local_map['u_to_e'].to(self.device)
        }

        # network structure
        self.knowledge_emb = nn.Embedding(self.knowledge_num, self.emb)
        self.exercise_emb = nn.Embedding(self.exer_num, self.emb)
        self.student_emb = nn.Embedding(self.stu_num, self.emb)

        self.k_index = torch.LongTensor(list(range(self.knowledge_num))).to(self.device)
        self.exer_index = torch.LongTensor(list(range(self.exer_num))).to(self.device)
        self.stu_index = torch.LongTensor(list(range(self.stu_num))).to(self.device)

        self.FusionLayer1 = Fusion(args, self.local_map)
        self.FusionLayer2 = Fusion(args, self.local_map)

        self.prednet_full1 = nn.Linear(2 * self.emb, self.emb, bias=False)
        # self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = nn.Linear(2 * self.emb, self.emb, bias=False)
        # self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = nn.Linear(1 * self.emb, 1)

        # initialization
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id, exer_id, kn_r):
        all_stu_emb = self.student_emb(self.stu_index).to(self.device)
        exer_emb = self.exercise_emb(self.exer_index).to(self.device)
        kn_emb = self.knowledge_emb(self.k_index).to(self.device)

        # Fusion layer 1
        kn_emb, exer_emb, all_stu_emb = self.FusionLayer1(kn_emb, exer_emb, all_stu_emb)
        # Fusion layer 2
        kn_emb, exer_emb, all_stu_emb = self.FusionLayer2(kn_emb, exer_emb, all_stu_emb)

        # get batch student data
        batch_stu_emb = all_stu_emb[stu_id]  # 32 123
        batch_stu_vector = batch_stu_emb.unsqueeze(1).expand(-1, self.knowledge_num, -1)

        # get batch exercise data
        batch_exer_emb = exer_emb[exer_id]  # 32 123
        batch_exer_vector = batch_exer_emb.unsqueeze(1).expand(-1, self.knowledge_num, -1)

        # get batch knowledge concept data
        # 因为一道题目可能会涉及多个知识点，所以不能和student，exercise一样，直接获取batch_emb
        kn_vector = kn_emb.unsqueeze(0).expand(batch_stu_emb.shape[0], -1, -1)

        # Cognitive diagnosis
        # 这里为了方便计算，前面扩展了student和exercise的batch_emb的第二维，和知识点对齐
        # todo:这里可能的优化策略——batch_stu_vector和kn_vector分别进行全连接，然后将结果相加。对比先cat再全连接
        preference = torch.sigmoid(self.prednet_full1(torch.cat((batch_stu_vector, kn_vector), dim=2)))
        diff = torch.sigmoid(self.prednet_full2(torch.cat((batch_exer_vector, kn_vector), dim=2)))
        o = torch.sigmoid(self.prednet_full3(preference - diff))

        # 计算结束，需要筛掉每条数据中没有考察的知识点
        sum_out = torch.sum(o * kn_r.unsqueeze(2), dim=1)
        count_of_concept = torch.sum(kn_r, dim=1).unsqueeze(1)
        output = sum_out / count_of_concept
        return output

    # 确保全连接层的权重值非负
    def apply_clipper(self):
        clipper = NoneNegClipper()
        self.prednet_full1.apply(clipper)
        self.prednet_full2.apply(clipper)
        self.prednet_full3.apply(clipper)


class NoneNegClipper(object):
    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            a = torch.relu(torch.neg(w))
            w.add_(a)
