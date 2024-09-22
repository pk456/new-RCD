import os

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score

from RCD.data_loader import TrainDataLoader, ValTestDataLoader
from RCD.dgl_graph import dgl_graph
from RCD.model import Net
from ctl import CommonArgParser


def train(args, local_map):
    data_loader = TrainDataLoader(args.data_name, args.knowledge_n, args.batch_size)
    device = torch.device(('cuda:%d' % (args.gpu)) if torch.cuda.is_available() else 'cpu')

    net = Net(args, local_map)
    net = net.to(device)
    print(net)

    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    print('training model1...')

    loss_function = nn.NLLLoss()
    for epoch in range(args.epoch_n):
        data_loader.reset()
        running_loss = 0.0
        batch_count = 0
        while not data_loader.is_end():
            batch_count += 1
            input_stu_ids, input_exer_ids, input_knowledge_embs, labels = data_loader.next_batch()
            input_stu_ids, input_exer_ids, input_knowledge_embs, labels = input_stu_ids.to(device), input_exer_ids.to(
                device), input_knowledge_embs.to(device), labels.to(device)
            optimizer.zero_grad()
            output_1 = net.forward(input_stu_ids, input_exer_ids, input_knowledge_embs)
            output_0 = torch.ones(output_1.size()).to(device) - output_1
            output = torch.cat((output_0, output_1), 1)

            loss = loss_function(torch.log(output + 1e-10), labels)
            loss.backward()
            optimizer.step()
            net.apply_clipper()

            running_loss += loss.item()

            if batch_count % args.log_interval == 0:
                print(
                    f'train--{epoch + 1}/{args.epoch_n} [{batch_count}/{data_loader.get_batch_num()}] loss: {running_loss / args.log_interval}')
                running_loss = 0.0

        # test and save current model1 every epoch
        if not os.path.exists(args.model_save_dir):
            os.makedirs(args.model_save_dir)
        save_snapshot(net, f'{args.model_save_dir}/model_epoch_{str(epoch + 1)}.pth')

        predict(args, net, epoch)


def predict(args, net, epoch):
    device = torch.device(('cuda:%d' % (args.gpu)) if torch.cuda.is_available() else 'cpu')
    data_loader = TrainDataLoader(args.data_name, args.knowledge_n, args.batch_size, True)
    print('predicting model1...')
    data_loader.reset()
    net.eval()

    correct_count, exer_count = 0, 0
    batch_count, batch_avg_loss = 0, 0.0
    pred_all, label_all = [], []
    while not data_loader.is_end():
        batch_count += 1
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels = data_loader.next_batch()
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels = input_stu_ids.to(device), input_exer_ids.to(
            device), input_knowledge_embs.to(device), labels.to(device)
        output = net.forward(input_stu_ids, input_exer_ids, input_knowledge_embs)
        output = output.view(-1)
        # compute accuracy
        for i in range(len(labels)):
            if (labels[i] == 1 and output[i] > 0.5) or (labels[i] == 0 and output[i] < 0.5):
                correct_count += 1
        exer_count += len(labels)
        pred_all += output.tolist()
        label_all += labels.tolist()

        if batch_count % args.log_interval == 0:
            print(
                f'val--{epoch + 1}/{args.epoch_n} [{batch_count}/{data_loader.get_batch_num()}] acc: {correct_count / exer_count}')

    pred_all = np.array(pred_all)
    label_all = np.array(label_all)
    # compute accuracy
    accuracy = correct_count / exer_count
    # compute RMSE
    rmse = np.sqrt(np.mean((label_all - pred_all) ** 2))
    # compute AUC
    auc = roc_auc_score(label_all, pred_all)
    print('epoch= %d, accuracy= %f, rmse= %f, auc= %f' % (epoch + 1, accuracy, rmse, auc))
    if not os.path.exists(args.result_save_dir):
        os.makedirs(args.result_save_dir)
    with open(f'{args.result_save_dir}/ncd_model_val.txt', 'a', encoding='utf8') as f:
        f.write('epoch= %d, accuracy= %f, rmse= %f, auc= %f\n' % (epoch + 1, accuracy, rmse, auc))


def save_snapshot(model, filename):
    f = open(filename, 'wb')
    torch.save(model.state_dict(), f)
    f.close()


def construct_local_map(args):
    local_map = {
        'directed_g': dgl_graph(args.data_name, 'direct', args.knowledge_n),
        'undirected_g': dgl_graph(args.data_name, 'undirect', args.knowledge_n),
        'e_to_k': dgl_graph(args.data_name, 'e_to_k', args.knowledge_n + args.exer_n),
        'k_to_e': dgl_graph(args.data_name, 'k_to_e', args.knowledge_n + args.exer_n),
        'e_to_u': dgl_graph(args.data_name, 'e_to_u', args.student_n + args.exer_n),
        'u_to_e': dgl_graph(args.data_name, 'u_to_e', args.student_n + args.exer_n),
    }
    return local_map


if __name__ == '__main__':
    args = CommonArgParser().parse_args()
    train(args, construct_local_map(args))
