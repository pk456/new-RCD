import argparse


class CommonArgParser(argparse.ArgumentParser):
    def __init__(self):
        super(CommonArgParser, self).__init__()
        self.add_argument('--exer_n', type=int, default=835,
                          help='The number for exercise.')
        self.add_argument('--knowledge_n', type=int, default=835,
                          help='The number for knowledge concept.')
        self.add_argument('--student_n', type=int, default=10000,
                          help='The number for student.')
        self.add_argument('--gpu', type=int, default=0,
                          help='The id of gpu, e.g. 0.')
        self.add_argument('--epoch_n', type=int, default=20,
                          help='The epoch number of training')
        self.add_argument('--lr', type=float, default=0.0001,
                          help='Learning rate')
        self.add_argument('--emb', type=int, default=64)
        self.add_argument('--batch_size', type=int, default=128)
        self.add_argument('--data_name', type=str, default='junyi')
        self.add_argument('--log_interval', type=int, default=10,
                          help='Output logs every log_interval batches.')
        self.add_argument('--model_save_dir', type=str, default='model',
                          help='The directory for saving the model results of each training round.')
        self.add_argument('--result_save_dir', type=str, default='result',
                          help='The directory for saving the model predict of each training round.')
