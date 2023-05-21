import argparse
import random
import os 

import torch 
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader

from configs import Configs
from utils.read_data import read_data
from utils.loader_utils import EssayDataset, collate_funcion, ForeverDataIterator
from models.model import BERTEncoder, RegressorWithDANN
from train.trainer import train, evaluate
from utils.analysis import collect_feature, visualize

import logging
import sys
import os
import datetime

now = datetime.datetime.now()

def get_logger(name, level=logging.INFO, filepath='bert_dann/logs/log_{}.txt'.format(now.strftime("%Y-%m-%d %H-%M-%S")),
        formatter='%(asctime)s - %(name)s - %(levelname)s - %(message)s'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(formatter)
    file_handler = logging.FileHandler(filepath, mode='w+')
    stream_handler = logging.StreamHandler(sys.stdout)

    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger


os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def main(args):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        # cudnn.deterministic = True
        # warnings.warn('You have chosen to seed training. '
        #               'This will turn on the CUDNN deterministic setting, '
        #               'which can slow down your training considerably! '
        #               'You may see unexpected behavior when restarting '
        #               'from checkpoints.')
    # cudnn.benchmark = True
    logger = get_logger('BERT DANN')
    configs = Configs()
    logger.info('source_prompt_id_{} target_prompt_id_{}  mu_{}'.format(args.source_prompt_id, args.target_prompt_id, configs.mu))
    train_df, dev_df, test_df = read_data(args.source_prompt_id, args.target_prompt_id, configs)
    train_dataset, dev_dataset, test_dataset = EssayDataset(train_df, configs), EssayDataset(dev_df, configs), EssayDataset(test_df, configs)

    train_source_loader = DataLoader(train_dataset, configs.batch_size, shuffle=True, drop_last=True, collate_fn=collate_funcion)
    train_target_loader = DataLoader(test_dataset, batch_size=configs.batch_size, shuffle=True, drop_last=True, collate_fn=collate_funcion)
    dev_loader = DataLoader(dev_dataset, batch_size=configs.batch_size * 16, shuffle=False, drop_last=False, collate_fn=collate_funcion)
    test_loader = DataLoader(test_dataset, batch_size=configs.batch_size * 16, shuffle=False, drop_last=False, collate_fn=collate_funcion)
    
    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)

    encoder = BERTEncoder(config=configs).cuda()
    model = RegressorWithDANN(encoder, config=configs).cuda()
    print(model)

    optimizer = torch.optim.RMSprop(model.get_parameters(base_lr=configs.lr))
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=configs.lr)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
    #                                     lr_lambda=lambda epoch: 0.95 ** epoch, verbose=True)
    
    best_dev_kappa = -1
    best_test_kappa = -1
    
    for epoch in range(configs.epochs):
        train_loss, src_accuracy, tgt_accuracy = train(train_source_iter, train_target_iter, model, optimizer, config=configs)
        dev_loss, dev_kappa = evaluate(model, dev_loader)
        test_loss, test_kappa = evaluate(model, test_loader)
        
        if dev_kappa > best_dev_kappa:
            best_dev_kappa = dev_kappa
            best_test_kappa = test_kappa

            if args.analysis == True:
                feature_extractor = model.encoder
                source_feature = collect_feature(train_source_loader, feature_extractor)
                target_feature = collect_feature(train_target_loader, feature_extractor)
                filename = 'bert_dann/images/{}_to_{}_mu_{}_lr_{}_epoch_{}_tsne.png'.format(args.source_prompt_id, args.target_prompt_id, configs.mu, configs.lr, epoch+1)
                visualize(source_feature, target_feature, filename)
                print("Saving t-SNE to", filename)
    
        logger.info("epoch {}   train_loss {:.6f}    src_accuracy {:.3f}    tgt_accuracy {:.3f}   dev_loss {:.6f}    dev_kappa {:.3f}\
                        test_loss {:.6f}    test_kappa {:.3f}    best_dev_kappa {:.3f}    best_test_kappa {:.3f}"\
                    .format(epoch+1, train_loss, src_accuracy, tgt_accuracy, dev_loss, dev_kappa, test_loss, test_kappa, best_dev_kappa, best_test_kappa))
        
        
        
        # scheduler.step()

    return


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="bert domain adversarial neural network")
    parser.add_argument('--source_prompt_id', type=int, default=1, help='labeled source prompt id')
    parser.add_argument('--target_prompt_id', type=int, default=2, help='unlabeled target prompt id')
    parser.add_argument('--seed', type=int, default=12, help='random seed')
    parser.add_argument('--analysis', type=bool, default=True, help='random seed')
    args = parser.parse_args()
    main(args)

    
