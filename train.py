#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @author Xu
# @date 2022/6/17
# @file cudatest.py
import torch
import os
import time
import math
import logging
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
# from utils.dialogue_dataset import dialogue_dataset,collate_func
from model.transformer_base import transformer_base
from utils.tokenizer import basic_tokenizer
from utils.loss import seq_generation_loss
from utils._utils import reset_log
from DataPreprocess import LoadData

def Train(args):
    #   super params
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_path = args['trainset_path']
    dev_path = args['testset_path']
    resume = args['resume']
    checkpoint_path = args['checkpoint_path']
    history_path = args['history_path']
    log_path = args['log_path']
    vocab_path = args['vocab_path']
    model_name = args['model_save_name']
    model_resume_name = args['model_resume_name']
    batch_size = args['batch_size']
    end_epoch = args['end_epoch']
    lr = args['lr']
    loss_check_freq = args['loss_check']
    check_steps = args['check_steps']
    save_steps = args['save_steps']
    embed_path = args['embed_path']
    embed_dim = args['embed_dim']
    nheads = args['nheads_transformer']
    #########
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    if not os.path.exists(history_path):
        os.makedirs(history_path)

    log_save_name = 'log_' + model_name + '.log'
    reset_log(log_path + log_save_name)
    #logger用于数据分析
    logger = logging.getLogger(__name__)
    for k, v in args.items():
        logger.info(k + ':' + str(v))
    checkpoint_name = os.path.join(checkpoint_path, model_name + '_best_ckpt.pth')
    model_ckpt_name = os.path.join(checkpoint_path, model_name + '_best.pkl')

    if not model_resume_name:
        model_resume_name = model_ckpt_name
    localtime = time.asctime(time.localtime(time.time()))
    logger.info('#####start time:%s' % (localtime))
    time_stamp = int(time.time())
    logger.info('time stamp:%d' % (time_stamp))
    logger.info('######Model: %s' % (model_name))
    logger.info('trainset path ：%s' % (train_path))
    logger.info('valset path: %s' % (dev_path))
    logger.info('batch_size:%d' % (batch_size))
    logger.info('learning rate:%f' % (lr))
    logger.info('end epoch:%d' % (end_epoch))

    tokenizer = basic_tokenizer(vocab_path)
    with open('./data/corpus_train.txt', "r", encoding="utf-8") as fp:
        txt1 = fp.readlines()
    with open('./data/corpus_test.txt', "r", encoding="utf-8") as fp:
        txt2 = fp.readlines()
    train_data = LoadData(txt1)
    test_data = LoadData(txt2)
    train_loader = DataLoader(train_data, batch_size=batch_size,shuffle=False, drop_last=True)
    dev_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    model = transformer_base(tokenizer.vocab_size, embed_dim, nheads, embed_path)
    model.to(device)

    if resume != 0:
        logger.info('Resuming from checkpoint...')
        model.load_state_dict(torch.load(model_resume_name))
        checkpoint = torch.load(checkpoint_name)
        best_loss = checkpoint['loss']
        start_epoch = checkpoint['epoch']
        history = checkpoint['history']
    else:
        best_loss = math.inf
        start_epoch = -1
        history = {'train_loss': [], 'val_loss': []}

    criterion = seq_generation_loss(device=device).to(device)
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)  # weight_decay=1e-5
    scheduler = StepLR(optim, step_size=5, gamma=0.9)

    steps_cnt = 0
    for epoch in range(start_epoch + 1, end_epoch):
        print('-------------epoch:%d--------------' % (epoch))
        logger.info('-------------epoch:%d--------------' % (epoch))
        model.train()
        loss_tr = 0
        local_steps_cnt = 0
        #########   train ###########
        print('start training!')
        for batch_idx, batch in tqdm(enumerate(train_loader)):
            src_batch, tgt_batch = batch[0], batch[1]

            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)

            model.zero_grad()
            out = model(src_batch, tgt_batch)
            loss = criterion(out, tgt_batch)
            loss.backward()  # compute gradients
            optim.step()  # update parameters
            steps_cnt += 1
            local_steps_cnt += 1
            loss_tr += loss.item()

            if batch_idx % loss_check_freq == 0:
                print('batch:%d' % (batch_idx))
                print('loss:%f' % (loss.item()))

            if steps_cnt % check_steps == 0:
                loss_tr /= local_steps_cnt
                print('trainset loss:%f' % (loss_tr))
                logger.info('trainset loss:%f' % (loss_tr))
                history['train_loss'].append(loss_tr)
                loss_tr = 0
                local_steps_cnt = 0
                #########  val ############
                loss_va = 0
                model.eval()
                with torch.no_grad():
                    print('start validating!')
                    for batch_idx, batch in tqdm(enumerate(dev_loader),
                                                 total=int(len(dev_loader.dataset) / batch_size) + 1):
                        src_batch, tgt_batch = batch[0], batch[1]

                        src_batch = src_batch.to(device)
                        tgt_batch = tgt_batch.to(device)
                        model.zero_grad()
                        out = model(src_batch, tgt_batch)
                        loss = criterion(out, tgt_batch)
                        loss_va += loss.item()

                    loss_va = loss_va / (batch_idx + 1)
                    print('valset loss:%f' % (loss_va))
                    logger.info('valset loss:%f' % (loss_va))
                    history['val_loss'].append(loss_va)

                    # save checkpoint and model
                    if loss_va < best_loss:
                        logger.info('Checkpoint Saving...')
                        print('best loss so far! Checkpoint Saving...')
                        state = {
                            'epoch': epoch,
                            'loss': loss_va,
                            'history': history
                        }
                        torch.save(state, checkpoint_name)
                        best_loss = loss_va
                        ## save model
                        torch.save(model.state_dict(), model_ckpt_name)
                scheduler.step()
                logger.info("current lr:%f" % (scheduler.get_last_lr()[0]))
                model.train()
            if steps_cnt % save_steps == 0:
                logger.info('match save steps,Checkpoint Saving...')
                torch.save(model.state_dict(),
                           os.path.join(checkpoint_path, model_name + '_steps_' + str(steps_cnt) + '.pkl'))

if __name__ == '__main__':
    args = {
        'trainset_path': './data/trainset/train.txt',
        'testset_path': './data/trainset/valid.txt',
        'checkpoint_path': './ckpt/',
        'history_path': './history/',
        'log_path': './log/',
        'vocab_path': './data/basic_vob.txt',
        'embed_path': '',  # default: ''
        'embed_dim': 300,  # default: 512
        'nheads_transformer': 15,  # embed_dim % nheads_transformer == 0
        'resume': 0,
        'model_save_name': 'trans_xhj_v2',
        'model_resume_name': '',
        'batch_size': 24,
        'end_epoch': 60,
        'check_steps': 2000,
        'save_steps': 5000,
        'lr': 1e-4,
        'loss_check': 300,
        'version_info': 'use pretrained embed , encode_layers=6 model.train() revise',
    }
    Train(args)
