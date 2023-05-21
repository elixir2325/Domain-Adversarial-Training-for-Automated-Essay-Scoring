from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import cohen_kappa_score

def train(train_source_iter, train_target_iter, model, optimizer, config):

    model.train()
    
    train_loss = 0
    src_correct_count = 0
    tgt_correct_count = 0
    count = 0
    # src_loss = 0
    # tgt_loss = 0
    for i in tqdm(range(config.iters_per_epoch)):
        optimizer.zero_grad()
        batch_s = next(train_source_iter)
        batch_t = next(train_target_iter)

        score_s, domain_s, domain_t = model(batch_s, batch_t)

        label_s = torch.ones(config.batch_size, requires_grad=False).long().cuda()
        label_t = torch.zeros(config.batch_size, requires_grad=False).long().cuda()
        src_domain_loss = F.cross_entropy(domain_s, label_s)
        tgt_domain_loss = F.cross_entropy(domain_t, label_t)
        score_loss = F.mse_loss(score_s.view(-1), batch_s['score_scaled'])

        loss = score_loss + config.mu * (src_domain_loss + tgt_domain_loss)
        # loss =  (src_domain_loss + tgt_domain_loss)
        loss.backward()
        optimizer.step()

        model.step()

        train_loss += loss.item()
        # src_loss += src_domain_loss.item()
        # tgt_loss += tgt_domain_loss.item()
        src_correct_count += (torch.argmax(domain_s, dim=-1) == label_s).sum().item()
        tgt_correct_count += (torch.argmax(domain_t, dim=-1) == label_t).sum().item()
        count += config.batch_size
        
    
    train_loss = train_loss / config.iters_per_epoch
    # src_loss = src_loss / config.iters_per_epoch
    # tgt_loss = tgt_loss / config.iters_per_epoch
    return train_loss, src_correct_count / count, tgt_correct_count / count

def quadratic_weighted_kappa(y_pred_scaled, y_true, prompt_ids):
    score_range = {
        1:(2,12),
        2:(1,6),
        3:(0,3),
        4:(0,3),
        5:(0,4),
        6:(0,4),
        7:(0,30),
        8:(0,60)
    }
    min_scores = np.array([score_range[prompt_id][0] for prompt_id in prompt_ids])
    max_scores = np.array([score_range[prompt_id][1] for prompt_id in prompt_ids])
    y_pred = (y_pred_scaled * (max_scores- min_scores)) + min_scores
    y_pred = np.round(y_pred).astype(int)

    kappa_scores =[]
    for i in range(1,9):
        pred_scores_i = y_pred[prompt_ids==i]
        true_scores_i = y_true[prompt_ids==i]
        if len(pred_scores_i) > 0:
            kappa_i = cohen_kappa_score(pred_scores_i, true_scores_i, weights='quadratic')
            kappa_scores.append(kappa_i)

    return np.mean(kappa_scores)

def evaluate(model, test_loader):
    model.eval()
    test_loss = 0

    with torch.no_grad():

        prompt_ids = []
        true_scores = []
        pred_scores = []
        for batch in tqdm(test_loader):
            pred_score = model.inference(batch)
            score_loss = F.mse_loss(pred_score.view(-1), batch['score_scaled'])
            test_loss += score_loss.item()

            prompt_ids.append(batch['essay_set'].view(-1).cpu().numpy())
            true_scores.append(batch['score'].view(-1).cpu().numpy())
            pred_scores.append(pred_score.view(-1).cpu().numpy())

        prompt_ids = np.concatenate(prompt_ids, axis=0)
        true_scores = np.concatenate(true_scores, axis=0)
        pred_scores = np.concatenate(pred_scores, axis=0)

        test_loss = test_loss / len(test_loader)
        kappa = quadratic_weighted_kappa(pred_scores, true_scores, prompt_ids)

    return test_loss, kappa