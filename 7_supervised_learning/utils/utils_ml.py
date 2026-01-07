import numpy as np
import random
from sklearn import metrics
import torch
from utils.utils_data import label_to_3c_01


def set_random_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)  # gpu
    torch.backends.cudnn.deterministic = True


def get_loss_function(args):
    '''
    Get loss function based on label mode
    '''
    if args.label_mode == '3c':
        loss_func = torch.nn.CrossEntropyLoss()
    elif args.label_mode in ['-1-1']:
        loss_func = torch.nn.MSELoss()
    elif args.label_mode in ['0-1']:
        loss_func = torch.nn.BCELoss()
    else:
        raise ValueError('Unsupported label_mode for loss function.')
    return loss_func


def get_optimazer(args, model):
    '''
    Get optimizer based on args
    '''
    def get_params(model):
        return filter(lambda p: p.requires_grad, model.parameters())
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(get_params(model), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(get_params(model), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'RMSprop':
        optimizer = torch.optim.RMSprop(get_params(model), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError('Unsupported optimizer type.')
    return optimizer


def train_classification_epoch(args,model, train_loader, my_loss, optimizer, epoch, threshold=1.0):
    '''
    Train for one epoch
    '''
    model.train()
    acc = 0
    total = 0
    loss_list = []
    pred_list = []
    true_list = []

    for batch_idx, (x_eeg, x_fnirs, y) in enumerate(train_loader):
        # If you run this code on CPU, please remove the '.to(args.device)'
        x_eeg = x_eeg.to(args.device)
        x_fnirs = x_fnirs.to(args.device)
        y = y.to(args.device)

        optimizer.zero_grad()
        outputs = model(x_eeg, x_fnirs)

        # print(outputs.shape, y.shape)
        loss = my_loss(outputs, y)
        loss.backward()

        torch.nn.utils.clip_grad_value_(model.parameters(), threshold)
        optimizer.step()

        pred = torch.argmax(outputs.data, 1)
        pred_list.append(pred.cpu().numpy())
        true_list.append(y.cpu().numpy())
        acc += ((pred == y).sum()).cpu().numpy()
        total += len(y)
        loss_list.append(loss.item())
        # print("[TR]epoch:%d, step:%d, loss:%f, l1:%f, l2:%f, acc:%f" %
        #       (epoch + 1, batch_idx, loss, result_dic['kl_g'], result_dic['kl_b'], (acc / total)))
    preds = np.concatenate(pred_list)
    trues = np.concatenate(true_list)
    f1score = metrics.f1_score(trues, preds, average='macro')
    loss_mean = np.mean(loss_list)
    print("[TR]epoch:%d, loss:%f, acc:%f, f1s:%f" % (epoch + 1, loss_mean, (acc / total), f1score), end='\t')
    return acc / total, f1score, loss_mean


def val_classification(args,model, val_loader, my_loss, epoch, output=True, tag='VA'):
    '''
    Validate for one epoch
    '''
    model.eval()
    acc = 0
    total = 0
    pred_list = []
    true_list = []
    loss_list = []

    with torch.no_grad():
        for batch_idx, (x_eeg, x_fnirs, y) in enumerate(val_loader):
            # If you run this code on CPU, please remove the '.to(args.device)'
            x_eeg = x_eeg.to(args.device)
            x_fnirs = x_fnirs.to(args.device)
            y = y.to(args.device)

            outputs = model(x_eeg, x_fnirs)
            loss = my_loss(outputs, y)

            pred = torch.argmax(outputs.data, 1)
            pred_list.append(pred.cpu().numpy())
            true_list.append(y.cpu().numpy())
            acc += ((pred == y).sum()).cpu().numpy()
            total += len(y)

            loss_list.append(loss.item())

    pred_list = np.concatenate(pred_list)
    true_list = np.concatenate(true_list)
    f1score = metrics.f1_score(true_list, pred_list, average='macro')
    loss_mean = np.mean(loss_list)
    print("[%s]epoch:%d, loss:%f, acc:%f, f1s:%f" % (tag, epoch + 1, loss_mean, (acc / total), f1score), end='\t')
    
    if output:
        return acc / total, f1score, loss_mean,  pred_list, true_list
    else:
        return acc / total, f1score, loss_mean
    

def train_regression_epoch(args,model, train_loader, my_loss, optimizer, epoch, mode='0-1', threshold=1.0):
    '''
    Train for one epoch
    '''
    model.train()
    loss_list = []
    pred_list = []
    true_list = []

    for batch_idx, (x_eeg, x_fnirs, y) in enumerate(train_loader):
        # If you run this code on CPU, please remove the '.to(args.device)'
        x_eeg = x_eeg.to(args.device)
        x_fnirs = x_fnirs.to(args.device)
        y = y.to(args.device)

        optimizer.zero_grad()
        outputs = model(x_eeg, x_fnirs)

        loss = my_loss(outputs, y)
        loss.backward()

        torch.nn.utils.clip_grad_value_(model.parameters(), threshold)
        optimizer.step()

        pred_value = outputs.data.cpu().numpy()
        true_value = y.cpu().numpy()
        pred_list.append(pred_value)
        true_list.append(true_value)
        loss_list.append(loss.item())
        # print("[TR]epoch:%d, step:%d, loss:%f, l1:%f, l2:%f, acc:%f" %
        #       (epoch + 1, batch_idx, loss, result_dic['kl_g'], result_dic['kl_b'], (acc / total)))
    preds = np.concatenate(pred_list)
    trues = np.concatenate(true_list)
    mae = metrics.mean_absolute_error(trues, preds)
    mse = metrics.mean_squared_error(trues, preds)
    rmse = np.sqrt(mse)
    r2 = metrics.r2_score(trues, preds)
    preds_3c = label_to_3c_01(preds, mode=mode)
    trues_3c = label_to_3c_01(trues, mode=mode)
    acc = ((preds_3c == trues_3c).sum())/len(preds_3c)
    f1score = metrics.f1_score(trues_3c, preds_3c, average='macro')
    loss_mean = np.mean(loss_list)
    print("[TR]epoch:%d, loss:%f, mae:%f, mse:%f, rmse:%f, r2:%f, acc:%f, f1s:%f" % (epoch + 1, loss_mean, mae, mse, rmse, r2, acc, f1score), end='\t')
    return mae, mse, r2, acc, f1score, loss_mean


def val_regression(args,model, val_loader, my_loss, epoch, mode='0-1', output=True, tag='VA'):
    '''
    Validate for one epoch
    '''
    model.eval()
    pred_list = []
    true_list = []
    loss_list = []

    with torch.no_grad():
        for batch_idx, (x_eeg, x_fnirs, y) in enumerate(val_loader):
            # If you run this code on CPU, please remove the '.to(args.device)'
            x_eeg = x_eeg.to(args.device)
            x_fnirs = x_fnirs.to(args.device)
            y = y.to(args.device)

            outputs = model(x_eeg, x_fnirs)
            loss = my_loss(outputs, y)

            pred_list.append(outputs.cpu().numpy())
            true_list.append(y.cpu().numpy())

            loss_list.append(loss.item())

    pred_list = np.concatenate(pred_list)
    true_list = np.concatenate(true_list)
    mae = metrics.mean_absolute_error(true_list, pred_list)
    mse = metrics.mean_squared_error(true_list, pred_list)
    rmse = np.sqrt(mse)
    r2 = metrics.r2_score(true_list, pred_list)
    preds_3c = label_to_3c_01(pred_list, mode=mode)
    trues_3c = label_to_3c_01(true_list, mode=mode)
    acc = ((preds_3c == trues_3c).sum())/len(preds_3c)
    f1score = metrics.f1_score(trues_3c, preds_3c, average='macro')
    loss_mean = np.mean(loss_list)
    print("[%s]epoch:%d, loss:%f, mae:%f, mse:%f, rmse:%f, r2:%f, acc:%f, f1s:%f" % (tag, epoch + 1, loss_mean, mae, mse, rmse, r2, acc, f1score), end='\t')
    
    if output:
        return mae, mse, r2, acc, f1score, loss_mean,  pred_list, true_list
    else:
        return mae, mse, r2, acc, f1score, loss_mean


def PrintScore(true, pred, savePath=None, average='macro',
               labels=[0, 1, 2], classes=['C0', 'C1', 'C2']):
    '''
    Print classification scores
    '''
    if savePath == None:
        saveFile = None
    else:
        saveFile = open(savePath + "Result.txt", 'a+')
    # Main scores
    F1 = metrics.f1_score(true, pred, average=None)
    if len(F1) == 2:
            F1 = np.append(F1, np.nan)
            classes = ['C0', 'C1']
    elif len(F1) == 1:
            F1 = np.array(F1, np.nan, np.nan)
            classes = ['C0']
    if len(F1) != 3:
            print("[Warning] The number of classes is not 3. Please check.")
    print("Main scores:", file=saveFile)
    print('Acc\twF1\tmF1', file=saveFile)
    print('%.4f\t%.4f\t%.4f' %
          (metrics.accuracy_score(true, pred),
           metrics.f1_score(true, pred, average='weighted'),
           metrics.f1_score(true, pred, average='macro')),
          file=saveFile)
    # Classification report
    print()
    print("Classification report:", file=saveFile)
    print(metrics.classification_report(true, pred, target_names=classes, digits=4), file=saveFile) #labels=labels, 
    # Confusion matrix
    print('Confusion matrix:', file=saveFile)
    print(metrics.confusion_matrix(true, pred), file=saveFile)
    if savePath != None:
        saveFile.close()
    return


def ComputeMetrics(label_pair, te_metrics, label_mode):
    '''
    Compute evaluation metrics based on label mode
    '''
    if label_mode == '3c':
        # Classification metrics
        te_metrics['acc'] = metrics.accuracy_score(label_pair['true'], label_pair['pred'])
        te_metrics['f1'] = metrics.f1_score(label_pair['true'], label_pair['pred'], average='macro')
        preds_3c = label_pair['pred']
        trues_3c = label_pair['true']
        print('[Result] Classification:')
    else:
        # Regression metrics, with conversion to 3 categories for acc and f1
        te_metrics['mae'] = metrics.mean_absolute_error(label_pair['true'], label_pair['pred'])
        te_metrics['mse'] = metrics.mean_squared_error(label_pair['true'], label_pair['pred'])
        te_metrics['r2'] = metrics.r2_score(label_pair['true'], label_pair['pred'])
        preds_3c = label_to_3c_01(label_pair['pred'], mode=label_mode)
        trues_3c = label_to_3c_01(label_pair['true'], mode=label_mode)
        te_metrics['acc'] = ((preds_3c == trues_3c).sum())/len(preds_3c)
        te_metrics['f1'] = metrics.f1_score(trues_3c, preds_3c, average='macro')
        print('[Result] Regression:')
        print(f'  MAE: {te_metrics["mae"]:.4f}, MSE: {te_metrics["mse"]:.4f}, R2: {te_metrics["r2"]:.4f}')
    PrintScore(trues_3c, preds_3c)
    return te_metrics