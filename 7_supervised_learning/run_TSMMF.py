import os
import gc
from utils.utils_basic import *
from utils.utils_args import *
from utils.utils_data import *
from utils.utils_ml import *
from torch.utils.data import DataLoader
from models.TSMMF_LateFusion import HybridTransformer as TSMMF


'''
Samples of training and evaluation of the TSMMF model on the REFED dataset for emotion recognition using EEG and fNIRS data.
- Data_split_strategy: leave-one-trial-out cross-validation, 10% validation set from training set.
- Training settings: refer to 'utils.utils_args.py'.
'''

if __name__ == "__main__":
    print_with_time('Start', '#')
    
    # Parse arguments
    args = get_args()
    args.device = torch.device(f"cuda:{args.cuda}") if torch.cuda.is_available() else torch.device("cpu")
    print('[Arguments]', args)
    print('[PyTorch]', torch.__version__)

    # Set random seed for reproducibility
    set_random_seed(args.seed)

    # make output directory
    out_dir_base = f'./Results_TSMMF/' + f'label{args.label_mode}_{args.modality}_{args.label_dim}_{args.seed}_lr{args.lr}_wd{args.weight_decay}_bs{args.batch_size}/'
    if os.path.exists(out_dir_base):
        print('[Warning] out_dir_base exists!')

    # Data list (Subjects 1 to 32)
    data_list = ['%d'%i for i in range(1,33)]
    print('Data list:', data_list)

    # Main training and evaluation loop
    all_metrics = {}
    for si in data_list:
        print_with_time(f'Training on Subject {si}', '=')
        
        out_dir = os.path.join(out_dir_base, si)
        os.makedirs(out_dir, exist_ok=True)

        data = load_data(si, args.path_data, args.path_label, 
                         modality= ['EEG', 'fNIRS'] if args.modality=='both' else [args.modality],
                         info= ['video', 'baseline', 'label'])
        data = process_data(data)
        data = process_label(data, args.label_dim, args.label_mode)

        ########## Model Training ##########
        # Classification metrics storage
        if args.label_mode == '3c':
            te_metrics_si = {'acc': [], 'f1': []}
        else:
            te_metrics_si = {'mae': [], 'mse': [], 'r2': [], 'acc': [], 'f1': []}
        label_pair_si = {'true' : [], 'pred' : []}

        # Cross-validation over trials
        for fi in range(args.num_trial):
            print_with_time(f'Training Fold {fi}', '-')

            # Prepare data for the current fold
            data_train_fi, data_valid_fi, data_test_fi = get_fold_data(data, fi, val_ratio=0.1, label_mode=args.label_mode)
            print('[Data] Train:', data_train_fi['EEG'].shape, data_train_fi['fNIRS'].shape, data_train_fi['label'].shape)
            print('[Data] Valid:', data_valid_fi['EEG'].shape, data_valid_fi['fNIRS'].shape, data_valid_fi['label'].shape)
            print('[Data] Test :', data_test_fi['EEG'].shape, data_test_fi['fNIRS'].shape, data_test_fi['label'].shape)

            # Create datasets and dataloaders
            trDataset = EEG_fNIRS_Dataset(data_train_fi['EEG'], data_train_fi['fNIRS'], data_train_fi['label'], args.label_mode)
            vaDataset = EEG_fNIRS_Dataset(data_valid_fi['EEG'], data_valid_fi['fNIRS'], data_valid_fi['label'], args.label_mode)
            teDataset = EEG_fNIRS_Dataset(data_test_fi['EEG'], data_test_fi['fNIRS'], data_test_fi['label'], args.label_mode)
            trGen = DataLoader(trDataset, batch_size = args.batch_size, shuffle = True, num_workers = args.load_workers)
            vaGen = DataLoader(vaDataset, batch_size = args.batch_size, shuffle = False, num_workers = args.load_workers)
            teGen = DataLoader(teDataset, batch_size = args.batch_size, shuffle = False, num_workers = args.load_workers)

            # Initialize model, loss function, and optimizer
            model = TSMMF(args, final_activation=args.final_activation, device = args.device).to(args.device)
            optimizer = get_optimazer(args, model)
            loss_func = get_loss_function(args)

            best_metric = 0 if args.label_mode == '3c' else np.inf # for early stopping
            patience_cnt = 0

            # Training loop for epochs
            for epoch in range(args.n_epoch):
                time_start = time.time()
                if args.label_mode == '3c':
                    # Classification
                    tr_acc, tr_f1s, tr_loss = train_classification_epoch(args,model, trGen, loss_func, optimizer, epoch)
                    va_acc, va_f1s, va_loss = val_classification(args,model, vaGen, loss_func, epoch, output=False)
                    te_acc, te_f1s, te_loss = val_classification(args,model, teGen, loss_func, epoch, output=False, tag='TE')
                else:
                    # Regression
                    tr_mae, tr_mse, tr_r2, tr_acc, tr_f1s, tr_loss = train_regression_epoch(args,model, trGen, loss_func, optimizer, epoch, mode=args.label_mode)
                    va_mae, va_mse, va_r2, va_acc, va_f1s, va_loss = val_regression(args,model, vaGen, loss_func, epoch, mode=args.label_mode, output=False)
                    te_mae, te_mse, te_r2, te_acc, te_f1s, te_loss = val_regression(args,model, teGen, loss_func, epoch, mode=args.label_mode, output=False, tag='TE')

                # early stopping and best model saving
                if args.label_mode == '3c' and va_acc > best_metric:
                    best_metric = va_acc
                    patience_cnt = 0
                    torch.save(model.state_dict(), os.path.join(out_dir, f'best_model_fold{fi}.pth'))
                    print('U', end=' ')
                elif args.label_mode != '3c' and va_mae < best_metric:
                    best_metric = va_mae
                    patience_cnt = 0
                    torch.save(model.state_dict(), os.path.join(out_dir, f'best_model_fold{fi}.pth'))
                    print('U', end=' ')
                else:
                    patience_cnt += 1

                time_end = time.time()
                print(f'(Time: {time_end - time_start:.2f} s)')

                if patience_cnt >= args.patience:
                    print(f'Early stopping at epoch {epoch+1}.')
                    break
                
            # Save last model
            torch.save(model.state_dict(), os.path.join(out_dir, f'last_model_fold{fi}.pth'))

            # Load best model for testing
            print_with_time(f'Load for testing', '-')
            model.eval()
            model.load_state_dict(torch.load(os.path.join(out_dir, f'best_model_fold{fi}.pth')))

            # Final evaluation on test set
            if args.label_mode == '3c':
                te_acc, te_f1s, te_loss, te_pred, te_true = val_classification(args, model, teGen, loss_func, epoch, output=True, tag='TE')
                [te_metrics_si[k].append(v) for k, v in zip(['acc', 'f1'], [te_acc, te_f1s])]
            else:
                te_mae, te_mse, te_r2, te_acc, te_f1s, te_loss, te_pred, te_true = val_regression(args, model, teGen, loss_func, epoch, mode=args.label_mode, output=True, tag='TE')
                [te_metrics_si[k].append(v) for k, v in zip(['mae', 'mse', 'r2', 'acc', 'f1'], [te_mae, te_mse, te_r2, te_acc, te_f1s])]
            label_pair_si['true'].append(te_true)
            label_pair_si['pred'].append(te_pred)

            # Clean up cache
            torch.cuda.empty_cache()
            del model, optimizer, trGen, vaGen, teGen, trDataset, vaDataset, teDataset, data_train_fi, data_valid_fi, data_test_fi, loss_func
            gc.collect()

            print()

        # Summary for the subject
        print_with_time(f'Subject [{si}] Summary', '-')
        np.savez(os.path.join(out_dir, 'output.npz'), label_pair_si)
        label_pair_si['true'] = np.concatenate(label_pair_si['true'])
        label_pair_si['pred'] = np.concatenate(label_pair_si['pred'])
        te_metrics_si = ComputeMetrics(label_pair_si, te_metrics_si, args.label_mode)
        all_metrics[si] = te_metrics_si
        
    # Overall summary
    print_with_time('Training Finished', '#')
    print()
    print('Overall Summary:')
    for si in data_list:
        print('Subject [%s]: '%si, all_metrics[si])
    print('Overall Average Metrics:')
    print('  Acc: %.4f (std:%.4f)' % (np.mean([all_metrics[i]['acc'] for i in all_metrics]), np.std([all_metrics[i]['acc'] for i in all_metrics])))
    print('  F1 : %.4f (std:%.4f)' % (np.mean([all_metrics[i]['f1'] for i in all_metrics]), np.std([all_metrics[i]['f1'] for i in all_metrics])))
    if args.label_mode != '3c':
        print('  MAE: %.4f (std:%.4f)' % (np.mean([all_metrics[i]['mae'] for i in all_metrics]), np.std([all_metrics[i]['mae'] for i in all_metrics])))
        print('  MSE: %.4f (std:%.4f)' % (np.mean([all_metrics[i]['mse'] for i in all_metrics]), np.std([all_metrics[i]['mse'] for i in all_metrics])))
        print('  R2 : %.4f (std:%.4f)' % (np.mean([all_metrics[i]['r2'] for i in all_metrics]), np.std([all_metrics[i]['r2'] for i in all_metrics])))
