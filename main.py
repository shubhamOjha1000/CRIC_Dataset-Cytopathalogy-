import argparse
import os
from dataset import CricDataset
import torch
from torch.utils.data import DataLoader
from model import CustomResNet, Multi_head_MLP, Multi_head_CNN_MLP_1, Multi_head_CNN_MLP_2, EnsembleModel_CNN_ViT_1, EnsembleModel_CNN_ViT_2, Custom_ViT 
import numpy as np
import pandas as pd
import engine 
from utils import Binary_classification_metrices, Multiclass_classification_metrices

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', default=2, type=int, help='2:- Normal vs Abnormal, 3:- Normal vs Low Grade vs High Grade, 6:- NILM, ASC-US, LSIL, ASC-H, HSIL, SCC')
    parser.add_argument('--num_epochs', default=100, type=int, help= 'Number of total training epochs')
    parser.add_argument('--img_dir', default='/path', type=str, help='path to the image directory')
    parser.add_argument('--model', default='CustomResNet', type=str, help='Model to be used')
    parser.add_argument('--loss', default='cross_entropy', type=str, help='loss to be used')
    parser.add_argument('--multi_head_loss', default='CombinedMultiHeadLoss', type=str, help='loss to be used for multi-heads')
    parser.add_argument('--num_heads', default=8, type=int, help='No of heads in case of multi-heads')
    parser.add_argument('--epsilon', default=0.2, type=int, help='epsilon value used in multi_head_loss')
    parser.add_argument('--batch_size', default=256, type=int, help='Batch size for the dataloader')
    parser.add_argument('--num_workers', default=4, type=int, help='num workers for dataloader')
    parser.add_argument('--lr', default=0.0002, type=float, help='Initial learning rate')
    parser.add_argument('--weight_decay', default=5e-3, type=float, help='Weight Decay')
    parser.add_argument('--patience', default=20, type=int, help='Representing the number of consecutive epochs where the performance metric does not improve before training stops')
    parser.add_argument('--folds', default=10, type=int, help='No of folds in K-folds')
    parser.add_argument('--combine_results', default='entropy', type=str, help='how to combine output of diff ensemble models :- avg, entropy, fuzzy')
    parser.add_argument('--expt_name', default='expt_1', type=str, help='current expt')
    parser.add_argument('--transformed', default=0, type=int, help='transformed labels')
    
    
    args = parser.parse_args()

    avg_val_metrices = {
        'val_auc' : [],
        'val_acc' : [],
        'val_p' : [],
        'val_r' : [],
        'val_Sens' : [],
        'val_Spec' : []
    }

    variance_val_metrices = {
        'val_auc' : [],
        'val_acc' : [],
        'val_p' : [],
        'val_r' : [],
        'val_Sens' : [],
        'val_Spec' : []
    }

    avg_test_metrices = {
        'test_auc' : [],
        'test_acc' : [],
        'test_p' : [],
        'test_r' : [],
        'test_Sens' : [],
        'test_Spec' : []
    }

    variance_test_metrices = {
        'test_auc' : [],
        'test_acc' : [],
        'test_p' : [],
        'test_r' : [],
        'test_Sens' : [],
        'test_Spec' : []
    }


    binary_val_metrices = {
        'fold' : [],
        'val_auc' : [],
        'val_acc' : [],
        'val_p' : [],
        'val_r' : [],
        'val_Sens' : [],
        'val_Spec' : []
    }

    multiclass_val_metrices = {
        'fold' : [],
        'val_auc' : [],
        'val_acc' : [],
        'val_p' : [],
        'val_r' : []
    }

    binary_test_metrices = {
        'fold' : [],
        'test_auc' : [],
        'test_acc' : [],
        'test_p' : [],
        'test_r' : [],
        'test_Sens' : [],
        'test_Spec' : []
    }

    multiclass_test_metrices = {
        'fold' : [],
        'test_auc' : [],
        'test_acc' : [],
        'test_p' : [],
        'test_r' : [],
    }


    for fold in range(args.folds):
        print(f'folde :- {fold}')

        train = str(fold) + 'train.csv'
        if args.transformed == 1:
            train_path = os.path.join(os.getcwd(), 'Transformed_Train_Val_split', train)
        else:
            train_path = os.path.join(os.getcwd(), 'Train_Val_split', train)
        train_dataset = CricDataset(img_dir = args.img_dir, annotation_file = train_path, img_transform = True)
        train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers)

        val = str(fold) + 'val.csv'
        if args.transformed == 1:
            val_path = os.path.join(os.getcwd(), 'Transformed_Train_Val_split', val)
        else:
            val_path = os.path.join(os.getcwd(), 'Train_Val_split', val)
        val_dataset = CricDataset(img_dir = args.img_dir, annotation_file = val_path, img_transform = True)
        val_loader = DataLoader(val_dataset, batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers)

        if args.transformed == 1:
            test_path = os.path.join(os.getcwd(), 'Transformed_Test', 'test.csv')
        else:
            test_path = os.path.join(os.getcwd(), 'Test', 'test.csv')
        test_dataset = CricDataset(img_dir = args.img_dir, annotation_file = test_path, img_transform = True)
        test_loader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers)
        

        

        # model :- 
        if args.model == 'CustomResNet':
            model = CustomResNet(args.num_classes)
            device = torch.device("cuda")
            model.to(device)

        elif args.model == 'Custom_ViT':
            model = Custom_ViT(args.num_classes)
            device = torch.device("cuda")
            model.to(device)

        elif args.model == 'Multi_head_MLP':
            model = Multi_head_MLP(args.num_classes, args.num_heads)
            device = torch.device("cuda")
            model.to(device)

        elif args.model == 'Multi_head_CNN_MLP_1':
            model = Multi_head_CNN_MLP_1(args.num_classes, args.num_heads)
            device = torch.device("cuda")
            model.to(device)

        elif args.model == 'Multi_head_CNN_MLP_2':
            model = Multi_head_CNN_MLP_2(args.num_classes, args.num_heads)
            device = torch.device("cuda")
            model.to(device)

        elif args.model == 'EnsembleModel_CNN_ViT_1':
            model = EnsembleModel_CNN_ViT_1(args.num_classes)
            device = torch.device("cuda")
            model.to(device)

        elif args.model == 'EnsembleModel_CNN_ViT_2':
            model = EnsembleModel_CNN_ViT_2(args.num_classes, args.num_heads)
            device = torch.device("cuda")
            model.to(device)



        # set up the optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.99))


        patience = args.patience
        best_val_loss = np.inf
        epochs_without_improvement = 0
        best_model_state = None
        

        
        for epoch in range(args.num_epochs):
            print(f'epoch :-{epoch}')
            engine.train(train_loader, model, args.model, optimizer, args.loss, args.multi_head_loss, args.epsilon, args.num_heads)
            
            

            _, _, val_loss = engine.val(val_loader, model, args.model, args.loss, args.num_heads, args.combine_results)

            # early stopping :- 
            if epoch>=10:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                    # Save the state dictionary of the best model
                    best_model_state = model.state_dict()

                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= patience:
                        print(f"Early stopping after {epoch+1} epochs without improvement.")
                        break

        # Load the best model state dictionary for val metrices :-
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        val_predictions, val_labels, _ = engine.val(val_loader, model, args.model, args.loss, args.num_heads, args.combine_results)
        

        
         # Val Metrices :-
        if args.num_classes == 2:
            val_auc, val_acc, val_p, val_r, val_Sens, val_Spec = Binary_classification_metrices(val_predictions, val_labels)
            binary_val_metrices['fold'].append(fold)
            binary_val_metrices['val_auc'].append(val_auc*100)
            binary_val_metrices['val_acc'].append(val_acc*100)
            binary_val_metrices['val_p'].append(val_p*100)
            binary_val_metrices['val_r'].append(val_r*100)
            binary_val_metrices['val_Sens'].append(val_Sens*100)
            binary_val_metrices['val_Spec'].append(val_Spec*100)
        
        elif args.num_classes == 3:
            val_auc, val_acc, val_p, val_r = Multiclass_classification_metrices(val_predictions, val_labels, num_classes=args.num_classes)
            multiclass_val_metrices['fold'].append(fold)
            multiclass_val_metrices['val_auc'].append(val_auc*100)
            multiclass_val_metrices['val_acc'].append(val_acc*100)
            multiclass_val_metrices['val_p'].append(val_p*100)
            multiclass_val_metrices['val_r'].append(val_r*100)
            
        elif args.num_classes == 6:
            val_auc, val_acc, val_p, val_r = Multiclass_classification_metrices(val_predictions, val_labels, num_classes=args.num_classes)
            multiclass_val_metrices['fold'].append(fold)
            multiclass_val_metrices['val_auc'].append(val_auc*100)
            multiclass_val_metrices['val_acc'].append(val_acc*100)
            multiclass_val_metrices['val_p'].append(val_p*100)
            multiclass_val_metrices['val_r'].append(val_r*100)


        test_predictions, test_labels = engine.test(test_loader, model, args.model, args.combine_results)
        

        
        # Test Metrices :- 
        if args.num_classes == 2:
            test_auc,test_acc, test_p, test_r, test_Sens, test_Spec = Binary_classification_metrices(test_predictions, test_labels)
            binary_test_metrices['fold'].append(fold)
            binary_test_metrices['test_auc'].append(test_auc*100)
            binary_test_metrices['test_acc'].append(test_acc*100)
            binary_test_metrices['test_p'].append(test_p*100)
            binary_test_metrices['test_r'].append(test_r*100)
            binary_test_metrices['test_Sens'].append(test_Sens*100)
            binary_test_metrices['test_Spec'].append(test_Spec*100)

        elif args.num_classes == 3:
            test_auc, test_acc, test_p, test_r = Multiclass_classification_metrices(test_predictions, test_labels, num_classes=args.num_classes)
            multiclass_test_metrices['fold'].append(fold)
            multiclass_test_metrices['test_auc'].append(test_auc*100)
            multiclass_test_metrices['test_acc'].append(test_acc*100)
            multiclass_test_metrices['test_p'].append(test_p*100)
            multiclass_test_metrices['test_r'].append(test_r*100)

        elif args.num_classes == 6:
            test_auc, test_acc, test_p, test_r = Multiclass_classification_metrices(test_predictions, test_labels, num_classes=args.num_classes)
            multiclass_test_metrices['fold'].append(fold)
            multiclass_test_metrices['test_auc'].append(test_auc*100)
            multiclass_test_metrices['test_acc'].append(test_acc*100)
            multiclass_test_metrices['test_p'].append(test_p*100)
            multiclass_test_metrices['test_r'].append(test_r*100)

        



    if args.num_classes == 2:
        temp_path = os.path.join(os.getcwd(), 'Metrices', 'Binary', args.expt_name)
        os.makedirs(temp_path, exist_ok=True)

        # val metrices:- 
        val_df = pd.DataFrame(binary_val_metrices)
        val_directory_path = os.path.join(temp_path, 'val_metrices.csv')
        val_df.to_csv(val_directory_path, index=False)

        # calculate avg for val metrices:-
        avg_val_metrices['val_auc'].append(val_df['val_auc'].mean())
        avg_val_metrices['val_acc'].append(val_df['val_acc'].mean())
        avg_val_metrices['val_p'].append(val_df['val_p'].mean())
        avg_val_metrices['val_r'].append(val_df['val_r'].mean())
        avg_val_metrices['val_Sens'].append(val_df['val_Sens'].mean())
        avg_val_metrices['val_Spec'].append(val_df['val_Spec'].mean())

        val_avg_df = pd.DataFrame(avg_val_metrices)
        val_directory_path = os.path.join(temp_path, 'avg_val_metrices.csv')
        val_avg_df.to_csv(val_directory_path, index=False)


        # calculate variance for val metrices:-
        variance_val_metrices['val_auc'].append(val_df['val_auc'].var())
        variance_val_metrices['val_acc'].append(val_df['val_acc'].var())
        variance_val_metrices['val_p'].append(val_df['val_p'].var())
        variance_val_metrices['val_r'].append(val_df['val_r'].var())
        variance_val_metrices['val_Sens'].append(val_df['val_Sens'].var())
        variance_val_metrices['val_Spec'].append(val_df['val_Spec'].var())

        val_var_df = pd.DataFrame(variance_val_metrices)
        val_directory_path = os.path.join(temp_path, 'var_val_metrices.csv')
        val_var_df.to_csv(val_directory_path, index=False)


        # test metrices:- 
        test_df = pd.DataFrame(binary_test_metrices)
        test_directory_path = os.path.join(temp_path, 'test_metrices.csv')
        test_df.to_csv(test_directory_path, index=False)

        # calculate avg for test metrices:-
        avg_test_metrices['test_auc'].append(test_df['test_auc'].mean())
        avg_test_metrices['test_acc'].append(test_df['test_acc'].mean())
        avg_test_metrices['test_p'].append(test_df['test_p'].mean())
        avg_test_metrices['test_r'].append(test_df['test_r'].mean())
        avg_test_metrices['test_Sens'].append(test_df['test_Sens'].mean())
        avg_test_metrices['test_Spec'].append(test_df['test_Spec'].mean())

        test_avg_df = pd.DataFrame(avg_test_metrices)
        val_directory_path = os.path.join(temp_path, 'avg_test_metrices.csv')
        test_avg_df.to_csv(val_directory_path, index=False)



        # calculate variance for test metrices:-
        variance_test_metrices['test_auc'].append(test_df['test_auc'].var())
        variance_test_metrices['test_acc'].append(test_df['test_acc'].var())
        variance_test_metrices['test_p'].append(test_df['test_p'].var())
        variance_test_metrices['test_r'].append(test_df['test_r'].var())
        variance_test_metrices['test_Sens'].append(test_df['test_Sens'].var())
        variance_test_metrices['test_Spec'].append(test_df['test_Spec'].var())

        test_var_df = pd.DataFrame(variance_test_metrices)
        val_directory_path = os.path.join(temp_path, 'var_test_metrices.csv')
        test_var_df.to_csv(val_directory_path, index=False)



    elif args.num_classes == 3:
        temp_path = os.path.join(os.getcwd(), 'Metrices', '3way', args.expt_name)
        os.makedirs(temp_path, exist_ok=True)

        val_df = pd.DataFrame(multiclass_val_metrices)
        val_directory_path = os.path.join(temp_path, 'val_metrices.csv')
        val_df.to_csv(val_directory_path, index=False)

        test_df = pd.DataFrame(multiclass_test_metrices)
        test_directory_path = os.path.join(temp_path, 'test_metrices.csv')
        test_df.to_csv(test_directory_path, index=False)


    elif args.num_classes == 6:
        temp_path = os.path.join(os.getcwd(), 'Metrices', '6way', args.expt_name)
        os.makedirs(temp_path, exist_ok=True)

        val_df = pd.DataFrame(multiclass_val_metrices)
        val_directory_path = os.path.join(temp_path, 'val_metrices.csv')
        val_df.to_csv(val_directory_path, index=False)

        test_df = pd.DataFrame(multiclass_test_metrices)
        test_directory_path = os.path.join(temp_path, 'test_metrices.csv')
        test_df.to_csv(test_directory_path, index=False)

        



if __name__ == '__main__':
    main()
    