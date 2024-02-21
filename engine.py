import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import CustomMultiHeadLoss, CombinedMultiHeadLoss, MetaMultiHeadLoss, MetaCombinedMultiHeadLoss, entropy, stochastic_dropout

def train(data_loader, model, model_name, optimizer, Loss, multi_head_loss, epsilon, no_of_heads):

    # put the model in train mode
    model.train()

    for data in data_loader:
        feature = data[0].float()
        label = data[1]

        # Check if CUDA is available
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        
        # Move the tensor to the selected device (CPU or CUDA)
        feature = feature.to(device)
        label = label.to(device)

        # do the forward pass through the model
        if model_name == 'CustomResNet' or model_name == 'Custom_ViT':

            outputs = model(feature)

            # calculate loss
            if Loss == 'cross_entropy':
                criterion = nn.CrossEntropyLoss()
                loss = criterion(outputs, label)
            
            elif Loss == 'binary_cross_entropy':
                criterion = nn.BCEWithLogitsLoss()
                loss = criterion(outputs, label)

            # zero grad the optimizer
            optimizer.zero_grad()

            # calculate the gradient
            loss.backward()

            # update the weights
            optimizer.step()


        else:

            _, outputs = model(feature)
            #print(outputs)
            outputs = stochastic_dropout(outputs, 0.1)
            #print(outputs)

             # calculate loss
            if multi_head_loss == 'CustomMultiHeadLoss':
                if Loss == 'cross_entropy':
                    meta_loss = CustomMultiHeadLoss(nn.CrossEntropyLoss(), no_of_heads)
                    losses = meta_loss(outputs, label)
                else:
                    meta_loss = CustomMultiHeadLoss(nn.BCEWithLogitsLoss(), no_of_heads)
                    losses = meta_loss(outputs, label)

                # zero grad the optimizer
                optimizer.zero_grad()

                 # Calculate the gradients for each head's loss and perform backpropagation
                for loss in losses:
                    loss.backward(retain_graph=True)

                # Update the weights after accumulating gradients from all heads
                optimizer.step()

                

            elif multi_head_loss == 'CombinedMultiHeadLoss':
                if Loss == 'cross_entropy':
                    meta_loss = CombinedMultiHeadLoss(nn.CrossEntropyLoss(), no_of_heads)
                    losses = meta_loss(outputs, label)
                else:
                    meta_loss = CombinedMultiHeadLoss(nn.BCEWithLogitsLoss(), no_of_heads)
                    losses = meta_loss(outputs, label)

                # zero grad the optimizer
                optimizer.zero_grad()

                # calculate the gradient
                losses.backward()

                # update the weights
                optimizer.step()
   


            elif multi_head_loss == 'MetaMultiHeadLoss':
                if Loss == 'cross_entropy':
                    meta_loss = MetaMultiHeadLoss(nn.CrossEntropyLoss(), no_of_heads, epsilon)
                    losses = meta_loss(outputs, label)
                else:
                    meta_loss = MetaMultiHeadLoss(nn.BCEWithLogitsLoss(), no_of_heads, epsilon)
                    losses = meta_loss(outputs, label)

                # zero grad the optimizer
                optimizer.zero_grad()

                # Calculate the gradients for each head's loss and perform backpropagation
                for loss in losses:
                    loss.backward(retain_graph=True)

                # Update the weights after accumulating gradients from all heads
                optimizer.step()



            elif multi_head_loss == 'MetaCombinedMultiHeadLoss':
                if Loss == 'cross_entropy':
                    meta_loss = MetaCombinedMultiHeadLoss(nn.CrossEntropyLoss(), no_of_heads, epsilon)
                    losses = meta_loss(outputs, label)
                else:
                    meta_loss = MetaCombinedMultiHeadLoss(nn.BCEWithLogitsLoss(), no_of_heads, epsilon)
                    losses = meta_loss(outputs, label)

                # zero grad the optimizer
                optimizer.zero_grad()

                # calculate the gradient
                losses.backward()

                # update the weights
                optimizer.step()





def val(data_loader, model, model_name, Loss, no_of_heads, combine_results):

    val_loss_list = []
    final_output = []
    final_label = []

    # put model in evaluation mode
    model.eval()

    with torch.no_grad():
        for data in data_loader:
            feature = data[0].float()
            label = data[1]

            # Check if CUDA is available
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

            # Move the tensor to the selected device (CPU or CUDA)
            feature = feature.to(device)
            label = label.to(device)

            # do the forward pass through the model
            if model_name == 'CustomResNet' or model_name == 'Custom_ViT':
                outputs = model(feature)

                # calculate val loss
                if Loss == 'cross_entropy':
                    criterion = nn.CrossEntropyLoss()
                    temp_val_loss = criterion(outputs, label)
                    val_loss_list.append(temp_val_loss)
                    softmax_values = F.softmax(outputs, dim=1)
                    outputs = torch.argmax(softmax_values, dim=1).int()
                    
                elif Loss == 'binary_cross_entropy':
                    criterion = nn.BCEWithLogitsLoss()
                    val_loss = criterion(outputs, label)
                    outputs = (outputs > 0.5).int()


                OUTPUTS = outputs.detach().cpu().tolist()


            else:
               
                outputs, Op = model(feature)
                #softmax_values = F.softmax(outputs, dim=1)
                
                # calculate val loss 
                if Loss == 'cross_entropy':
                    meta_loss = CombinedMultiHeadLoss(nn.CrossEntropyLoss(), no_of_heads)
                    temp_val_loss = meta_loss(Op, label).detach().cpu().tolist() / no_of_heads
                    val_loss_list.append(temp_val_loss)



                OUTPUTS = []
                if combine_results == 'entropy':
                    for idx in range(len(outputs)):
                        #softmax_values = F.softmax(outputs[idx], dim=1)
                        if entropy(softmax_values[idx][:, 0]) < entropy(softmax_values[idx][:, 1]):
                            OUTPUTS.append(int(0))
                        else:
                            OUTPUTS.append(int(1))

                    
                elif combine_results == 'avg':
                    for idx in range(len(outputs)):
                        softmax_values = F.softmax(outputs[idx], dim=1)
                        #print(softmax_values)
                        #print(f'avg0:-{sum(softmax_values[:, 0])/len(softmax_values[:, 0])}')
                        #print(f'avg1:-{sum(softmax_values[:, 1])/len(softmax_values[:, 1])}')
                        if sum(softmax_values[:, 0])/len(softmax_values[:, 0]) > sum(softmax_values[:, 1])/len(softmax_values[:, 1]):
                            OUTPUTS.append(int(0))
                        else:
                            OUTPUTS.append(int(1))


            final_output.extend(OUTPUTS)
            final_label.extend(label.detach().cpu().tolist())

    return final_output, final_label, sum(val_loss_list)/len(val_loss_list)
        




def test(data_loader, model, model_name, combine_results):

    final_output = []
    final_label = []


    # put model in evaluation mode
    model.eval()
    with torch.no_grad():
        for data in data_loader:
            feature = data[0].float()
            label = data[1]

            # Check if CUDA is available
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

            # Move the tensor to the selected device (CPU or CUDA)
            feature = feature.to(device)
            label = label.to(device)


            # do the forward pass through the model
            if model_name == 'CustomResNet' or model_name == 'Custom_ViT':
                outputs = model(feature)
                softmax_values = F.softmax(outputs, dim=1)
                outputs = torch.argmax(softmax_values, dim=1).int()
                OUTPUTS = outputs.detach().cpu().tolist()
                


            else:
                outputs, _ = model(feature)

                OUTPUTS = []
                if combine_results == 'entropy':
                    for idx in range(len(outputs)):
                        softmax_values = F.softmax(outputs[idx], dim=1)
                        if entropy(softmax_values[:, 0]) < entropy(softmax_values[:, 1]):
                            OUTPUTS.append(int(0))
                        else:
                            OUTPUTS.append(int(1))

                    
                elif combine_results == 'avg':
                    for idx in range(len(outputs)):
                        softmax_values = F.softmax(outputs[idx], dim=1)
                        if sum(softmax_values[:, 0])/len(softmax_values[:, 0]) > sum(softmax_values[:, 1])/len(softmax_values[:, 1]):
                            OUTPUTS.append(int(0))
                        else:
                            OUTPUTS.append(int(1))


            final_output.extend(OUTPUTS)
            final_label.extend(label.detach().cpu().tolist())

    return final_output, final_label
    

        










        



        




                    

                


        

                

                
                












                


                    



            




        


                






                
            



       
        

        
