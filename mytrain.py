import myvalidation
import wandb
import os
import torch
import numpy as np
from utils import label_accuracy_score, add_hist

def save_model(model, saved_dir, file_name='fcn_resnet101_best_model(pretrained).pt'):
    check_point = {'net': model.state_dict()}

    try: 
        if not os.path.exists(saved_dir):
            os.makedirs(saved_dir)
    except OSError: 
       if not os.path.isdir(saved_dir): 
           raise

    output_path = os.path.join(saved_dir, file_name)
    torch.save(model, output_path)

def train(num_epochs, model, data_loader, val_loader, criterion, optimizer, saved_dir, val_every, device, category_names, saved_modelname):
    print(f'Start training..')
    n_class = 11
    best_loss = 9999999
    best_mIoU = -1

    for epoch in range(num_epochs):
        model.train()

        total_loss = 0
        cnt = 0

        hist = np.zeros((n_class, n_class))
        for step, (images, masks, _) in enumerate(data_loader):
            images = torch.stack(images)       
            masks = torch.stack(masks).long() 
            
            # gpu 연산을 위해 device 할당
            images, masks = images.to(device), masks.to(device)
            
            # device 할당
            model = model.to(device)
            
            # inference
            outputs = model(images)['out']
            
            # loss 계산 (cross entropy loss)
            loss = criterion(outputs, masks)

            total_loss += loss
            cnt += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            masks = masks.detach().cpu().numpy()
            
            hist = add_hist(hist, masks, outputs, n_class=n_class)
            acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)
            
            # step 주기에 따른 loss 출력
            if (step + 1) % 25 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{len(data_loader)}], \
                        Loss: {round(loss.item(),4)}, mIoU: {round(mIoU,4)}')
        
        _, _, train_mIoU, _, _ = label_accuracy_score(hist)
        train_avg_loss = total_loss / cnt

        train_avg_loss = round(train_avg_loss.item(), 4)
        train_mIoU = round(train_mIoU, 4)
        
        # validation 주기에 따른 loss 출력 및 best model 저장
        if (epoch + 1) % val_every == 0:
            val_avg_loss, val_avg_mIoU = myvalidation.validation(epoch + 1, model, val_loader, criterion, device, category_names)
            val_avg_loss = round(val_avg_loss.item(), 4)
            val_avg_mIoU = round(val_avg_mIoU, 4)

            wandb.log({f'Train_Loss': train_avg_loss, 'Train_mIoU': train_mIoU, 'Eval_Loss': val_avg_loss, 'Eval_mIoU': val_avg_mIoU})

            if val_avg_mIoU > best_mIoU:
                print(f"Best performance at epoch: {epoch + 1}")
                print(f"Save model in {saved_dir}")
                best_mIoU = val_avg_mIoU
                save_model(model, saved_dir, saved_modelname)