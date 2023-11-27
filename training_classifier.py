import torch
from data_loader import get_data_blobs
import numpy as np

import torchvision.transforms as transforms
from tqdm import tqdm
import torch.optim as optim

import json

from classification_model import ClassificationModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.cuda.empty_cache()

NUMBER_OF_LABELS = 3


def loss_fun(preds, target):
    return torch.nn.CrossEntropyLoss()(preds, target)


def output_activate(pred_logits):
    return torch.nn.Softmax(dim=1)(pred_logits)

def accuracy(preds, target):
    return (preds.argmax(dim=1) == target).cpu().float().mean().item()



def train(model, opt, epochs, train_loader, valid_loader, test_loader):

    results = {
        "train_acc" : [],
        "train_loss" : [],
        "valid_acc" : [],
        "valid_loss" : [],
        "test_acc" : [],
        "test_loss" : []
    }
    
    for epoch in range(epochs):
        print('* Epoch %d/%d' % (epoch+1, epochs))
        
        train_acc = []
        train_loss = []

        model.train()  # train mode
        for i, (X_batch, Y_target) in enumerate(tqdm(train_loader, total=len(train_loader))):
            X_batch = X_batch.to(device)
            Y_target = Y_target.to(device)

            # set parameter gradients to zero
            opt.zero_grad()

            # forward
            Y_pred_logits = model(X_batch)

            loss = loss_fun(Y_pred_logits, Y_target) 
            loss.backward()  # backward-pass
            opt.step()  # update weights

            # calculate metrics             
            train_loss.append(loss.item())
            train_acc.append(accuracy(output_activate(Y_pred_logits), Y_target))

        # Add to results
        results["train_acc"].append(np.mean(train_acc))
        results["train_loss"].append(np.mean(train_loss))

        print("\navg train acc : ",results["train_acc"][-1]," - loss : ", results["train_loss"][-1])
        

        # Validation 
        valid_loss = []
        valid_acc = []
        model.eval()
        for data, target in valid_loader:
            data, target = data.to(device), target.to(device)
            with torch.no_grad():
                preds_logits = model(data)
            valid_loss.append(loss_fun(preds_logits, target).item())
            valid_acc.append(accuracy(output_activate(preds_logits), target))

        # Add to results
        results["valid_acc"].append(np.mean(valid_acc))
        results["valid_loss"].append(np.mean(valid_loss))

        
        print("avg valid acc : ",results["valid_acc"][-1]," - loss : ", results["valid_loss"][-1],"\n")

        


    # Testing
    test_loss = []
    test_acc = []
    model.eval()
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        with torch.no_grad():
            preds_logits = model(data)
        test_loss.append(loss_fun(preds_logits, target).item())
        test_acc.append(accuracy(output_activate(preds_logits), target))


    # Add to results
    results["test_acc"].append(np.mean(test_acc))
    results["test_loss"].append(np.mean(test_loss))
    

    print("\n\navg test acc : ",results["test_acc"][-1]," - loss : ", results["test_loss"][-1])
    

    return results, model





# ClassificationModel()
batch_size = 32            # Number of images in a batch 
target_shape = (128,128)   # Shape of patch
epochs = 14               # Number of iterations      
validation_ratio=0.15
testing_ratio=0.10


# Transformations 
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(target_shape),
    transforms.RandomRotation(179)
])
    
general_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(target_shape),
    transforms.RandomRotation(179)
])

model = ClassificationModel().to(device)


train_loader, valid_loader, test_loader = get_data_blobs(batch_size, validation_ratio, testing_ratio, train_transform=train_transform, general_transform=general_transform)

optimizer = optim.Adam(model.parameters(), lr = 1e-6)
results, model = train(model, optimizer, epochs=epochs, train_loader=train_loader, valid_loader=valid_loader, test_loader=test_loader)


# Save model weights and results
torch.save(model.state_dict(), "./Data/model_loadings.pt")

with open("classifier_training_results.json","w") as fp:
    json.dump(results, fp, indent=2)







