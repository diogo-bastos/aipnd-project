import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
from image_processing import *

def create_model(arch, hidden_units):
    
    if arch != 'vgg' and arch != 'densenet' and arch != 'alexnet':
        print('No valid architecture was selected. The possible values are: \'vgg\', \'densenet\' and \'alexnet\'. Alexnet will be used as a default.')
        arch = 'alexnet'
        
    if arch == 'vgg':
        # Downloading the static vgg model
        model = models.vgg19(pretrained=True)
        in_features = 25088
    elif arch == 'densenet':
        # Downloading the static densenet model
        model = models.densenet201(pretrained=True)
        in_features = 1920
    else:
        # Downloading the static alexnet model
        model = models.alexnet(pretrained=True)
        in_features = 9216
        
    print(f'The architecture {arch} will be used. The number of units in the hidden layer will be {hidden_units}.')
    #Freezing the model's parameters so we don't backpropagate through them
    for param in model.parameters():
        param.requires_grad = False

    #Redefine the model's classifier to our feedforward network
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(in_features, hidden_units)),
                              ('relu1', nn.ReLU()),
                              ('drop1', nn.Dropout(p=0.2)),
                              ('fc2', nn.Linear(hidden_units, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))
    model.classifier = classifier
    return model

#Define function for the network training
def do_deep_learning(model, trainloader, validationloader, criterion, optimizer, epochs = 10, print_every = 50, use_gpu = True):
    epochs = epochs
    print_every = print_every
    steps = 0

    if use_gpu:
        device=torch.device('cuda')
    else:
        device=torch.device('cpu')

    model.to(device)
    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                accuracy = check_accuracy(validationloader, model, use_gpu)
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Loss: {:.4f}... ".format(running_loss/print_every),
                     "Validation accuracy: {:.4f}".format(accuracy))

                running_loss = 0

#Define function for an accuracy check
def check_accuracy(loader, model, use_gpu):   
    if use_gpu:
        device=torch.device('cuda')
    else:
        device=torch.device('cpu')
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            images, labels = data
            labels = labels.to(device)
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy


#Define the transformations for the image data
def define_transform_operations():

    transformations = {}
    transformations['train'] = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])

    transformations['test_validation'] = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])
   
    return transformations



# code to predict the class from an image file
def predict(image_path, model, checkpoint_directory, topk=5, use_gpu=True):
    
    #convert model to cuda for gpu usage
    if use_gpu:
        device=torch.device('cuda')
    else:
        device=torch.device('cpu')
        
    print(f'Using device {device} for prediction...')
    model.to(device)
    model.eval()
    
    #get image and process it according to the train transforms
    img = Image.open(image_path)
    np_image = process_image(img)
    
    #add a dummy dimension to the image tensor
    np_image = np.expand_dims(np_image, axis=0)
    
    #generate the prediction
    with torch.no_grad():
        output = model.forward(torch.from_numpy(np_image).float().to(device)) 
    ps = torch.exp(output)
    
    #get the topk values
    predicted = ps.topk(topk)
    
    #convert from folder index to folder name and build the return variables
    classes = []
    predictions = []
    index = 0
    for class_idx in predicted[1][0]:
        index += 1
        for class_folder_name, idx in model.class_to_idx.items():    
            if idx == class_idx:
                classes.append(class_folder_name)
                predictions.append(predicted[0][0][index-1].item())
    return predictions, classes
    
    model = torch.load(checkpoint_directory)
    return model