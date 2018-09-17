#imports
import torch
from torch import optim
from functions import *
from utility import *
from checkpoint import *
    
def main():
    
    print("train.py started...")
    
    #configure url parsing
    parser = configure_arguments_parsing_for_train()
    train_dir = ''
    valid_dir = ''
    test_dir = ''
    number_of_epochs = 0
    learning_rate = 0.001
    use_gpu = False
    save_directory = '.'
    architecture = 'vgc11'
    hidden_units = 1000
    #parse values from the program's arguments
    train_dir, valid_dir, test_dir, number_of_epochs, learning_rate, use_gpu, save_directory, architecture, hidden_units = parse_constants_from_program_input_train(parser)
    
    #Define your transforms for the training, validation, and testing sets
    transformations = define_transform_operations()

    #Load the datasets with ImageFolder and apply the transforms
    print("loading images...")
    images_data = load_images_from_folders(transformations, train_dir, valid_dir, test_dir)

    #Using the image datasets and the trainforms, define the dataloaders
    data_loaders = define_data_loaders(images_data)
    
    #create the model according to the desired hyper parameters
    print("creating the model...")
    model = create_model(architecture, hidden_units)
    
    #define the criterion and the optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    #train the model
    print("training started...")
    do_deep_learning(model, data_loaders['train'], data_loaders['validation'], criterion, optimizer, number_of_epochs, 100, use_gpu)
    
    #check the accuracy of the validation set and the testing set
    print("checking the accuracy of the validation and training sets...")

    accuracy, loss = check_accuracy(data_loaders['test'], model, criterion, use_gpu)
    print(f'Accuracy of the network on the test images: {accuracy:.4f} %. Total loss was: {loss:.4f}')
    
    #save the model to a checkpoint
    print("saving the model to a checkpoint file...")
    save_checkpoint(model, images_data['train'].class_to_idx, save_directory, architecture, hidden_units)
    
    print("train.py finished!")
    
if __name__ == "__main__":
        main()