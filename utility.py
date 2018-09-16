import argparse
import torch
import json
from torchvision import datasets

def configure_arguments_parsing_for_train():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help="a data dir is required")
    parser.add_argument('--learning_rate', action='store',
                        dest='learning_rate',
                        default= 0.001,
                        type=float,
                        help='define the learning rate')
    parser.add_argument('--epochs', action='store',
                        dest='epochs',
                        default=5,
                        type=int,
                        help='define the number of epochs that the model will be trained for')
    parser.add_argument('--save_dir', action='store',
                        dest='save_directory',
                        default='checkpoint.pth',
                        help='define the directory in which files will be saved')
    parser.add_argument('--arch', action='store',
                        dest='model_architecture',
                        default='',
                        help='define the model\'s architecture')
    parser.add_argument('--hidden_units', action='store',
                        dest='hidden_units',
                        default=1000,
                        type=int,
                        help='define the number of hidden units in the model\'s architecture')
    parser.add_argument('-gpu', action='store_true',
                        default=False,
                        dest='use_gpu',
                        help='Use gpu for the model\'s training')
    
    return parser

def configure_arguments_parsing_for_predict():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', action='store',
                        default='',
                        help='Define the path of the image whose class you want to predict')
    parser.add_argument('save_directory', action='store',
                        default='checkpoint.pth',
                        help='Define the directory in which to load the checkpoint file from')    
    parser.add_argument('-gpu', action='store_true',
                        default=False,
                        dest='use_gpu',
                        help='Use gpu for the model\'s training')
    parser.add_argument('--topk', action='store',
                        default=5,
                        dest='topk',
                        type=int,
                        help='Define the number of classes you want to know the predictions for')
    
    return parser


def parse_constants_from_program_input_train(parser):

    results = parser.parse_args()
    data_dir = results.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    number_of_epochs = results.epochs
    learning_rate = results.learning_rate
    use_gpu = results.use_gpu
    save_directory = results.save_directory
    architecture = results.model_architecture
    hidden_units = results.hidden_units
    return train_dir, valid_dir, test_dir, number_of_epochs, learning_rate, use_gpu, save_directory, architecture, hidden_units

def parse_constants_from_program_input_predict(parser):

    results = parser.parse_args()
    use_gpu = results.use_gpu
    topk = results.topk
    image_path = results.image_path
    save_directory = results.save_directory
    return use_gpu, save_directory, topk, image_path

def load_images_from_folders(transformations, train_dir, valid_dir, test_dir):

    images_data = {}
    images_data['train'] = datasets.ImageFolder(train_dir, transform=transformations['train'])
    images_data['validation'] = datasets.ImageFolder(valid_dir, transform=transformations['test_validation'])
    images_data['test'] = datasets.ImageFolder(test_dir, transform=transformations['test_validation'])
    return images_data

def define_data_loaders(images_data):

    data_loaders = {}
    data_loaders['train'] = torch.utils.data.DataLoader(images_data['train'], batch_size=64, shuffle=True)
    data_loaders['validation'] = torch.utils.data.DataLoader(images_data['validation'], batch_size=32)
    data_loaders['test'] = torch.utils.data.DataLoader(images_data['test'], batch_size=32)
    return data_loaders

#get the class names in a human format
def convert_indexes_to_names(classes):

    class_names = []
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    for output_class in classes:
        class_names.append(cat_to_name[output_class])

    return class_names