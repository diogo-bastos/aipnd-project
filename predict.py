#imports
import torch
import torch.nn.functional as F
from functions import *
from utility import *
from checkpoint import *

def main():

    print("predict.py started...")
    
    #configure url parsing
    parser = configure_arguments_parsing_for_predict()

    #parse values from the program's arguments
    use_gpu, save_directory, topk, image_path = parse_constants_from_program_input_predict(parser)

    #load model from checkpoint
    model = load_checkpoint(save_directory)
    
    predictions, classes = predict(image_path, model, save_directory, topk, use_gpu);
    class_names = convert_indexes_to_names(classes)

    index = 0
    for prediction in predictions:
        print(f'the {index+1}nth prediction probability is: {prediction} and the class name is {class_names[index]}')
        index += 1

    print("predict.py ended!")

if __name__ == "__main__":
        main()