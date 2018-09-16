import numpy as np

# Process a PIL image and return a numpy array
def process_image(image):
    #Thumbnail with the smallest size side as 256
    if image.size[0]>image.size[1]:
        image.thumbnail((image.size[0], 256))
    else:
        image.thumbnail((256, image.size[1]))
    width, height = image.size
    
    #Crop as a 224x244 image
    width_padding = (width-224)/2
    height_padding = (height-224)/2
    area = int(width_padding), int(height_padding), int(width-width_padding), int(height-height_padding)
    image = image.crop(area)
    np_image = np.asarray(image)
    np_image = np_image / 255
    
    #Normalize the color channels
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean)/std
    
    #Transpose the color channel to the first dimension
    np_image = np_image.transpose((2,0,1))
    return np_image	
    
def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
   
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image*255)
    
    return ax
