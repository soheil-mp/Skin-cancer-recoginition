# Importing the libraries
import numpy as np
import scipy
import keras
from keras.layers import Input
from keras.preprocessing import image    
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Flatten, BatchNormalization, Activation, Dropout
import matplotlib.pyplot as plt
#%matplotlib qt

# Uploading an image to a tensor
def path_to_tensor(img_path):
    """
    Getting a tensor from a given path.
    """
    # Loading the image
    img = image.load_img(img_path, target_size=(512, 512))
    # Converting the image to numpy array
    x = image.img_to_array(img)   
    # convert 3D tensor to 4D tensor with shape (1, 512, 512, 3)
    x =  np.expand_dims(x, axis=0)
    return x

# Single input for multiple models
model_input = Input(shape=(512, 512, 3))

def mobilenet_architecture():
    """
    Pre-build architecture of mobilenet for our dataset.
    """
    # Imprting the model
    from keras.applications.mobilenet import MobileNet

    # Pre-build model
    base_model = MobileNet(include_top = False, weights = None, input_tensor = model_input)

    # Adding output layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    output = Dense(units = 2, activation = 'softmax')(x)

    # Creating the whole model
    mobilenet_model = Model(base_model.input, output)
    
    # Getting the summary of architecture
    #mobilenet_model.summary()
    
    # Compiling the model
    mobilenet_model.compile(optimizer = keras.optimizers.Adam(lr = 0.001), 
                            loss = 'categorical_crossentropy', 
                            metrics = ['accuracy'])

    return mobilenet_model

def inception_architecture():
    """
    Pre-build architecture of inception for our dataset.
    """
    # Imprting the model 
    from keras.applications.inception_v3 import InceptionV3

    # Pre-build model
    base_model = InceptionV3(include_top = False, weights = None, input_tensor = model_input)

    # Adding output layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    output = Dense(units = 2, activation = 'softmax')(x)

    # Creating the whole model
    inception_model = Model(base_model.input, output)
    
    # Summary of the model
    #inception_model.summary()
    
    # Compiling the model
    inception_model.compile(optimizer = keras.optimizers.Adam(lr = 0.001), 
                            loss = 'categorical_crossentropy', 
                            metrics = ['accuracy'])
    
    return inception_model

def xception_architecture():
    """
    Pre-build architecture of inception for our dataset.
    """
    # Imprting the model
    from keras.applications.xception import Xception

    # Pre-build model
    base_model = Xception(include_top = False, weights = None, input_tensor = model_input)

    # Adding output layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    output = Dense(units = 2, activation = 'softmax')(x)

    # Creating the whole model
    xception_model = Model(base_model.input, output)

    # Summary of the model
    #xception_model.summary()
    
    # Compiling the model
    xception_model.compile(optimizer = keras.optimizers.Adam(lr = 0.001), 
                           loss = 'categorical_crossentropy', 
                           metrics = ['accuracy'])

    return xception_model


# Model 1
mobilenet_model = mobilenet_architecture()
mobilenet_model.load_weights("./Saved models/weights.best.mobilenet_epoch_2.hdf5")
print("Model 1 has been loaded!")


# Model 2
inception_model = inception_architecture()
inception_model.load_weights("./Saved models/weights.best.inception.hdf5")
print("Model 2 has been loaded!")


# Model 3
xception_model = xception_architecture()
xception_model.load_weights("./Saved models/weights.best.xception.hdf5")
print("Model 3 has been loaded!")


# Appending all models
models = [mobilenet_model, inception_model, xception_model]


def ensemble(models, model_input):
    """
    Ensembling multiple methods.
    """
    outputs = [model.outputs[0] for model in models]
    
    y = keras.layers.Average()(outputs)
    
    model = Model(model_input, y, name='ensemble')
    
    return model

# Getting ensemble model
ensemble_model = ensemble(models, model_input)
print("Ensemble model has been build sucessfully!!")

def getting_two_layer_weights(path_model_weight):
    # The model

    # Imprting the model
    from keras.applications.mobilenet import MobileNet

    # Pre-build model
    base_model = MobileNet(include_top = False, weights = None, input_shape = (512, 512, 3))

    # Adding output layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    output = Dense(units = 2, activation = 'softmax')(x)

    # Creating the whole model
    model = Model(base_model.input, output)
    #model.summary()

    # Compiling the model
    model.compile(optimizer = keras.optimizers.Adam(lr = 0.001), 
                  loss = 'categorical_crossentropy', 
                  metrics = ['accuracy'])
    
    # loading the weights
    model.load_weights(path_model_weight)
    
    # Getting the AMP layer weight
    all_amp_layer_weights = model.layers[-1].get_weights()[0]
    
    # Extracting the wanted output
    mobilenet_model = Model(inputs = model.input, outputs = (model.layers[-3].output, model.layers[-1].output))
    
    return mobilenet_model, all_amp_layer_weights


def mobilenet_CAM(img_path, model, all_amp_layer_weights):
    # Getting filtered images from last convolutional layer + model prediction output
    last_conv_output, predictions = model.predict(path_to_tensor(img_path)) # last_conv_output.shape = (1, 16, 16, 1024)
    
    # Converting the dimension of last convolutional layer to 16 x 16 x 1024     
    last_conv_output = np.squeeze(last_conv_output)
    
    # Model's prediction
    predicted_class = np.argmax(predictions)
    
    # Bilinear upsampling (resize each image to size of original image)
    mat_for_mult = scipy.ndimage.zoom(last_conv_output, (32, 32, 1), order = 1)  # dim from (16, 16, 1024) to (512, 512, 1024)
    
    # Getting the AMP layer weights
    amp_layer_weights = all_amp_layer_weights[:, predicted_class] # dim: (1024,)    
    
    # CAM for object class that is predicted to be in the image
    final_output = np.dot(mat_for_mult, amp_layer_weights) # dim: 512 x 512

    # Return class activation map (CAM)
    return final_output, predicted_class


def plot_CAM(img_path, ax, model, all_amp_layer_weights):
    # Loading the image / resizing to 512x512 / Converting BGR to RGB
    #im = cv2.resize(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB), (512, 512))
    im = path_to_tensor(img_path).astype("float32")/255.
    
    # Plotting the image
    ax.imshow(im.squeeze(), vmin=0, vmax=255)
    
    # Getting the class activation map
    CAM, pred = mobilenet_CAM(img_path, model, all_amp_layer_weights)
    
    CAM = (CAM - CAM.min()) / (CAM.max() - CAM.min())
    
    # Plotting the class activation map
    ax.imshow(CAM, cmap = "jet", alpha = 0.5, interpolation='nearest', vmin=0, vmax=1)
    
    
# Test test
if __name__ == "__main__":
    
    # Getting the image path
    img_path = "./Dataset/melanoma/ISIC_0027060_90_angle.jpg"
    
    # Getting the iamge tensor
    image_to_predict = path_to_tensor(img_path).astype('float32')/255
    image_to_plot = path_to_tensor(img_path)
    
    print("Image uploaded sucessfully!..(1/5)")
    print("Image preprocessed sucessfully!..(2/5)")
      
    # model's weight for localization
    path_to_model_weight = "./Saved models/weights.best.mobilenet.hdf5"
    
    # Getting the weights of last activation and last dense for localization
    mobilenet_model, all_amp_layer_weights = getting_two_layer_weights(path_to_model_weight)
    
    # Calculating the localization    
    final_output, predicted_class = mobilenet_CAM(img_path, mobilenet_model, all_amp_layer_weights)
    
    print("Image localization calculated sucessfully!..(3/5)")
    
    # Predicting the image
    prediction = ensemble_model.predict(image_to_predict)
    prediction_final = "Melanoma: " + str(np.round(prediction[0][0]*100, decimals = 2)) + "%" + \
                       " | Other illness: " + str(np.round(prediction[0][1]*100, decimals = 2)) + "%"

    # Canvas initialization
    plt.figure(figsize = (10, 10))
    
    # First image
    plt.subplot(121)
    plt.imshow(image_to_predict.squeeze())
    plt.text(0, 650, prediction_final, fontsize = 24)

    print("Image plotted sucessfully!..(4/5)")
    
    # Second image
    plt.subplot(122)
    
    # Plotting the image
    plot_CAM(img_path, plt, mobilenet_model, all_amp_layer_weights)
    
    print("Localized image plotted sucessfully!..(5/5)")

    plt.show()

