import os
from keras.models import model_from_json
from keras.layers import Input, LeakyReLU, Add, UpSampling3D, Activation, SpatialDropout3D, Conv3D
from keras.engine import Model
from keras.optimizers import Adam

from unet3d.metrics import weighted_dice_coefficient_loss, tversky_loss, minh_dice_coef_loss, minh_dice_coef_metric

from unet3d.training import load_old_model
from keras.utils import multi_gpu_model

def load_model_multi_gpu(model_file):

    print(">> load old model")
    model = load_old_model(model_file)

    from unet3d.utils.path_utils import get_filename
    filename = get_filename(model_file)

    model_json_path = filename.replace(".h5", ".json")
    weights_path = filename
    if os.path.exists(model_json_path):
        print(">> remove old json")
        os.remove(model_json_path)
    if os.path.exists(weights_path):
        print(">> remove old weights")
        os.remove(weights_path)

    # ------------ save the template model rather than the gpu_mode ----------------
    # serialize model to JSON
    print(">> save architecture to disk")
    model_json = model.to_json()
    with open(model_json_path, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    print(">> save weights to disk")
    model.save_weights(weights_path)

    # -------------- load the saved model --------------
    from keras.models import model_from_json

    # load json and create model
    print(">> load architecture from disk")
    json_file = open(model_json_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    print(">> load model from disk")
    loaded_model.load_weights(weights_path)

    for i_layer in range(len(loaded_model.layers)):
        layer_name = loaded_model.layers[i_layer].name
        if type(loaded_model.layers[i_layer]) is Model:
            model = loaded_model.layers[i_layer]

    print(">> remove temp weights")
    os.remove(weights_path)
    print(">> remove temp json")
    os.remove(model_json_path)
    return model


def generate_model(model_file, loss_function="weighted", metrics=minh_dice_coef_metric,
                   initial_learning_rate=0.001):

    model = load_model_multi_gpu(model_file)

    try:
        model = multi_gpu_model(model, gpus=2)
        print('!! train on multi gpus')
    except:
        print('!! train on single gpu')
        pass
    if loss_function == "tversky":
        loss = tversky_loss
    elif loss_function == "minh":
        loss = minh_dice_coef_loss
    else:
        loss = weighted_dice_coefficient_loss
    model.compile(optimizer=Adam(lr=initial_learning_rate, beta_1=0.9, beta_2=0.999),
                  loss=loss, metrics=[metrics])
    return model