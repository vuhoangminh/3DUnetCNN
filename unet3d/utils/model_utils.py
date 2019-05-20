from keras.utils import multi_gpu_model
from unet3d.training import load_old_model
from tensorflow.python.client import device_lib
import os
from keras.models import model_from_json
from keras.layers import Input, LeakyReLU, Add, UpSampling3D, Activation, SpatialDropout3D, Conv3D
from keras.engine import Model
from keras.optimizers import Adam

from unet3d.metrics import weighted_dice_coefficient_loss
from unet3d.metrics import tversky_loss
from unet3d.metrics import minh_dice_coef_loss
from unet3d.metrics import tv_minh_loss
from unet3d.metrics import tv_weighted_loss
from unet3d.metrics import minh_dice_coef_metric
from unet3d.metrics import dice_coefficient_loss


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


def generate_model(model_file, loss_function="weighted",
                   metrics=minh_dice_coef_metric,
                   initial_learning_rate=0.001,
                   weight_tv_to_main_loss=0.1):

    model = load_model_multi_gpu(model_file)

    return compile_model(model, loss_function=loss_function,
                         metrics=metrics,
                         initial_learning_rate=initial_learning_rate,
                         alpha=weight_tv_to_main_loss)


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    name_gpus = [x.name for x in local_device_protos if x.device_type == 'GPU']
    return len(name_gpus)


def compile_model(model, loss_function="weighted",
                  metrics=minh_dice_coef_metric,
                  initial_learning_rate=0.001,
                  alpha=0.00001):
    try:
        num_gpus = get_available_gpus()
        model = multi_gpu_model(model, gpus=num_gpus)
        print('!! train on multi gpus')
    except:
        print('!! train on single gpu')
        pass
    if loss_function == "tversky":
        loss = tversky_loss
    elif loss_function == "minh":
        loss = minh_dice_coef_loss
    elif loss_function == "tv_minh":
        loss = tv_minh_loss(alpha=alpha)
    elif loss_function == "tv_weighted":
        loss = tv_weighted_loss(alpha=alpha)
    elif loss_function == "weighted":
        loss = weighted_dice_coefficient_loss
    if loss_function == "casweighted":
        model.compile(optimizer=Adam(lr=initial_learning_rate, beta_1=0.9, beta_2=0.999),
                      loss={'out_whole': dice_coefficient_loss,
                            'out_core': dice_coefficient_loss,
                            'out_enh': dice_coefficient_loss
                            },
                      loss_weights={'out_whole': 1,
                                    'out_core': 1,
                                    'out_enh': 1
                                    }
                      )
    elif loss_function == "sepweighted":
        model.compile(optimizer=Adam(lr=initial_learning_rate, beta_1=0.9, beta_2=0.999),
                      loss={'out_1': dice_coefficient_loss,
                            'out_2': dice_coefficient_loss,
                            'out_4': dice_coefficient_loss
                            },
                      loss_weights={'out_1': 1,
                                    'out_2': 1,
                                    'out_4': 1
                                    }
                      )
    else:
        model.compile(optimizer=Adam(lr=initial_learning_rate, beta_1=0.9, beta_2=0.999),
                      loss=loss, metrics=[metrics])

    return model
