from keras.models import model_from_json
from keras.layers import Input, LeakyReLU, Add, UpSampling3D, Activation, SpatialDropout3D, Conv3D
from keras.engine import Model
from keras.optimizers import Adam

from unet3d.metrics import weighted_dice_coefficient_loss, tversky_loss, minh_dice_coef_loss, minh_dice_coef_metric

from unet3d.training import load_old_model
from keras.utils import multi_gpu_model


def generate_model(model_file, loss_function="weighted", metrics=minh_dice_coef_metric,
                   initial_learning_rate=0.001):
    print(">> load old model")
    model = load_old_model(model_file)
    print(">> save weights")
    model.save_weights('my_model_weights.h5')
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)

    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)

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
