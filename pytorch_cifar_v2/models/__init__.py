from .nas_models import DARTS_V1, DARTS_V2, AmoebaNet, NASNet
model_dict = {
    'darts_v1': DARTS_V1(),
    'darts_v2': DARTS_V2(),
    'amoebaNet': AmoebaNet(),
    'nasnet': NASNet()
}


def get_model(model_name):
    return model_dict[model_name]
