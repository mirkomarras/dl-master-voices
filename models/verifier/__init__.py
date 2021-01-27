from models.verifier.thinresnet34 import ThinResNet34

from models.verifier.resnet50 import ResNet50
from models.verifier.resnet34 import ResNet34
from models.verifier.xvector import XVector
from models.verifier.vggvox import VggVox

MODEL_DICT = {'xvector': XVector, 'vggvox': VggVox, 'resnet50': ResNet50, 'resnet34': ResNet34, 'thin_resnet': ThinResNet34}


def get_model(netv):
    if '/v' in netv:
        arch, version = netv.split('/v')
        version = int(version)
    else:
        version = -1
    assert arch in MODEL_DICT, "Specified model is not supported!"
    return MODEL_DICT[arch](id=version)
