# Disable eager execution if using Tensorflow 2.
try:
    from tensorflow.python.framework.ops import disable_eager_execution

    disable_eager_execution()
except:
    pass

from gloro.models import GloroNet

from gloro.constants import GLORO_CUSTOM_OBJECTS