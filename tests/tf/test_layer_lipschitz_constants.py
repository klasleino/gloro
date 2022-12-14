from pytest import fixture

from ..shared.layer_lipschitz_constants import LayerLipschitzConstants
from .injection.layers import Layers
from .injection.tensors import Tensors


class TestLayerLipschitzConstants(LayerLipschitzConstants):

    @fixture(autouse=True)
    def tensors(self):
        self.tensors = Tensors()
        yield self.tensors

    @fixture
    def layers(self):
        yield Layers()
