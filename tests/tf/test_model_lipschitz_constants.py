from pytest import fixture

from ..shared.model_lipschitz_constants import ModelLipschitzConstants
from .injection.models import Models
from .injection.tensors import Tensors


class TestModelLipschitzConstants(ModelLipschitzConstants):

    @fixture(autouse=True)
    def tensors(self):
        self.tensors = Tensors()
        yield self.tensors

    @fixture
    def models(self):
        yield Models()
