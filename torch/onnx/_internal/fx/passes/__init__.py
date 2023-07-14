from .decomp import Decompose
from .functionalization import Functionalize, RemoveInputMutation
from .readability import RestoreParameterAndBufferNames
from .shape_inference import ShapeInferenceWithFakeTensor
from .type_promotion import InsertTypePromotion
from .virtualization import MovePlaceholderToFront, ReplaceGetAttrWithPlaceholder

__all__ = [
    "Decompose",
    "InsertTypePromotion",
    "Functionalize",
    "MovePlaceholderToFront",
    "RemoveInputMutation",
    "RestoreParameterAndBufferNames",
    "ReplaceGetAttrWithPlaceholder",
    "ShapeInferenceWithFakeTensor",
]
