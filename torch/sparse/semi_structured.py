import warnings
from collections import namedtuple
from typing import Any, Optional

import torch


__all__ = [
    "to_sparse_semi_structured",
    "SparseSemiStructuredTensor",
]

_SEMI_STRUCTURED_SPARSE_CONFIG = namedtuple(
    "_SEMI_STRUCTURED_SPARSE_CONFIG", "compression_factor min_size"
)
_DTYPE_TO_SEMI_STRUCTURED_SPARSE_CONFIG = {
    torch.float16: _SEMI_STRUCTURED_SPARSE_CONFIG(9, 64),
    torch.int8: _SEMI_STRUCTURED_SPARSE_CONFIG(10, 128),
}

_WARNING_SHOWN = False

class SparseSemiStructuredTensor(torch.Tensor):
    """This class implementes semi-structured sparsity as a Tensor subclass.

    Semi-structured sparsity describes a sparsity pattern where n in every 2n elements are sparse,
    depending on the datatype. It is also referred to as 2:4 sparsity or fine-grained
    structured sparsity.

    Currently, this class supports 2:4 sparsity for int8 and float16 dtypes.

    This subclass stores the dense tensor in a compressed form by only storing the specified elemenets and a metadata mask.
    These two are stored next to each other in one contiguous tensor.

    We choose to store the specified elements and the metadata in a single tensor for future compatibilty with cuSPARSELt.

    compressed tensor = [ specified elements of original tensor |   mask_metadata     ]

    For an original tensor of size (m, k) we expect the first m * k // 2 elements to be the kept elements
    The rest of the tensor is metadata.

    This subclass also overrides __torch_dispatch__ to use _structured_sparse_linear for faster matrix multiplications
    via sparse CUTLASS kernels. In the future we will also call into cuSPARSELt kernels for more performance gains.
    """

    @staticmethod
    def __new__(
        cls,
        original_tensor: Optional[torch.Tensor],
        original_shape: Optional[torch.Size] = None,
        mask: Optional[torch.Tensor] = None,
        compressed_tensor: Optional[torch.Tensor] = None,
        transposed: bool = False,
    ):
        """
        Create a new instance of the class.

        When original_tensor is passed in, we compress it and store the compresed representation.
        We can also create new instance of the class from the compressed representation without the original tensor.

        Args:
            original_tensor: The original dense tensor, or None, if we have already compressed the tensor.
            original_shape: The shape of the original dense tensor
            mask: Mask to be applied to the original tensor.
            compressed_tensor: A flattened tensor to store the specified elements and mask metadata.
            transposed: Whether the tensor is transposed or not.

        Returns:
            torch.Tensor: A torch.Tensor wrapper subclass.

        Raises:
            ValueError: If both original_tensor and compressed_tensor are None.

        """
        if original_tensor is not None:
            previous_tensor = original_tensor
            original_shape = original_tensor.shape
        elif compressed_tensor is not None:
            previous_tensor = compressed_tensor
        else:
            raise ValueError("Both compressed_tensor and original_tensor are None!")

        kwargs = {}
        kwargs["device"] = previous_tensor.device  # type: ignore[assignment]
        kwargs["dtype"] = previous_tensor.dtype  # type: ignore[assignment]
        kwargs["layout"] = previous_tensor.layout  # type: ignore[assignment]
        kwargs["requires_grad"] = False  # type: ignore[assignment]

        return torch.Tensor._make_wrapper_subclass(cls, original_shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(
        self,
        original_tensor: Optional[torch.Tensor],
        original_shape: Optional[torch.Size] = None,
        mask: Optional[torch.Tensor] = None,
        compressed_tensor: Optional[torch.Tensor] = None,
        transposed: bool = False,
    ) -> None:
        """SparseSemiStructuredTensor constructor.

        Args:
            original_tensor: The original dense tensor, or None, if we have already compressed the tensor.
            original_shape: The shape of the original dense tensor
            mask: Mask to be applied to the original tensor.
            compressed_tensor: A flattened tensor to store the specified elements and mask metadata.
            transposed: Whether the tensor is transposed or not.

        Returns:
            None

        Raises:
            NotImplementedError: If ``mask=None``, as we currently do not support inferring a mask from the dense tensor.
            RuntimeError: If original_tensor is not a supported dtype, dim, shape, or device.
        """
        global _WARNING_SHOWN
        if not _WARNING_SHOWN:
            warnings.warn(
                (
                    "The PyTorch API of SparseSemiStructuredTensor is in prototype stage "
                    "and will change in the near future. Please open a Github issue "
                    "for features requests and see our documentation on the torch.sparse "
                    "module for further information about the project."
                ),
                UserWarning,
            )
            _WARNING_SHOWN = True

        # if original tensor is passed in, we need to compress it and store the compressed representation.
        if original_tensor is not None:
            # check if mask passed in
            if mask is None:
                raise NotImplementedError("You must pass in a mask to to_sparse_semi_structured, currently mask=None.")

            # check device
            if not original_tensor.is_cuda:
                raise RuntimeError(
                    (
                        f"Error original_tensor.device= {original_tensor.device} is not supported! "
                        "Only CUDA tensors are currently supported."
                    )
                )

            # check dim
            if original_tensor.dim() != 2:
                raise RuntimeError(
                    (
                        f"Error original_tensor.dim = {original_tensor.dim()} is not supported! "
                        "Only 2d tensors are currently supported."
                    )
                )

            # check dtype
            if original_tensor.dtype not in _DTYPE_TO_SEMI_STRUCTURED_SPARSE_CONFIG:
                raise RuntimeError(
                    (
                        f"Error original_tensor.dtype {original_tensor.dtype} is not a supported dtype! "
                        "dtype must be one of: {_DTYPE_TO_SEMI_STRUCTURED_SPARSE_CONFIG}"
                    )
                )

            # check shape
            m, n = original_tensor.shape
            min_size = _DTYPE_TO_SEMI_STRUCTURED_SPARSE_CONFIG[original_tensor.dtype].min_size
            if m < min_size or m % min_size or n < min_size or n % min_size:
                # TODO in the future we can add in padding to support dimensions that aren't perfect multiples
                raise RuntimeError(
                    (
                        f"Error original_tensor.shape {original_tensor.shape} is not supported! "
                        "Both dimensions must be larger than and a multiple of {min_size}"
                    )
                )

            # This code calculates the size of the compressed tensor.
            # compression factor is different based on dtype it's given by the formula below for 2:4 sparsity:
            # compression_factor = 1/2 + 1/bitwidth(dtype)
            original_size = original_tensor.nelement()
            compression_factor = _DTYPE_TO_SEMI_STRUCTURED_SPARSE_CONFIG[
                original_tensor.dtype
            ].compression_factor
            compressed_size = original_size * compression_factor // 16

            compressed_tensor = torch.empty(
                (compressed_size,),
                dtype=original_tensor.dtype,
                device=original_tensor.device,
            )

            # TODO This is a temporoary hack to get the mask in compressed form so we can store the compressed tensor.
            # In the future, we will add in a conversion function from the mask to the meta that we can use instead.
            placeholder = torch.ones(
                (128, n), dtype=original_tensor.dtype, device=original_tensor.device
            )
            specified = original_tensor.masked_select(mask).view(m, n // 2)
            _, meta = torch._structured_sparse_linear(placeholder, specified, mask)
            # set the specified elements
            compressed_tensor[: m * n // 2] = specified.view(-1)
            # set the metadata
            compressed_tensor[m * n // 2 :] = meta.view(original_tensor.dtype).view(-1)

        # set values
        self.original_tensor = None
        self.compressed_tensor = compressed_tensor
        self.transposed = transposed

    def __repr__(self) -> str:
        """Return string representation of SparseSemiStructuredTensor

        Returns:
            str: String representation

        Raises:
            None
        """
        return (
            f"SparseSemiStructuredTensor(shape={self.shape}, "
            f"transposed={self.transposed}"
            f"values={self.values()}"
            f"metadata={self.indices()})"
        )

    __torch_function__ = torch._C._disabled_torch_function_impl

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs) -> Any:
        """Overload __torch_dispatch__ to use torch._structured_sparse_linear.

        `torch.structured_sparse_linear` uses accelerated sparse CUTLASS kernels.
        In the future we plan to also add in support for cuSPARSELt kernels.

        Args:
            func: The function being dispatched.
            types: The types of the arguments.
            args: The arguments passed to the function.
            kwargs: The keyword arguments passed to the function.

        Returns:
            Any: The result of the dispatched operation.

        Raises:
            NotImplementedError: If the dispatched operation is not implemented.
        """
        # Since this code runs below autograd, a detach corresponds to only returning a new object
        if func is torch.ops.aten.detach.default:
            return SparseSemiStructuredTensor(
                args[0].original_tensor,
                original_shape=args[0].shape,
                mask=None,
                compressed_tensor=args[0].compressed_tensor,
                transposed=args[0].transposed,
            )

        # Because we cannot go from the compressed representation back to the dense representation currently,
        # we just keep track of how many times we have been transposed. Depending on whether the sparse matrix
        # is the first or second argument, we expect an even / odd number of calls to transpose respectively.
        if func is torch.ops.aten.t.default:
            return SparseSemiStructuredTensor(
                args[0].original_tensor,
                original_shape=args[0].shape,
                mask=None,
                compressed_tensor=args[0].compressed_tensor,
                transposed=not args[0].transposed,
            )

        # handle addmm
        if func is torch.ops.aten.addmm.default:
            bias, input_A, input_B = args

            # Currently, we only support the first matrix being sparse for addmm/mm in cuSPARSELT and CUTLASS.
            # CUTLASS only supports the first input to be sparse for a given matmul.
            # cuSPARSELt does not have this limitation, although our implementation is only for sparse first.

            # We support second matrix sparse matmul by taking advantage of some transpose properties:
            # This is also why we want an odd number of transposed for second matrix sparse vs an even number
            # of transpose calss for first matrix sparse.
            # F.linear(x) = addmm(bias, input, weight.t()) = b + xW' = (b + xW')''
            #        = (W''x' + b')' = (Wx' + b')' = addmm(bias.T, weight, input).T
            if isinstance(input_B, cls) and input_B.transposed:
                result, _ = torch._structured_sparse_linear(
                    input_A, input_B.values(), input_B.indices(), bias=bias
                )
                return result

        # handle mm
        if func is torch.ops.aten.mm.default:
            input_A, input_B = args

            if isinstance(input_A, cls) and not input_A.transposed:
                transposed_result, _ = torch._structured_sparse_linear(
                    input_B.t(), input_A.values(), input_A.indices()
                )
                return transposed_result.t()

            elif isinstance(input_B, cls) and input_B.transposed:
                result, _ = torch._structured_sparse_linear(
                    input_A, input_B.values(), input_B.indices()
                )
                return result

        # When torch is run with inference mode, pytorch does not decompose torch.ops.aten.linear into a .t() and addmm(),
        # so we must match the aten.linear op.
        # TODO see if there's a way to force pytorch to decompose the op so we don't have to handle this here.
        if func is torch.ops.aten.linear.default:
            input_tensor, weight, bias = args
            if isinstance(weight, cls):
                result, _ = torch._structured_sparse_linear(
                    input_tensor, weight.values(), weight.indices(), bias=bias
                )
                return result

        # handle values
        if func is torch.ops.aten.values.default:
            m, k = args[0].shape
            num_kept_elements = m * k // 2
            return args[0].compressed_tensor[:num_kept_elements].view(m, k // 2)

        # handle indices
        if func is torch.ops.aten.indices.default:
            m, k = args[0].shape
            num_kept_elements = m * k // 2
            metadata = args[0].compressed_tensor[num_kept_elements:].view(m, -1)

            # the metadata is expected to be in different datatypes for fp16/int8 respectively for CUTLASS.
            if args[0].dtype is torch.int8:
                return metadata.view(torch.int32)
            elif args[0].dtype is torch.float16:
                return metadata.view(torch.int16)

        error_string = "\n".join(
            [f"func {func} with args: "]
            + [f"arg{i}: {arg}" for i, arg in enumerate(args)]
        )
        raise NotImplementedError(error_string)


def to_sparse_semi_structured(
    original_tensor: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    transposed: bool = False,
) -> SparseSemiStructuredTensor:
    """
    This function converts a dense tensor into a sparse semi-structured tensor.
    It will return a SparseSemiStructuredTensor, a subclass of torch.Tensor.

    This function will check to ensure the dense tensor has the right dtype, size, dims, and device.
    We currently only support semi-structured sparse tensors for 2d CUDA tensors.
    Additionally, your tensor must be a positive multiple of a block size given the dtype

    - torch.float16  (r, c) must be >= and a multiple of 64
    - torch.int8     (r, c) must be >= and a multiple of 128

    Args:
        original_tensor (Tensor): the dense tensor to convert
        mask (Optional BoolTensor): boolean mask to apply to the original tensor
        transposed (bool, optional): whether the dense tensor is transposed

    Returns:
        SparseSemiStructuredTensor: A sparse semi-structured tensor created from the given original_tensor and mask

    Raises:
        None

    Example:
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
        >>> A = torch.Tensor([0, 0, 1, 1]).tile((128, 32)).half().cuda()
        tensor([[0., 0., 1.,  ..., 0., 1., 1.],
                [0., 0., 1.,  ..., 0., 1., 1.],
                [0., 0., 1.,  ..., 0., 1., 1.],
                ...,
                [0., 0., 1.,  ..., 0., 1., 1.],
                [0., 0., 1.,  ..., 0., 1., 1.],
                [0., 0., 1.,  ..., 0., 1., 1.]], device='cuda:0', dtype=torch.float16)
        >>> A_sparse = to_sparse_semi_structured(A, mask=A.bool())
        SparseSemiStructuredTensor(shape=torch.Size([128, 128]), transposed=False, values=tensor([[1., 1., 1.,  ..., 1., 1., 1.],
                [1., 1., 1.,  ..., 1., 1., 1.],
                [1., 1., 1.,  ..., 1., 1., 1.],
                ...,
                [1., 1., 1.,  ..., 1., 1., 1.],
                [1., 1., 1.,  ..., 1., 1., 1.],
                [1., 1., 1.,  ..., 1., 1., 1.]], device='cuda:0', dtype=torch.float16),
            metadata=tensor([[-4370, -4370, -4370,  ..., -4370, -4370, -4370],
                [-4370, -4370, -4370,  ..., -4370, -4370, -4370],
                [-4370, -4370, -4370,  ..., -4370, -4370, -4370],
                ...,
                [-4370, -4370, -4370,  ..., -4370, -4370, -4370],
                [-4370, -4370, -4370,  ..., -4370, -4370, -4370],
                [-4370, -4370, -4370,  ..., -4370, -4370, -4370]], device='cuda:0',
       dtype=torch.int16))
    """
    return SparseSemiStructuredTensor(original_tensor, mask=mask, transposed=transposed)
