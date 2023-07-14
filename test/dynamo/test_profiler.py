# Owner(s): ["module: dynamo"]
from unittest.mock import patch

import torch

import torch._dynamo.test_case
import torch._dynamo.testing
import torch._dynamo.utils

from torch._dynamo.utils import dynamo_timed


class DynamoProfilerTests(torch._dynamo.test_case.TestCase):
    def test_dynamo_timed_profiling_isolated(self):
        # @dynamo_timed functions should appear in profile traces.
        @dynamo_timed
        def inner_fn(x):
            return x.sin()

        def outer_fn(x, y):
            return inner_fn(x) * y

        x, y = [torch.rand((2, 2)) for _ in range(2)]

        with torch.profiler.profile(with_stack=False) as prof:
            outer_fn(x, y)

        self.assertTrue(
            any("inner_fn (dynamo_timed)" in evt.name for evt in prof.events())
        )

    def test_dynamo_timed_profiling_backend_compile(self):
        # @dynamo_timed functions should appear in profile traces.
        # this checks whether these actually appear in actual dynamo execution.
        # "backend_compile" is just chosen as an example; if it gets renamed
        # this test can be replaced or deleted

        fn_name = "call_user_compiler"

        def fn(x, y):
            return x.sin() * y.cos()

        x, y = [torch.rand((2, 2)) for _ in range(2)]

        with torch.profiler.profile(with_stack=False) as prof:
            torch._dynamo.optimize("aot_eager")(fn)(x, y)

        self.assertTrue(
            any(f"{fn_name} (dynamo_timed)" in evt.name for evt in prof.events())
        )

    @patch.object(torch._dynamo.config, "assume_static_by_default", False)
    def test_profile_dynamic_shapes_runtime(self):
        def fn(x, y, z):
            return x @ y + z

        opt_fn = torch._dynamo.optimize("aot_eager", dynamic=True, nopython=True)(fn)

        inputs = [
            (torch.rand(a, b), torch.rand(b, c), torch.rand(a, c))
            for (a, b, c) in [(15, 16, 17), (15, 15, 16), (16, 16, 16)]
        ]

        opt_fn(*inputs[0])
        opt_fn(*inputs[1])

        with torch.profiler.profile(record_shapes=True):
            opt_fn(*inputs[2])

    @patch.object(torch._dynamo.config, "assume_static_by_default", False)
    def test_profile_dynamic_shapes_compilation(self):
        def fn(x, y, z):
            return x @ y + z

        opt_fn = torch._dynamo.optimize("aot_eager", dynamic=True, nopython=True)(fn)

        inputs = (torch.rand(15, 16), torch.rand(16, 17), torch.rand(15, 17))

        with torch.profiler.profile(record_shapes=True):
            opt_fn(*inputs)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
