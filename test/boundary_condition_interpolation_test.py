import logging
import unittest

import numpy as np

from gpuocean.utils.Common import BoundaryConditions, BoundaryType, BoundaryConditionsData, SingleBoundaryConditionData
from gpuocean.utils.gpu.arrays.bc_arkawa import BoundaryConditionsArakawaA


class DummyStream:
    def synchronize(self):
        pass


class DummyArray:
    def __init__(self):
        self.calls = []

    def upload(self, stream, data):
        self.calls.append(np.array(data, copy=True))


class BoundaryConditionInterpolationTest(unittest.TestCase):
    def test_update_bc_values_uploads_current_and_next_boundary_data_to_separate_arrays(self):
        obj = object.__new__(BoundaryConditionsArakawaA)
        obj.logger = logging.getLogger("test.bc")
        obj.boundary_conditions = BoundaryConditions(
            north=BoundaryType.FLOW_RELAXATION_SCHEME,
            south=BoundaryType.FLOW_RELAXATION_SCHEME,
            east=BoundaryType.FLOW_RELAXATION_SCHEME,
            west=BoundaryType.FLOW_RELAXATION_SCHEME,
            sponge_cells=None,
        )

        h0 = np.array([1.0, 2.0], dtype=np.float32)
        h1 = np.array([3.0, 4.0], dtype=np.float32)
        hu0 = np.array([10.0, 20.0], dtype=np.float32)
        hu1 = np.array([30.0, 40.0], dtype=np.float32)
        hv0 = np.array([100.0, 200.0], dtype=np.float32)
        hv1 = np.array([300.0, 400.0], dtype=np.float32)

        obj.bc_data = BoundaryConditionsData(
            t=[0.0, 10.0],
            north=SingleBoundaryConditionData(h=[h0, h1], hu=[hu0, hu1], hv=[hv0, hv1]),
            south=SingleBoundaryConditionData(h=[h0, h1], hu=[hu0, hu1], hv=[hv0, hv1]),
            east=SingleBoundaryConditionData(h=[h0, h1], hu=[hu0, hu1], hv=[hv0, hv1]),
            west=SingleBoundaryConditionData(h=[h0, h1], hu=[hu0, hu1], hv=[hv0, hv1]),
        )
        obj.bc_timestamps = [None, None]
        obj.bc_NS_current_arr = DummyArray()
        obj.bc_NS_next_arr = DummyArray()
        obj.bc_EW_current_arr = DummyArray()
        obj.bc_EW_next_arr = DummyArray()

        obj.update_bc_values(DummyStream(), 5.0)

        self.assertEqual(len(obj.bc_NS_current_arr.calls), 1)
        self.assertEqual(len(obj.bc_NS_next_arr.calls), 1)
        self.assertEqual(len(obj.bc_EW_current_arr.calls), 1)
        self.assertEqual(len(obj.bc_EW_next_arr.calls), 1)
        self.assertAlmostEqual(obj.bc_t, 0.5)


if __name__ == "__main__":
    unittest.main()
