from stochastic.OceanStateNoise_test import OceanStateNoiseTest

class OceanStateNoiseLCGTest(OceanStateNoiseTest):
    """
    Executing all the same tests as OceanStateNoiseTest, but
    using the LCG algorithm for random numbers.
    """
        
    def useLCG(self):
        return True