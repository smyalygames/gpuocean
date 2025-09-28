from stochastic.RandThroughOceanNoise_test import RandThroughOceanNoiseTest


class RandThroughOceanNoiseLCGTest(RandThroughOceanNoiseTest):
    """
    Executing all the same tests as RandomNumbersTest, but
    using the LCG algorithm for random numbers.
    """
        
    def useLCG(self):
        return True