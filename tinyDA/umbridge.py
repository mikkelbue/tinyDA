import numpy as np

class UmBridgeModel:

    def __init__(self, umbridge_model, pre=None, post=None, umbridge_config={}):

        self.umbridge_model = umbridge_model

        if pre is not None:
            self.pre = pre
        else:
            self.pre = lambda x: x

        if post is not None:
            self.post = post
        else:
            self.post = lambda x: x
            
        self.umbridge_config = umbridge_config

    def __call__(self, parameters):

        umbridge_input = [self.pre(parameters).tolist()]

        umbridge_output = self.umbridge_model(umbridge_input, self.umbridge_config)

        model_output = np.array(umbridge_output).flatten()

        return self.post(model_output)
