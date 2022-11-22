import numpy as np

class UmBridgeModel:

    """UmBridgeModel provides a wrapper for an UM-Bridge HTTPModel,
    which allows for using UM-Bridge forward operators directly in a
    tinyDA BlackBoxLinkFactory.

    Attributes
    ----------
    umbridge_model : umbridge.HTTPModel
        An initialised UM-Bridge HTTPModel
    pre : function, optional
        An (optional) function that can be applied to the model input
        before passing it to the UM-Bridge model. Default is None.
    post : function, optional
        An (optional) function that can be applied to the model output
        before passing it to the likelihood. Default is None.
    umbridge_config : dict, optional
        Configuration parameter for UM-Bridge for interoperability with
        the standard UM-Bridge model definitition.
    """

    def __init__(self, umbridge_model, pre=None, post=None, umbridge_config={}):

        """
        Parameters
        ----------
        umbridge_model : umbridge.HTTPModel
            An initialised UM-Bridge HTTPModel
        pre : function, optional
            An (optional) function that can be applied to the model input
            before passing it to the UM-Bridge model. Default is None.
        post : function, optional
            An (optional) function that can be applied to the model output
            before passing it to the likelihood. Default is None.
        umbridge_config : dict, optional
            Configuration parameter for UM-Bridge for interoperability with
            the standard UM-Bridge model definitition.
        """

        # internalise the UM-Bridge model.
        self.umbridge_model = umbridge_model

        # set the preprocessing function.
        if pre is not None:
            self.pre = pre
        else:
            self.pre = lambda x: x

        # set the pstprocessing function.
        if post is not None:
            self.post = post
        else:
            self.post = lambda x: x

        # internalise the UM-Bridge config.
        self.umbridge_config = umbridge_config

    def __call__(self, parameters):

        """
        Parameters
        ----------
        parameters : numpy.ndarray
            A numpy array of model parameters.

        Returns
        ----------
        numpy.ndarray
            The (post-processed) model output.
        """

        # convert NumPy array to standard UM-Bridge input.
        umbridge_input = [self.pre(parameters).tolist()]

        # send converted model input the the UM-Bridge model.
        umbridge_output = self.umbridge_model(umbridge_input, self.umbridge_config)

        # convert the UM-Bridge output back to a NumPy array.
        model_output = np.array(umbridge_output).flatten()

        # return the postprocessed model output.
        return self.post(model_output)
