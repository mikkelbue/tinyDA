# external imports
from itertools import compress
import numpy as np
from tqdm import tqdm

# internal imports
from .proposal import *
from .utils import *


class Chain:

    """Chain is a single level MCMC sampler. It is initialsed with a
    LinkFactory (which holds the model and the distributions, and returns
    Links), and a proposal (transition kernel).

    Attributes
    ----------
    link_factory : tinyDA.LinkFactory
        A link factory responsible for communation between prior, likelihood
        and model. It also generates instances of tinyDA.Link (sample objects).
    proposal : tinyDA.Proposal
        Transition kernel for MCMC proposals.
    initial_parameters : numpy.ndarray
        Starting point for the MCMC sampler
    chain : list
        Samples ("Links") in the MCMC chain.
    accepted : list
        List of bool, signifying whether a proposal was accepted or not.

    Methods
    -------
    sample(iterations)
        Runs the MCMC for the specified number of iterations.
    """

    def __init__(self, link_factory, proposal, initial_parameters=None):

        """
        Parameters
        ----------
        link_factory : tinyDA.LinkFactory
            A link factory responsible for communation between prior, likelihood
            and model. It also generates instances of tinyDA.Link (sample objects).
        proposal : tinyDA.Proposal
            Transition kernel for MCMC proposals.
        initial_parameters : numpy.ndarray, optional
            Starting point for the MCMC sampler, default is None (random draw
            from the prior).
        """

        # internalise the link factory and the proposal
        self.link_factory = link_factory
        self.proposal = proposal

        # initialise a list, which holds the links.
        self.chain = []

        # initialise a list, which holds boolean acceptance values.
        self.accepted = []

        # if the initial parameters are given, use them. otherwise,
        # draw a random sample from the prior.
        if initial_parameters is not None:
            self.initial_parameters = initial_parameters
        else:
            self.initial_parameters = self.link_factory.prior.rvs()

        # append a link with the initial parameters (this will automatically
        # compute the model output, and the relevant probabilties.
        self.chain.append(self.link_factory.create_link(self.initial_parameters))
        self.accepted.append(True)

        # setup the proposal
        self.proposal.setup_proposal(
            parameters=self.initial_parameters, link_factory=self.link_factory
        )

    def sample(self, iterations, progressbar=True):

        """
        Parameters
        ----------
        iterations : int
            Number of MCMC samples to generate.
        progressbar : bool, optional
            Whether to draw a progressbar, default is True.
        """

        # Set up a progressbar, if required.
        if progressbar:
            pbar = tqdm(range(iterations))
        else:
            pbar = range(iterations)

        # start the iteration
        for i in pbar:
            if progressbar:
                pbar.set_description(
                    "Running chain, \u03B1 = %0.2f" % np.mean(self.accepted[-100:])
                )

            # draw a new proposal, given the previous parameters.
            proposal = self.proposal.make_proposal(self.chain[-1])

            # create a link from that proposal.
            proposal_link = self.link_factory.create_link(proposal)

            # compute the acceptance probability, which is unique to
            # the proposal.
            alpha = self.proposal.get_acceptance(proposal_link, self.chain[-1])

            # perform Metropolis adjustment.
            if np.random.random() < alpha:
                self.chain.append(proposal_link)
                self.accepted.append(True)
            else:
                self.chain.append(self.chain[-1])
                self.accepted.append(False)

            # adapt the proposal. if the proposal is set to non-adaptive,
            # this has no effect.
            self.proposal.adapt(
                parameters=self.chain[-1].parameters,
                jumping_distance=self.chain[-1].parameters - self.chain[-2].parameters,
                accepted=self.accepted,
            )

        # close the progressbar if it was initialised.
        if progressbar:
            pbar.close()


class DAChain:

    """DAChain is a two-level Delayed Acceptance sampler. It takes a coarse and
    a fine link factory as input, as well as a proposal, which applies to the
    coarse level only.

    Attributes
    ----------
    link_factory_coarse : tinyDA.LinkFactory
        A "coarse" link factory responsible for communation between prior,
        likelihood and model. It also generates instances of tinyDA.Link
        (sample objects).
    link_factory_fine : tinyDA.LinkFactory
        A "fine" link factory responsible for communation between prior,
        likelihood and model. It also generates instances of tinyDA.Link
        (sample objects).
    proposal : tinyDA.Proposal
        Transition kernel for coarse MCMC proposals.
    subsampling_rate : int
        The subsampling rate for the coarse chain.
    initial_parameters : numpy.ndarray
        Starting point for the MCMC sampler
    chain_coarse : list
        Samples ("Links") in the coarse MCMC chain.
    accepted_coarse : list
        List of bool, signifying whether a coarse proposal was accepted or not.
    is_coarse : list
        List of bool, signifying whether a coarse link was generated by the
        coarse sampler or not.
    chain_fine : list
        Samples ("Links") in the fine MCMC chain.
    accepted_fine : list
        List of bool, signifying whether a fine proposal was accepted or not.
    adaptive_error_model : str or None
        The adaptive error model, see e.g. Cui et al. (2019).
    R : numpy.ndarray
        Restriction matrix for the adaptive error model.
    bias : tinaDA.RecursiveSampleMoments
        A recursive Gaussian error model that computes the sample moments of
        the coarse bias.

    Methods
    -------
    sample(iterations)
        Runs the MCMC for the specified number of iterations.
    """

    def __init__(
        self,
        link_factory_coarse,
        link_factory_fine,
        proposal,
        subsampling_rate,
        initial_parameters=None,
        adaptive_error_model=None,
        R=None,
    ):

        """
        Parameters
        ----------
        link_factory_coarse : tinyDA.LinkFactory
            A "coarse" link factory responsible for communation between prior,
            likelihood and model.It also generates instances of tinyDA.Link
            (sample objects).
        link_factory_fine : tinyDA.LinkFactory
            A "fine" link factory responsible for communation between prior,
            likelihood and model. It also generates instances of tinyDA.Link
            (sample objects).
        proposal : tinyDA.Proposal
            Transition kernel for coarse MCMC proposals.
        subsampling_rate : int
            The subsampling rate for the coarse chain.
        initial_parameters : numpy.ndarray, optional
            Starting point for the MCMC sampler, default is None (random draw
            from prior).
        adaptive_error_model : str or None, optional
            The adaptive error model, see e.g. Cui et al. (2019). Default is
            None (no error model), options are 'state-independent' or
            'state-dependent'. If an error model is used, the likelihood MUST
            have a set_bias() method, use e.g. tinyDA.AdaptiveLogLike.
        R : numpy.ndarray or None, optional
            Restriction matrix for the adaptive error model. Default is None
            (identity matrix).
        """

        # internalise link factories and the proposal
        self.link_factory_coarse = link_factory_coarse
        self.link_factory_fine = link_factory_fine
        self.proposal = proposal
        self.subsampling_rate = subsampling_rate

        # set up lists to hold coarse and fine links, as well as acceptance
        # accounting
        self.chain_coarse = []
        self.accepted_coarse = []
        self.is_coarse = []

        self.chain_fine = []
        self.accepted_fine = []

        # if the initial parameters are given, use them. otherwise,
        # draw a random sample from the prior.
        if initial_parameters is not None:
            self.initial_parameters = initial_parameters
        else:
            self.initial_parameters = self.link_factory_coarse.prior.rvs()

        # append a link with the initial parameters to the coarse chain.
        self.chain_coarse.append(
            self.link_factory_coarse.create_link(self.initial_parameters)
        )
        self.accepted_coarse.append(True)
        self.is_coarse.append(False)

        # append a link with the initial parameters to the coarse chain.
        self.chain_fine.append(
            self.link_factory_fine.create_link(self.initial_parameters)
        )
        self.accepted_fine.append(True)

        # setup the proposal
        self.proposal.setup_proposal(
            parameters=self.initial_parameters, link_factory=self.link_factory_coarse
        )

        # set the adaptive error model as an attribute.
        self.adaptive_error_model = adaptive_error_model

        # set up the adaptive error model.
        if self.adaptive_error_model is not None:

            if R is None:
                self.R = np.eye(self.chain_fine[-1].model_output.shape[0])
            else:
                self.R = R

            # compute the difference between coarse and fine level.
            self.model_diff = (
                self.R.dot(self.chain_fine[-1].model_output)
                - self.chain_coarse[-1].model_output
            )

            if self.adaptive_error_model == "state-independent":
                # for the state-independent error model, the bias is
                # RecursiveSampleMoments, and the corrector is the mean
                # of all sampled differences.
                self.bias = RecursiveSampleMoments(
                    self.model_diff,
                    np.zeros((self.model_diff.shape[0], self.model_diff.shape[0])),
                )
                self.link_factory_coarse.likelihood.set_bias(
                    self.bias.get_mu(), self.bias.get_sigma()
                )

            elif self.adaptive_error_model == "state-dependent":
                # for the state-dependent error model, the bias is
                # ZeroMeanRecursiveSampleMoments, and the corrector is
                # the last sampled difference.
                self.bias = ZeroMeanRecursiveSampleMoments(
                    np.zeros((self.model_diff.shape[0], self.model_diff.shape[0]))
                )
                self.link_factory_coarse.likelihood.set_bias(
                    self.model_diff, self.bias.get_sigma()
                )

            self.chain_coarse[-1] = self.link_factory_coarse.update_link(
                self.chain_coarse[-1]
            )

    def sample(self, iterations, progressbar=True):
        """
        Parameters
        ----------
        iterations : int
            Number of MCMC samples to generate.
        progressbar : bool, optional
            Whether to draw a progressbar, default is True.
        """

        # Set up a progressbar, if required.
        if progressbar:
            pbar = tqdm(range(iterations))
        else:
            pbar = range(iterations)

        # start the iteration
        for i in pbar:
            if progressbar:
                pbar.set_description(
                    "Running chain, \u03B1_c = {0:.3f}, \u03B1_f = {1:.2f}".format(
                        np.mean(
                            self.accepted_coarse[-int(100 * self.subsampling_rate) :]
                        ),
                        np.mean(self.accepted_fine[-100:]),
                    )
                )

            # subsample the coarse model.
            for j in range(self.subsampling_rate):

                # draw a new proposal, given the previous parameters.
                proposal = self.proposal.make_proposal(self.chain_coarse[-1])

                # create a link from that proposal.
                proposal_link_coarse = self.link_factory_coarse.create_link(proposal)

                # compute the acceptance probability, which is unique to
                # the proposal.
                alpha_1 = self.proposal.get_acceptance(
                    proposal_link_coarse, self.chain_coarse[-1]
                )

                # perform Metropolis adjustment.
                if np.random.random() < alpha_1:
                    self.chain_coarse.append(proposal_link_coarse)
                    self.accepted_coarse.append(True)
                    self.is_coarse.append(True)
                else:
                    self.chain_coarse.append(self.chain_coarse[-1])
                    self.accepted_coarse.append(False)
                    self.is_coarse.append(True)

                # adapt the proposal. if the proposal is set to non-adaptive,
                # this has no effect.
                self.proposal.adapt(
                    parameters=self.chain_coarse[-1].parameters,
                    jumping_distance=self.chain_coarse[-1].parameters
                    - self.chain_coarse[-2].parameters,
                    accepted=list(compress(self.accepted_coarse, self.is_coarse)),
                )

            if sum(self.accepted_coarse[-self.subsampling_rate :]) == 0:
                self.chain_fine.append(self.chain_fine[-1])
                self.accepted_fine.append(False)
                self.chain_coarse.append(
                    self.chain_coarse[-(self.subsampling_rate + 1)]
                )
                self.accepted_coarse.append(False)
                self.is_coarse.append(False)

            else:
                # when subsampling is complete, create a new fine link from the
                # previous coarse link.
                proposal_link_fine = self.link_factory_fine.create_link(
                    self.chain_coarse[-1].parameters
                )

                # compute the delayed acceptance probability.
                if self.adaptive_error_model == "state-dependent":

                    bias_next = (
                        self.R.dot(proposal_link_fine.model_output)
                        - self.chain_coarse[-1].model_output
                    )
                    coarse_state_biased = self.link_factory_coarse.update_link(
                        self.chain_coarse[-(self.subsampling_rate + 1)], bias_next
                    )

                    if self.proposal.is_symmetric:
                        q_x_y = q_y_x = 0
                    else:
                        q_x_y = self.proposal.get_q(
                            self.chain_fine[-1], proposal_link_fine
                        )
                        q_y_x = self.proposal.get_q(
                            proposal_link_fine, self.chain_fine[-1]
                        )

                    alpha_2 = np.exp(
                        min(
                            proposal_link_fine.posterior + q_y_x,
                            coarse_state_biased.posterior + q_x_y,
                        )
                        - min(
                            self.chain_fine[-1].posterior + q_x_y,
                            self.chain_coarse[-1].posterior + q_y_x,
                        )
                    )

                else:
                    alpha_2 = np.exp(
                        proposal_link_fine.posterior
                        - self.chain_fine[-1].posterior
                        + self.chain_coarse[-(self.subsampling_rate + 1)].posterior
                        - self.chain_coarse[-1].posterior
                    )

                # Perform Metropolis adjustment, and update the coarse chain
                # to restart from the previous accepted fine link.
                if np.random.random() < alpha_2:
                    self.chain_fine.append(proposal_link_fine)
                    self.accepted_fine.append(True)
                    self.chain_coarse.append(self.chain_coarse[-1])
                    self.accepted_coarse.append(True)
                    self.is_coarse.append(False)
                else:
                    self.chain_fine.append(self.chain_fine[-1])
                    self.accepted_fine.append(False)
                    self.chain_coarse.append(
                        self.chain_coarse[-(self.subsampling_rate + 1)]
                    )
                    self.accepted_coarse.append(False)
                    self.is_coarse.append(False)

            # update the adaptive error model.
            if self.adaptive_error_model is not None:

                if self.adaptive_error_model == "state-independent":
                    # for the state-independent AEM, we simply update the
                    # RecursiveSampleMoments with the difference between
                    # the fine and coarse model output
                    self.model_diff = (
                        self.R.dot(self.chain_fine[-1].model_output)
                        - self.chain_coarse[-1].model_output
                    )
                    self.bias.update(self.model_diff)

                    # and update the likelihood in the coarse link factory.
                    self.link_factory_coarse.likelihood.set_bias(
                        self.bias.get_mu(), self.bias.get_sigma()
                    )

                elif self.adaptive_error_model == "state-dependent":
                    # for the state-dependent error model, we want the
                    # difference, corrected with the previous difference
                    # to compute the error covariance.
                    self.model_diff_corrected = self.R.dot(
                        self.chain_fine[-1].model_output
                    ) - (self.chain_coarse[-1].model_output + self.model_diff)

                    # and the "pure" model difference for the offset.
                    self.model_diff = (
                        self.R.dot(self.chain_fine[-1].model_output)
                        - self.chain_coarse[-1].model_output
                    )

                    # update the ZeroMeanRecursiveSampleMoments with the
                    # corrected difference.
                    self.bias.update(self.model_diff_corrected)

                    # and update the likelihood in the coarse link factory
                    # with the "pure" difference.
                    self.link_factory_coarse.likelihood.set_bias(
                        self.model_diff, self.bias.get_sigma()
                    )

                self.chain_coarse[-1] = self.link_factory_coarse.update_link(
                    self.chain_coarse[-1]
                )

        # close the progressbar if it was initialised.
        if progressbar:
            pbar.close()


class MLDAChain:

    """MLDAChain is a Multilevel Delayed Acceptance sampler. It takes a list of
    link factories of increasing level as input, as well as a proposal, which
    applies to the coarsest level only.

    Attributes
    ----------
    link_factory : tinyDA.LinkFactory
        The finest level link factory responsible for communation between
        prior, likelihood and model. It also generates instances of
        tinyDA.Link (sample objects).
    level : int
        The (numeric) level of the finest level sampler,
    proposal : tinyDA.MLDA
        MLDA transition kernel for the next-coarser MCMC proposals.
    subsampling_rate : int
        The subsampling rate for the next-coarser chain.
    initial_parameters : numpy.ndarray
        Starting point for the MCMC sampler
    chain : list
        Samples ("Links") in the finest level MCMC chain.
    accepted : list
        List of bool, signifying whether a proposal was accepted or not.
    adaptive_error_model : str or None
        The adaptive error model, see e.g. Cui et al. (2019).
    R : numpy.ndarray
        Restriction matrix for the adaptive error model on the finest level.
    bias : tinaDA.RecursiveSampleMoments
        A recursive Gaussian error model that computes the sample moments
        of the next-coarser bias.

    Methods
    -------
    sample(iterations)
        Runs the MCMC for the specified number of iterations.
    """

    def __init__(
        self,
        link_factories,
        proposal,
        subsampling_rates,
        initial_parameters=None,
        adaptive_error_model=None,
        R=None,
    ):

        """
        Parameters
        ----------
        link_factories : list
            List of instances of tinyDA.LinkFactory, in increasing order.
        proposal : tinyDA.Proposal
            Transition kernel for coarsest MCMC proposals.
        subsampling_rates : list
            List of subsampling rates. It must have length
            len(link_factories) - 1, in increasing order.
        initial_parameters : numpy.ndarray, optional
            Starting point for the MCMC sampler, default is None (random
            draw from prior).
        adaptive_error_model : str or None, optional
            The adaptive error model, see e.g. Cui et al. (2019). Default
            is None (no error model), options are 'state-independent' or
            'state-dependent'. If an error model is used, the likelihood
            MUST have a set_bias() method, use e.g. tinyDA.AdaptiveLogLike.
        R : list or None, optional
            Restriction matrices for the adaptive error model. If list of
            restriction matrices is given, it must have length
            len(link_factories) - 1 and they must be in increasing order,
            as the model hierachy. Default is None (identity matrices).
        """

        # internalise the finest link factory and set the level.
        self.link_factory = link_factories[-1]
        self.level = len(link_factories) - 1

        # set the furrent level subsampling rate.
        self.subsampling_rate = subsampling_rates[-1]

        # initialise a list, which holds the links.
        self.chain = []

        # initialise a list, which holds boolean acceptance values.
        self.accepted = []

        # if the initial parameters are given, use them. otherwise,
        # draw a random sample from the prior.
        if initial_parameters is not None:
            self.initial_parameters = initial_parameters
        else:
            self.initial_parameters = self.link_factory.prior.rvs()

        # append a link with the initial parameters (this will automatically
        # compute the model output, and the relevant probabilties.
        self.chain.append(self.link_factory.create_link(self.initial_parameters))
        self.accepted.append(True)

        # make sure R can be passed recursively to MLDA proposal, if MLDAChain
        # was initialised without tda.sample().
        if R is None:
            R = [None] * self.level

        # set the effective proposal to MLDA which runs on the next-coarser level.
        self.proposal = MLDA(
            link_factories[:-1],
            proposal,
            subsampling_rates[:-1],
            self.initial_parameters,
            adaptive_error_model,
            R[:-1],
        )

        # set the adative error model as an attribute.
        self.adaptive_error_model = adaptive_error_model

        # set up the adaptive error model.
        if self.adaptive_error_model is not None:

            # extract the finest-level restriction matrix from the list.
            self.R = R[-1]

            # if no restriction matrix was given, create an identity matrix.
            if self.R is None:
                self.R = np.eye(self.chain[-1].model_output.shape[0])

            # compute the difference between coarse and fine level.
            self.model_diff = (
                self.R.dot(self.chain[-1].model_output)
                - self.proposal.chain[-1].model_output
            )

            # set up the state-independent adaptive error model.
            if self.adaptive_error_model == "state-independent":
                # for the state-independent error model, the bias is
                # RecursiveSampleMoments, and the corrector is the mean
                # of all sampled differences.
                self.bias = RecursiveSampleMoments(
                    self.model_diff,
                    np.zeros((self.model_diff.shape[0], self.model_diff.shape[0])),
                )
                self.proposal.link_factory.likelihood.set_bias(
                    self.bias.get_mu(), self.bias.get_sigma()
                )

            # state-dependent error model has not been implemented, since
            # it may not be ergodic.
            elif self.adaptive_error_model == "state-dependent":
                pass

            # update the first coarser link with the adaptive error model.
            self.proposal.chain[-1] = self.proposal.link_factory.update_link(
                self.proposal.chain[-1]
            )

            # collect biases and restriction matrices in a list.
            self.biases = [self.bias]
            self.Rs = [self.R]

            # pass the bias models and restriction levels to the next-coarser level.
            self.proposal.setup_adaptive_error_model(self.biases, self.Rs)

    def sample(self, iterations, progressbar=True):

        """
        Parameters
        ----------
        iterations : int
            Number of MCMC samples to generate.
        progressbar : bool, optional
            Whether to draw a progressbar, default is True.
        """

        # Set up a progressbar, if required.
        if progressbar:
            pbar = tqdm(range(iterations))
        else:
            pbar = range(iterations)

        # start the iteration
        for i in pbar:
            if progressbar:
                pbar.set_description(
                    "Running chain, \u03B1 = %0.2f" % np.mean(self.accepted[-100:])
                )

            # draw a new proposal, given the previous parameters.
            proposal = self.proposal.make_proposal(self.subsampling_rate)

            if sum(self.proposal.accepted[-self.subsampling_rate :]) == 0:
                self.chain.append(self.chain[-1])
                self.accepted.append(False)

            else:

                # create a link from that proposal.
                proposal_link = self.link_factory.create_link(proposal)

                # compute the acceptance probability, which is unique to
                # the proposal.
                alpha = self.proposal.get_acceptance(
                    proposal_link,
                    self.chain[-1],
                    self.proposal.chain[-1],
                    self.proposal.chain[-(self.subsampling_rate + 1)],
                )

                # perform Metropolis adjustment.
                if np.random.random() < alpha:
                    self.chain.append(proposal_link)
                    self.accepted.append(True)
                else:
                    self.chain.append(self.chain[-1])
                    self.accepted.append(False)

            # make sure the next-coarser level is aligned.
            self.proposal.align_chain(self.chain[-1].parameters, self.accepted[-1])

            # apply the adaptive error model.
            if self.adaptive_error_model is not None:

                # compute the difference between coarse and fine level.
                if self.accepted[-1]:
                    self.model_diff = (
                        self.R.dot(self.chain[-1].model_output)
                        - self.proposal.chain[-1].model_output
                    )

                # update the state-independent adaptive error model.
                if self.adaptive_error_model == "state-independent":
                    # for the state-independent error model, the bias is
                    # RecursiveSampleMoments, and the corrector is the mean
                    # of all sampled differences.
                    self.bias.update(self.model_diff)
                    self.proposal.link_factory.likelihood.set_bias(
                        self.bias.get_mu(), self.bias.get_sigma()
                    )

                # state-dependent error model has not been implemented.
                elif self.adaptive_error_model == "state-dependent":
                    pass

                # update the latest coarser link with the adaptive error model.
                self.proposal.chain[-1] = self.proposal.link_factory.update_link(
                    self.proposal.chain[-1]
                )

        # close the progressbar if it was initialised.
        if progressbar:
            pbar.close()
