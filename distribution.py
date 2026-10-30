#        Adaptive-width ESM embedding distillation
#
#   File:     distribution.py
#   Authors:  Federico Errica (federico.errica@neclab.eu)
#             Nithya Bhasker (nithya.bhasker@neclab.eu)
#             Raman Siarheyeu (raman.siarheyeu@neclab.eu)
#             Catherine Chaput (catherine.chaput@neclab.eu)
#
# NEC Laboratories Europe GmbH, Copyright (c) 2026, All rights reserved.
#
#        THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
#
#        PROPRIETARY INFORMATION ---
#
# SOFTWARE LICENSE AGREEMENT
#
# ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY
#
# BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS
# LICENSE AGREEMENT.  IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR
# DOWNLOAD THE SOFTWARE.
#
# This is a license agreement ("Agreement") between your academic institution
# or non-profit organization or self (called "Licensee" or "You" in this
# Agreement) and NEC Laboratories Europe GmbH (called "Licensor" in this
# Agreement).  All rights not specifically granted to you in this Agreement
# are reserved for Licensor.
#
# RESERVATION OF OWNERSHIP AND GRANT OF LICENSE: Licensor retains exclusive
# ownership of any copy of the Software (as defined below) licensed under this
# Agreement and hereby grants to Licensee a personal, non-exclusive,
# non-transferable license to use the Software for noncommercial research
# purposes, without the right to sublicense, pursuant to the terms and
# conditions of this Agreement. NO EXPRESS OR IMPLIED LICENSES TO ANY OF
# LICENSOR'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. As used in this
# Agreement, the term "Software" means (i) the actual copy of all or any
# portion of code for program routines made accessible to Licensee by Licensor
# pursuant to this Agreement, inclusive of backups, updates, and/or merged
# copies permitted hereunder or subsequently supplied by Licensor,  including
# all or any file structures, programming instructions, user interfaces and
# screen formats and sequences as well as any and all documentation and
# instructions related to it, and (ii) all or any derivatives and/or
# modifications created or made by You to any of the items specified in (i).
#
# CONFIDENTIALITY/PUBLICATIONS: Licensee acknowledges that the Software is
# proprietary to Licensor, and as such, Licensee agrees to receive all such
# materials and to use the Software only in accordance with the terms of this
# Agreement.  Licensee agrees to use reasonable effort to protect the Software
# from unauthorized use, reproduction, distribution, or publication. All
# publication materials mentioning features or use of this software must
# explicitly include an acknowledgement the software was developed by NEC
# Laboratories Europe GmbH.
#
# COPYRIGHT: The Software is owned by Licensor.
#
# PERMITTED USES:  The Software may be used for your own noncommercial
# internal research purposes. You understand and agree that Licensor is not
# obligated to implement any suggestions and/or feedback you might provide
# regarding the Software, but to the extent Licensor does so, you are not
# entitled to any compensation related thereto.
#
# DERIVATIVES: You may create derivatives of or make modifications to the
# Software, however, You agree that all and any such derivatives and
# modifications will be owned by Licensor and become a part of the Software
# licensed to You under this Agreement.  You may only use such derivatives and
# modifications for your own noncommercial internal research purposes, and you
# may not otherwise use, distribute or copy such derivatives and modifications
# in violation of this Agreement.
#
# BACKUPS:  If Licensee is an organization, it may make that number of copies
# of the Software necessary for internal noncommercial use at a single site
# within its organization provided that all information appearing in or on the
# original labels, including the copyright and trademark notices are copied
# onto the labels of the copies.
#
# USES NOT PERMITTED:  You may not distribute, copy or use the Software except
# as explicitly permitted herein. Licensee has not been granted any trademark
# license as part of this Agreement.  Neither the name of NEC Laboratories
# Europe GmbH nor the names of its contributors may be used to endorse or
# promote products derived from this Software without specific prior written
# permission.
#
# You may not sell, rent, lease, sublicense, lend, time-share or transfer, in
# whole or in part, or provide third parties access to prior or present
# versions (or any parts thereof) of the Software.
#
# ASSIGNMENT: You may not assign this Agreement or your rights hereunder
# without the prior written consent of Licensor. Any attempted assignment
# without such consent shall be null and void.
#
# TERM: The term of the license granted by this Agreement is from Licensee's
# acceptance of this Agreement by downloading the Software or by using the
# Software until terminated as provided below.
#
# The Agreement automatically terminates without notice if you fail to comply
# with any provision of this Agreement.  Licensee may terminate this Agreement
# by ceasing using the Software.  Upon any termination of this Agreement,
# Licensee will delete any and all copies of the Software. You agree that all
# provisions which operate to protect the proprietary rights of Licensor shall
# remain in force should breach occur and that the obligation of
# confidentiality described in this Agreement is binding in perpetuity and, as
# such, survives the term of the Agreement.
#
# FEE: Provided Licensee abides completely by the terms and conditions of this
# Agreement, there is no fee due to Licensor for Licensee's use of the
# Software in accordance with this Agreement.
#
# DISCLAIMER OF WARRANTIES:  THE SOFTWARE IS PROVIDED "AS-IS" WITHOUT WARRANTY
# OF ANY KIND INCLUDING ANY WARRANTIES OF PERFORMANCE OR MERCHANTABILITY OR
# FITNESS FOR A PARTICULAR USE OR PURPOSE OR OF NON- INFRINGEMENT.  LICENSEE
# BEARS ALL RISK RELATING TO QUALITY AND PERFORMANCE OF THE SOFTWARE AND
# RELATED MATERIALS.
#
# SUPPORT AND MAINTENANCE: No Software support or training by the Licensor is
# provided as part of this Agreement.
#
# EXCLUSIVE REMEDY AND LIMITATION OF LIABILITY: To the maximum extent
# permitted under applicable law, Licensor shall not be liable for direct,
# indirect, special, incidental, or consequential damages or lost profits
# related to Licensee's use of and/or inability to use the Software, even if
# Licensor is advised of the possibility of such damage.
#
# EXPORT REGULATION: Licensee agrees to comply with any and all applicable
# export control laws, regulations, and/or other laws related to embargoes and
# sanction programs administered by law.
#
# SEVERABILITY: If any provision(s) of this Agreement shall be held to be
# invalid, illegal, or unenforceable by a court or other tribunal of competent
# jurisdiction, the validity, legality and enforceability of the remaining
# provisions shall not in any way be affected or impaired thereby.
#
# NO IMPLIED WAIVERS: No failure or delay by Licensor in enforcing any right
# or remedy under this Agreement shall be construed as a waiver of any future
# or other exercise of such right or remedy by Licensor.
#
# GOVERNING LAW: This Agreement shall be construed and enforced in accordance
# with the laws of Germany without reference to conflict of laws principles.
# You consent to the personal jurisdiction of the courts of this country and
# waive their rights to venue outside of Germany.
#
# ENTIRE AGREEMENT AND AMENDMENTS: This Agreement constitutes the sole and
# entire agreement between Licensee and Licensor as to the matter set forth
# herein and supersedes any previous agreements, understandings, and
# arrangements between the parties relating hereto.
#
#        THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
#

from typing import Tuple

import torch
from mlwiz.evaluation.util import return_class_and_args
from torch.nn import Parameter, Module
from torch.nn.functional import softplus, relu, sigmoid


def inv_softplus(bias: float | torch.Tensor) -> float | torch.Tensor:
    """Inverse softplus function.

    Args:
        bias (float or tensor): the value to be softplus-inverted.
    """
    is_tensor = True
    if not isinstance(bias, torch.Tensor):
        is_tensor = False
        bias = torch.tensor(bias)
    out = bias.expm1().clamp_min(1e-6).log()
    if not is_tensor and out.numel() == 1:
        return out.item()
    return out


class ContinuousDistribution(Module):
    """
    Implements an interface for this package
    """

    def __init__(self, quantile: float = None):
        super().__init__()
        self.register_buffer("_one", torch.tensor(1.0))
        self.default_quantile = quantile

    def _validate_args(self, value):
        assert isinstance(
            value, torch.Tensor
        ), f"expected torch tensor, found {type(value)}"

        # assert isinstance(value, torch.FloatTensor) or (
        #     value.dtype == torch.float32
        # ), f"expected float tensor, found {value.dtype}"

        assert (
            len(value.shape) == 2
        ), f"expected shape: (N,1), found {value.shape}"

        assert (
            value.shape[1] == 1
        ), f"expected one-dimensional values, found {value.shape}"

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """
        Computes the log pdf of the distribution

        :param value: a tensor of shape Nx1, where N is the number of samples

        :return: a tensor of shape Nx1
        """
        raise NotImplementedError(
            "You should subclass Distribution and " "implement this method."
        )

    def cdf(self, value):
        """
        Computes the cdf of the distribution

        :param value: a tensor of shape Nx1, where N is the number of samples

        :return: a tensor of shape Nx1
        """
        raise NotImplementedError(
            "You should subclass Distribution and " "implement this method."
        )

    def quantile(
        self, p: float = 0.95, override_rate: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the p-quantile for the distribution.

        :param p: the parameter p of the quantile
        :param override_rate: optional rate to use for quantile computation.
            Distributions that do not use a rate can ignore this.

        :return: lower and upper bounds for the p-quantile. If the p-quantile
            can be computed exactly then they are the same

        """
        raise NotImplementedError(
            "You should subclass Distribution and " "implement this method."
        )

    @property
    def parameter(self) -> torch.Tensor:
        raise NotImplementedError(
            "You should subclass Distribution and " "implement this method."
        )


class RateParameterizedDistribution(ContinuousDistribution):
    """Continuous width distribution calibrated through a positive rate.

    Subclasses expose a trainable unconstrained ``_rate`` parameter and map a
    requested quantile width to its positive rate through ``_find_rate``.
    AWN uses this contract when bootstrapping widths from a fixed model.
    """

    def _find_rate(self, desired_size: float, quantile: float) -> torch.Tensor:
        raise NotImplementedError

    @property
    def rate(self) -> torch.Tensor:
        raise NotImplementedError


class Exponential(RateParameterizedDistribution):
    def __init__(
        self, starting_size: int, quantile: float = 0.95, max_size: int = None
    ):
        super().__init__(quantile=quantile)
        self.register_buffer(
            "_one", torch.tensor([1.0], dtype=torch.get_default_dtype())
        )

        assert starting_size >= 1

        self._rate = Parameter(
            inv_softplus(
                torch.tensor([self._find_rate(starting_size, quantile)]).to(
                    torch.get_default_dtype()
                )
            ),
            requires_grad=True,
        )

        if max_size is not None:
            self.boundary = Parameter(
                torch.tensor([self._find_rate(max_size, quantile)]).to(
                    torch.get_default_dtype()
                ),
                requires_grad=False,
            )
        else:
            self.boundary = None

        self.min_boundary = Parameter(
            torch.tensor([self._find_rate(2, quantile)]).to(
                torch.get_default_dtype()
            ),
            requires_grad=False,
        )

    def _validate_args(self, value):
        super()._validate_args(value)

        assert torch.all(value >= 0), (
            f"Input values cannot be smaller"
            f" than 0. Rate is {self.rate} and values are {value}"
        )

    def _find_rate(self, desired_size: float, quantile: float):
        return -torch.log(self._one - quantile) / desired_size

    @property
    def rate(self) -> torch.Tensor:
        r = softplus(self._rate) + 1e-32

        # Export is inference-only here and cannot capture parameter mutation.
        if torch.compiler.is_exporting():
            return torch.clamp(
                r,
                min=self.boundary,
                max=self.min_boundary,
            )

        if self.boundary is not None and self.boundary > r:
            self._rate.data = inv_softplus(self.boundary.data)
            r = softplus(self._rate) + 1e-32
            return r

        if self.boundary is not None and self.min_boundary < r:
            self._rate.data = inv_softplus(self.min_boundary.data)
            r = softplus(self._rate) + 1e-32
            return r

        return r

    @property
    def parameter(self) -> torch.Tensor:
        return self.rate

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        return torch.log(self.rate) - self.rate * value

    def cdf(self, value):
        return self._one - torch.exp(-self.rate * value)

    def quantile(
        self, p: float = 0.95, override_rate: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # t = time.time()

        # Allow callers to infer widths for a specific rate without mutating
        # the trainable distribution parameter.
        rate = self.rate if override_rate is None else override_rate
        q = -torch.log(self._one - p) / rate

        return q, q


class DiscretizedDistribution(Module):
    def __init__(self, **kwargs):
        """
        Creates a discretized version of a continuous distribution such that

            p(x) = phi(x+1) - phi(x)

        where phi is the cdf of the original distribution.

        :param kwargs: a dictionary with a key 'base_distribution' that
            allows us to instantiate a discretized distribution
        """
        super().__init__()

        base_d_cls, base_d_args = return_class_and_args(
            kwargs, "base_distribution"
        )
        base_d_args = {} if base_d_args is None else dict(base_d_args)
        base_d_args["quantile"] = kwargs["quantile"]
        self.base_distribution = base_d_cls(**base_d_args)
        self.register_buffer("_one", torch.tensor(1.0))

    def get_q_named_parameters(self) -> dict:
        return self.base_distribution.get_q_named_parameters()

    def _validate_args(self, value):
        self.base_distribution._validate_args(value)

        # check values are integers
        assert torch.allclose(
            value, value.int().to(torch.get_default_dtype())
        ), f"expected float tensor with integer values, got {value}."

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """
        Computes the log pdf of the distribution

        :param value: a tensor of shape Nx1, where N is the number of samples

        :return: a tensor of shape Nx1
        """
        # self._validate_args(value)

        # one = torch.ones(1).to(value.device)
        # avoids a degenerate case where the base distribution has the
        # same cdf for both value and value+1
        # which leads to nan. Also, a too small value can cause some
        # distributions to have prob 1 for a single neuron, and the model
        # gets trapped in there
        # tmp = torch.ones(1, device=value.device) * 1e-6
        return torch.log(
            self.base_distribution.cdf(value + self._one)
            - self.base_distribution.cdf(value)
            + 1e-6
        )

    def cdf(self, value: torch.Tensor) -> torch.Tensor:
        """
        Computes the cdf of the distribution

        :param value: a tensor of shape Nx1, where N is the number of samples

        :return: a tensor of shape Nx1
        """
        # self._validate_args(value)
        # one = torch.ones(1, device=value.device)

        return self.base_distribution.cdf(value + self._one)

    def quantile(
        self, p: float = 0.95, override_rate: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the approximated p-quantile for the discrete distribution.
        The lower and upper bounds returned by the method will coincide, since
        we provide the smallest integer x such that cdf(x) >= p

        :param p: the parameter p
        :param override_rate: optional base-distribution rate used to
            evaluate quantiles without updating model parameters.

        :return: lower and upper bounds for the p-quantile. If the p-quantile
            can be computed exactly then they are the same
        """
        lower_bound, upper_bound = self.base_distribution.quantile(
            p, override_rate=override_rate
        )

        # TODO TEST: see if we can avoid the call to CDF
        # corner case or case when quantile is known exactly
        if lower_bound is upper_bound or lower_bound == upper_bound:
            u = upper_bound.to(self._one.device)
            return u, u

        # Now perform binary search over the integers to find the smallest x
        # such that cdf(x) >= p. The boundaries of the search are given by the
        # bounds, and we use the fact that the cdf forms an ordered sequence
        l = torch.floor(lower_bound).to(self._one.device)
        u = torch.ceil(upper_bound).to(self._one.device)

        # ------------------------------------------------------------------ #
        # code that saves the day in case you implemented wrong bounds
        if self.cdf((u).unsqueeze(1)) < p:
            print(
                "WARNING: your upper bound on the quantile is "
                f"not working as expected:{(u, self.cdf((u).unsqueeze(1)))}"
            )
            ok = False
            while not ok:
                u += 1
                if self.cdf((u).unsqueeze(1)) >= p:
                    ok = True
            return u, u  # + 2 to avoid degenerate cases
        # ------------------------------------------------------------------ #

        # if lower bound is already sufficient, stop, the normal and folded
        # normal curves are very similar at the desired quantile
        if self.cdf(l.unsqueeze(1)) >= p:
            # we could test l-1 (because of discretization), but in the end
            # it does not make a big difference
            assert self.cdf(l.unsqueeze(1)) >= p
            return l, l  # + 2 to avoid degenerate cases

        # Binary search over integer-valued points.
        # Invariant: cdf(l) < p <= cdf(u). Keep u as a valid candidate.
        while l < u:
            m = torch.floor((l + u) / 2.0)
            cdf_m = self.cdf(m.unsqueeze(1))

            if cdf_m < p:
                # Move L to the right.
                l = m + 1.0
            else:
                # Keep M in the candidate set to preserve cdf(u) >= p.
                u = m

        assert self.cdf(u.unsqueeze(1)) >= p
        return u, u

    def compute_probability_vector(self, x) -> torch.Tensor:
        """
        Computes the **renormalized** vector of probabilities on the fly

        :return: a vector of arbitrary length with the probabilities
        """
        # Support values are not registered module state, so callers may still
        # hold them on the device used before the distribution was moved.
        x = x.to(device=self._one.device, dtype=self._one.dtype)

        if isinstance(self.base_distribution, Exponential):
            # For an exponential distribution,
            # P(k <= X < k + 1) = (1 - exp(-rate)) * exp(-rate * k).
            # The first factor is shared by every bin and cancels when the
            # vector is normalized, so normalize the geometric logits with
            # softmax directly.
            rate = self.base_distribution.rate
            return torch.softmax(-rate * x.squeeze(1), dim=0)

        log_probs = self.log_prob(x).squeeze(1)
        probs = log_probs.exp()
        probs = probs / probs.sum()
        return probs

    @property
    def mean(self) -> torch.Tensor:
        return self.base_distribution.mean

    @property
    def variance(self) -> torch.Tensor:
        return self.base_distribution.variance


class TruncatedDistribution(Module):
    def __init__(self, truncation_quantile: float, **kwargs):
        """
        Truncates a discretized distribution to a given quantile and
        renormalizes its probability.

        :param truncation_quantile: the quantile in [0,1] at which we want
            to truncate the discrete distribution.
        :param kwargs: a dictionary with a key 'discretized_distribution' that
            allows us to instantiate a discretized distribution
        """
        super().__init__()

        dist_d_cls, dist_d_args = return_class_and_args(
            kwargs, "discretized_distribution"
        )
        dist_d_args = {} if dist_d_args is None else dict(dist_d_args)
        dist_d_args["quantile"] = truncation_quantile
        self.discretized_distribution = dist_d_cls(**dist_d_args)
        self.truncation_quantile = truncation_quantile

        self.x = None

    def get_q_named_parameters(self) -> dict:
        return self.discretized_distribution.get_q_named_parameters()

    def compute_truncation_number(self) -> int:
        """
        Computes the truncation number at the specified quantile.

        :return: a positive integer holding the truncation number

        """

        # A captured module graph has fixed parameter shapes, so its width must
        # match the support that was materialized before export.  Reusing that
        # static size also avoids converting a rate-dependent tensor to a
        # Python integer while tracing.  Eager execution continues to derive
        # the width from the current rate below.
        if torch.compiler.is_compiling() and self.x is not None:
            return self.x.size(0)

        # exploits the implementation of quantile() for the
        # DiscretizedDistribution, which returns

        _, truncation_number = self.discretized_distribution.quantile(
            p=self.truncation_quantile
        )

        truncation_number = int(torch.round(truncation_number).item())

        return truncation_number

    def compute_width_from_rate(self, rate: torch.Tensor) -> int:
        """
        Infer an integer width from a base-distribution rate.

        This utility keeps the rate-width calibration in the base
        distribution by using its quantile implementation.
        """
        _, width = self.quantile(
            p=self.truncation_quantile, override_rate=rate
        )
        return max(int(torch.round(width).item()), 1)

    def compute_probability_vector(self) -> torch.Tensor:
        """
        Computes the **renormalized** vector of probabilities on the fly

        :return: a vector of arbitrary length with the probabilities
        """
        truncation_number = self.compute_truncation_number()
        reference = self.discretized_distribution._one

        if (
            self.x is None
            or self.x.size(0) != truncation_number
            or self.x.device != reference.device
            or self.x.dtype != reference.dtype
        ):
            # no gradient so far, we detach on purpose
            self.x = torch.arange(
                truncation_number,
                dtype=reference.dtype,
                device=reference.device,
            ).unsqueeze(1)

        probs = self.discretized_distribution.compute_probability_vector(
            self.x
        )

        # assert torch.allclose(probs.sum(), torch.ones(1, device=probs.device)),(
        #    probs, self.discretized_distribution.base_distribution.k, self.discretized_distribution.base_distribution.b)
        return probs

    def compute_probability_vector_known_width(self, width) -> torch.Tensor:
        """
        Computes the **renormalized** vector of probabilities on the fly

        :return: a vector of arbitrary length with the probabilities
        """
        reference = self.discretized_distribution._one

        if (
            self.x is None
            or self.x.size(0) != width
            or self.x.device != reference.device
            or self.x.dtype != reference.dtype
        ):
            # no gradient so far, we detach on purpose
            self.x = torch.arange(
                width,
                dtype=reference.dtype,
                device=reference.device,
            ).unsqueeze(1)

        probs = self.discretized_distribution.compute_probability_vector(
            self.x
        )

        # assert torch.allclose(probs.sum(), torch.ones(1, device=probs.device)),(
        #    probs, self.discretized_distribution.base_distribution.k, self.discretized_distribution.base_distribution.b)
        return probs

    @property
    def mean(self) -> torch.Tensor:
        proba = self.compute_probability_vector()
        return (proba * torch.arange(len(proba)).to(proba.device)).sum()

    def quantile(
        self, p: float = 0.95, override_rate: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the approximated p-quantile for the discrete distribution.
        The lower and upper bounds returned by the method will coincide, since
        we provide the smallest integer x such that cdf(x) >= p

        :param p: the parameter p
        :param override_rate: optional base-distribution rate used to
            evaluate quantiles without updating model parameters.

        :return: lower and upper bounds for the p-quantile. If the p-quantile
            can be computed exactly then they are the same
        """
        return self.discretized_distribution.quantile(
            p, override_rate=override_rate
        )
