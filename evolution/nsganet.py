""" 
based on nsga2 from https://github.com/msu-coinlab/pymoo
"""
import numpy as np
from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.core.individual import Individual
from pymoo.core.survival import Survival
from pymoo.docs import parse_doc_string
from pymoo.operators.crossover.pntx import PointCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.selection.tournament import TournamentSelection, compare
from pymoo.util.dominator import Dominator
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.randomized_argsort import randomized_argsort


class NSGANet(GeneticAlgorithm):
    """Custom Genetic Algorithm"""

    def __init__(self, **kwargs):
        kwargs["individual"] = Individual(rank=np.inf, crowding=-1)
        super().__init__(**kwargs)

        self.tournament_type = "comp_by_dom_and_crowding"


# pylint: disable = unused-argument
def binary_tournament(pop, P, algorithm, **kwargs):
    """
    Binary Tournament Selection Function
    """
    if P.shape[1] != 2:
        raise ValueError("Only implemented for binary tournament!")

    tournament_type = algorithm.tournament_type
    # pylint: disable = invalid-name
    S = np.full(P.shape[0], np.nan)

    for i in range(P.shape[0]):
        a, b = P[i, 0], P[i, 1]

        # if at least one solution is infeasible
        if pop[a].CV > 0.0 or pop[b].CV > 0.0:
            S[i] = compare(
                a, pop[a].CV, b, pop[b].CV, method="smaller_is_better", return_random_if_equal=True
            )

        # both solutions are feasible
        else:
            if tournament_type == "comp_by_dom_and_crowding":
                rel = Dominator.get_relation(pop[a].F, pop[b].F)
                if rel == 1:
                    S[i] = a
                elif rel == -1:
                    S[i] = b

            elif tournament_type == "comp_by_rank_and_crowding":
                S[i] = compare(a, pop[a].rank, b, pop[b].rank, method="smaller_is_better")

            else:
                raise ValueError("Unknown tournament type.")

            # if rank or domination relation didn't make a decision compare by crowding
            if np.isnan(S[i]):
                S[i] = compare(
                    a,
                    pop[a].get("crowding"),
                    b,
                    pop[b].get("crowding"),
                    method="larger_is_better",
                    return_random_if_equal=True,
                )

    return S[:, None].astype(int)


# pylint: disable = too-few-public-methods
class RankAndCrowdingSurvival(Survival):
    """
    Survival Selection
    """

    def __init__(self) -> None:
        super().__init__(True)

    # pylint: disable = arguments-differ
    def _do(self, problem, pop, n_survive, D=None, **kwargs):
        # get the objective space values and objects
        # pylint: disable = invalid-name
        F = pop.get("F")

        # the final indices of surviving individuals
        survivors = []

        # do the non-dominated sorting until splitting front
        fronts = NonDominatedSorting().do(F, n_stop_if_ranked=n_survive)

        for k, front in enumerate(fronts):
            # calculate the crowding distance of the front
            crowding_of_front = calc_crowding_distance(F[front, :])

            # save rank and crowding in the individual class
            for j, i in enumerate(front):
                pop[i].set("rank", k)
                pop[i].set("crowding", crowding_of_front[j])

            # current front sorted by crowding distance if splitting
            if len(survivors) + len(front) > n_survive:
                I = randomized_argsort(crowding_of_front, order="descending", method="numpy")
                I = I[: (n_survive - len(survivors))]

            # otherwise take the whole front unsorted
            else:
                I = np.arange(len(front))

            # extend the survivors by all or selected individuals
            survivors.extend(front[I])

        return pop[survivors]


# pylint: disable = invalid-name
def calc_crowding_distance(F):
    """
    Calculate Crowd Distance
    """
    infinity = 1e14

    n_points = F.shape[0]
    n_obj = F.shape[1]

    if n_points <= 2:
        return np.full(n_points, infinity)
    else:
        # sort each column and get index
        I = np.argsort(F, axis=0, kind="mergesort")

        # now really sort the whole array
        F = F[I, np.arange(n_obj)]

        # get the distance to the last element in sorted list and replace zeros with actual values
        dist = np.concatenate([F, np.full((1, n_obj), np.inf)]) - np.concatenate(
            [np.full((1, n_obj), -np.inf), F]
        )

        index_dist_is_zero = np.where(dist == 0)

        dist_to_last = np.copy(dist)
        for i, j in zip(*index_dist_is_zero):
            dist_to_last[i, j] = dist_to_last[i - 1, j]

        dist_to_next = np.copy(dist)
        for i, j in reversed(list(zip(*index_dist_is_zero))):
            dist_to_next[i, j] = dist_to_next[i + 1, j]

        # normalize all the distances
        norm = np.max(F, axis=0) - np.min(F, axis=0)
        norm[norm == 0] = np.nan
        dist_to_last, dist_to_next = dist_to_last[:-1] / norm, dist_to_next[1:] / norm

        # if we divided by zero because all values in one columns are equal replace by none
        dist_to_last[np.isnan(dist_to_last)] = 0.0
        dist_to_next[np.isnan(dist_to_next)] = 0.0

        # sum up the distance to next and last and norm by objectives -
        # also reorder from sorted list
        J = np.argsort(I, axis=0)
        crowding = (
            np.sum(dist_to_last[J, np.arange(n_obj)] + dist_to_next[J, np.arange(n_obj)], axis=1)
            / n_obj
        )

    # replace infinity with a large number
    crowding[np.isinf(crowding)] = infinity

    return crowding


def nsganet(
    pop_size=100,
    sampling=IntegerRandomSampling(),
    selection=TournamentSelection(func_comp=binary_tournament),
    crossover=PointCrossover(n_points=2),
    mutation=PolynomialMutation(eta=3, vtype=int),
    eliminate_duplicates=True,
    n_offsprings=None,
    **kwargs
):
    """
    Interface

    Parameters
    ----------
    pop_size : {pop_size}
    sampling : {sampling}
    selection : {selection}
    crossover : {crossover}
    mutation : {mutation}
    eliminate_duplicates : {eliminate_duplicates}
    n_offsprings : {n_offsprings}

    Returns
    -------
    nsganet : :class:`~pymoo.model.algorithm.Algorithm`
        Returns an NSGANet algorithm object.


    """

    return NSGANet(
        pop_size=pop_size,
        sampling=sampling,
        selection=selection,
        crossover=crossover,
        mutation=mutation,
        survival=RankAndCrowdingSurvival(),
        eliminate_duplicates=eliminate_duplicates,
        n_offsprings=n_offsprings,
        **kwargs
    )


parse_doc_string(nsganet)
