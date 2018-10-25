#-*- coding: utf-8 -*-
import pymzn

pymzn.debug(False)

"""
PyMzn is a Python library that wraps the MiniZinc tools for constraint programming. PyMzn is built on top 
of the minizinc and enables to parse the solutions into Python objects. Author: Paolo Dragone, PhD student at the 
University of Trento (Italy).

Minizic is an open-source constraint modeling language. Minizinc compiles the model into FlatZinc, a low-level
common input language that is understood by a wide range of solvers. The solver used is Gecode, an open source C++ 
toolkit for developing constraint-based applications and is included in the Minizinc package.

"""
def solver(service, length, env):
    """
    Python wrapper calls minizinc model(.mzn) using python variables instead of stored data file(.dzn). To execute
    the model as standalone use:

    ./minizinc -c placement.mzn placement.dzn   It generates placement.fzn (Flatzinc model)
    ./fzn-gecode placement.fzn

    Args:
        service[length] -- Collects the service chain
        length(int) -- Length of the service
        env(obj) -- Instance of the environment
    """

    chain = service.tolist()
    chain = chain[:length]
    chain = [x + 1 for x in chain]          # Minizinc array indexes start at 1

    weights = [env.service_properties[i]['size'] for i in range(env.numDescriptors)]

    s = pymzn.minizinc('placement.mzn', solver='gecode', timeout=30, parallel=4, data={'numBins': env.numBins,
                                                                        'numSlots': env.numSlots,
                                                                        'chainLen': length,
                                                                        'chain': chain,
                                                                        'numDescriptors': env.numDescriptors,
                                                                        'weights': weights
                                                                     })
    placement = s[0]['placement']
    placement = [x - 1 for x in placement]

    return placement
