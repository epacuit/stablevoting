
# Stable Voting

README.md for the code used in the paper [Stable Voting](https://arxiv.org/abs/2108.00542) by W. Holliday and E. Pacuit.

## Where to start

tl;dr: An overview of profiles and voting methods is found in 01-Profiles.ipynb and 03-VotingMethods.ipynb, respectively.   See 04-StableVoting.ipynb for the implementation of (Simple) Stable Voting and some examples. See 05-AnalyzingElectionData.ipynb for the code to anaylize actual elections. See 02-ProfilesWithTies.ipynb  for an overview of elections in which voters submit strict weak orders.   Finally, 06-RunningTimes.ipynb compares the running time of Simple Stable Voting and Stable Voting. 

## Dev Notes

* In order to optimize some of the code for reasoning about profiles within linear orders, it is assumed that in any profile the candidates are named by the initial segment of the non-negative integers.  So, in a profile with 5 candidates, the candidate names are "0, 1, 2, 3, and 4".   Use the `cmap` variable for different candidate names: `cmap` is a dictionary with keys 0, 1, ..., num_cands - 1 and values the "real" names of the candidates.  

## Notebooks

1. 01-Profile.ipynb: This notebook is an overview of how to create profiles, remove candidates from a profile and generate profiles according to various probability models.    

2. 02-ProfilesWithTies.ipynb: Profiles with voters that submit strict weak orders over the candidates (implemented in voting/profiles_with_ties.py). 

3. 03-VotingMethods.ipynb: This notebook is an overview of the voting methods that are available (implemented in voting/voting_methods.py). 

4. 04-StableVoting.ipynb: This main notebook discussing Simple Stable Voting and Stable Voting. 

5. 05-AnalyzingElectionData.ipynb: Read and find the winnners in real elections from preflib.org (the election data is available in preflib-data/). 

6. 06-RunningTimes.ipynb: Graphs comparing the running times of Simple Stabel Voting and Stable Voting. 

## Profiles and Voting Methods

A profile (of linear orders over the candidates) is created by initializing a Profile class object.  This needs a list of rankings (each ranking is a tuple of numbers), the number of candidates, and a list giving the number of each ranking in the profile:

```python
from voting.profiles import Profile

rankings = [(0, 1, 2, 3), (2, 3, 1, 0), (3, 1, 2, 0), (1, 2, 0, 3), (1, 3, 2, 0)]
num_cands = 4
rcounts = [5, 3, 2, 4, 3]

prof = Profile(rankings, num_cands, rcounts=rcounts)
```

The function generate_profile is used to generate a profile for a given number of candidates and voters:  
```python
from voting.generate_profiles import generate_profile

# generate a profile using the Impartial Culture probability model
prof = generate_profile(3, 4) # prof is a Profile object with 3 candidate and 4 voters

# generate a profile using the Impartial Anonymous Culture probability model
prof = generate_profile(3, 4, probmod = "IAC") # prof is a Profile object with 3 candidate and 4 voters
```

Many voting methods are implemented in voting/voting_methods.py: 

```python
from voting.profiles import Profile
from voting.voting_methods import *

prof = Profile(rankings, num_cands, rcounts=rcounts)
print(f"The {split_cycle_faster.name} winners are {split_cycle_faster(prof)}")
```
## Other Files/Directories

1. voting/profiles.py: Implementation of the Profile class used to create and reason about profile (see 01-Profile.ipynb for an overview).

2. voting/profiles_with_ties.py: Implementation of the ProfileWithTies class used from elections in which voters submit strict weak orders (see 02-ProfileWithTies.ipynb for an overview). 

3. voting/voting_methods.py: Implementations of the voting methods (see 03-VotingMethods.ipynb for an overview).

4. voting/generate_profiles.py: Implementation of  the function `generate_profile` to interface with the Preflib tools to generate profiles according to different probability models. 

5. preflib-data/: Data from preflib.org of actual elections discussed in the paper. 

6. data/: CSV files containing data for the graphs comparing the running times of Simple Stable Voting and Stable Voting. 

## Requirements

All the code is written in Python 3. 

- [Preflib tools](https://github.com/PrefLib/PrefLib-Tools) (available in the voting/preflibtools directory)
- The notebooks and most of the library is built around a full SciPy stack: [MatPlotLib](https://matplotlib.org/), [Numpy](https://numpy.org/), [Pandas](https://pandas.pydata.org/)
- [numba](http://numba.pydata.org/) 
- [networkx](https://networkx.org/)
- [tabulate](https://github.com/astanin/python-tabulate)
- [seaborn](https://seaborn.pydata.org/)  
- [multiprocess](https://pypi.org/project/multiprocess/) (only needed if running the simulations in  05-ProbabilisticStabilityWinners.ipynb) 
- [tqdm.notebook](https://github.com/tqdm/tqdm)


 
