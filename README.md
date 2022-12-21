
# Stable Voting

README.md for the code used in the paper [Stable Voting](https://arxiv.org/abs/2108.00542) by W. Holliday and E. Pacuit.

The notebooks use the Python package ``pref_voting``.   Consult [https://pref-voting.readthedocs.io/](https://pref-voting.readthedocs.io/) for an overview of this package.  

## Notebooks

* 01-StableVotingExamples.ipynb: This notebook is the main notebook discussing Simple Stable Voting and Stable Voting, including all the examples from the papers.   

* 02-AnalyzingElectionData.ipynb: Read and find the winnners in real elections from preflib.org (the election data is available in preflib-data/). 

* 03-RunningTimes.ipynb: Graphs comparing the running times of Simple Stable Voting and Stable Voting. 



## Other Files/Directories

1. preflib-data/: Data from preflib.org of actual elections discussed in the paper. 

2. data/: CSV files containing data for the graphs comparing the running times of Simple Stable Voting and Stable Voting. 

## Requirements

All the code is written in Python 3. 

- [Preferential Voting Tools](https://pref-voting.readthedocs.io/en/latest/)
- The notebooks and the pref_voting library is built around a full SciPy stack: [MatPlotLib](https://matplotlib.org/), [Numpy](https://numpy.org/), [Pandas](https://pandas.pydata.org/), [numba](http://numba.pydata.org/), [networkx](https://networkx.org/), and [tabulate](https://github.com/astanin/python-tabulate)
- [tqdm.notebook](https://github.com/tqdm/tqdm)
- [seaborn](https://seaborn.pydata.org/)  
- [multiprocess](https://pypi.org/project/multiprocess/) (only needed if running the simulations in  05-ProbabilisticStabilityWinners.ipynb) 
- [PrefLibTools](https://github.com/PrefLib/preflibtools) (for analyzing the elections)


 

```python

```
