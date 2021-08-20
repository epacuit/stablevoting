'''
    File: profile_optimized.py
    Author: Eric Pacuit (epacuit@umd.edu)
    Date: December 7, 2020
    
    Functions to generate profile
    
    Most of the functions to generate the profile are from the preflib tools (preflib.org)
    
    Main Functions
    --------------
    prob_models: dictionary with functions to create profiles for various probability models
    generate_profile: generates a profile
    
'''


from .profiles import Profile
import numpy as np # for the SPATIAL model
import math
import random

# ############
# wrapper functions to interface with preflib tools for generating profiles
# ############

# ## URN model ###

# Generate votes based on the URN Model.
# we need num_cands and num_voters  with replace replacements.
# This function is a small modification of the same function used
# in preflib.org to generate profiles 
def gen_urn(num_cands, num_voters, replace):
    
    voteMap = {}
    ReplaceVotes  = {}
    
    ICsize = math.factorial(num_cands)
    ReplaceSize = 0

    for x in range(num_voters):
        flip =  random.randint(1, ICsize+ReplaceSize)
        if flip <= ICsize:
            #generate an IC vote and make a suitable number of replacements...
            tvote = tuple(np.random.permutation(num_cands)) # gen_ic_vote(alts)
            voteMap[tvote] = (voteMap.get(tvote, 0) + 1)
            ReplaceVotes[tvote] = (ReplaceVotes.get(tvote, 0) + replace)
            ReplaceSize += replace
        else:
            #iterate over replacement hash and select proper vote.
            flip = flip - ICsize
            for vote in ReplaceVotes.keys():
                flip = flip - ReplaceVotes[vote]
                if flip <= 0:
                    voteMap[vote] = (voteMap.get(vote, 0) + 1)
                    ReplaceVotes[vote] = (ReplaceVotes.get(vote, 0) + replace)
                    ReplaceSize += replace
                    break
            else:
                print("We Have a problem... replace fell through....")
                exit()
    return voteMap

def create_rankings_urn(num_cands, num_voters, replace):
    """create a list of rankings using the urn model
    """
    vote_map = gen_urn(num_cands, num_voters, replace)
    return [vc[0] for vc in vote_map.items()], [vc[1] for vc in vote_map.items()]        



# ### Mallows Model ####

# For Phi and a given number of candidates, compute the
# insertion probability vectors.
def compute_mallows_insertvec_dist(ncand, phi):
    #Compute the Various Mallows Probability Distros
    vec_dist = {}
    for i in range(1, ncand+1):
        #Start with an empty distro of length i
        dist = [0] * i
        #compute the denom = phi^0 + phi^1 + ... phi^(i-1)
        denom = sum([pow(phi,k) for k in range(i)])
        #Fill each element of the distro with phi^i-j / denom
        for j in range(1, i+1):
            dist[j-1] = pow(phi, i - j) / denom
        #print(str(dist) + "total: " + str(sum(dist)))
        vec_dist[i] = dist
    return vec_dist

# Return a value drawn from a particular distribution.
def draw(values, distro):
    #Return a value randomly from a given discrete distribution.
    #This is a bit hacked together -- only need that the distribution
    #sums to 1.0 within 5 digits of rounding.
    if round(sum(distro),5) != 1.0:
        print("Input Distro is not a Distro...")
        print(str(distro) + "  Sum: " + str(sum(distro)))
        exit()
    if len(distro) != len(values):
        print("Values and Distro have different length")

    cv = 0
    draw = random.random() - distro[cv]
    while draw > 0.0:
        cv+= 1
        draw -= distro[cv]
    return values[cv]

# Generate a Mallows model with the various mixing parameters passed in
# nvoters is the number of votes we need
# candmap is a candidate map
# mix is an array such that sum(mix) == 1 and describes the distro over the models
# phis is an array len(phis) = len(mix) = len(refs) that is the phi for the particular model
# refs is an array of dicts that describe the reference ranking for the set.
def gen_mallows(num_cands, num_voters, mix, phis, refs):

    if len(mix) != len(phis) or len(phis) != len(refs):
        print("Mix != Phis != Refs")
        exit()

    #Precompute the distros for each Phi and Ref.
    #Turn each ref into an order for ease of use...
    m_insert_dists = []
    for i in range(len(mix)):
        m_insert_dists.append(compute_mallows_insertvec_dist(num_cands, phis[i]))
    #Now, generate votes...
    votemap = {}
    for cvoter in range(num_voters):
        cmodel = draw(list(range(len(mix))), mix)
        #print("cmodel is ", cmodel)
        #Generate a vote for the selected model
        insvec = [0] * num_cands
        for i in range(1, len(insvec)+1):
            #options are 1...max
            insvec[i-1] = draw(list(range(1, i+1)), m_insert_dists[cmodel][i])
        vote = []
        for i in range(len(refs[cmodel])):
            #print("building vote insvec[i] - 1", insvec[i]-1)
            vote.insert(insvec[i]-1, refs[cmodel][i])
        tvote = tuple(vote)
        
        votemap[tuple(vote)] = votemap.get(tuple(vote), 0) + 1
    return votemap

def create_rankings_mallows(num_cands, num_voters, phi, ref=None):
    
    ref = tuple(np.random.permutation(num_cands))
    
    vote_map = gen_mallows(num_cands, num_voters, [1.0], [phi], [ref])
    
    return [vc[0] for vc in vote_map.items()], [vc[1] for vc in vote_map.items()]        


def create_rankings_mallows_two_rankings(num_cands, num_voters, phi, ref=None):
    '''create a profile using a Mallows model with dispersion param phi
    ref is two linear orders that are reverses of each other 
    
    wrapper function to call the preflib function gen_mallows with 2 reference rankings
    
    '''
    
    ref = np.random.permutation(range(num_cands))
    ref2 = ref[::-1]

    vote_map = gen_mallows(num_cands, num_voters, [0.5,0.5], [phi,phi], [ref,ref2])
        
    return [vc[0] for vc in vote_map.items()], [vc[1] for vc in vote_map.items()]        



# #####
# SinglePeaked
# #####

# Return a Tuple for a IC-Single Peaked... with alternatives in range 1....range.
def gen_icsp_single_vote(alts):
    a = 0
    b = len(alts)-1
    temp = []
    while a != b:
        if random.randint(0,1) == 1:
            temp.append(alts[a])
            a+= 1
        else:
            temp.append(alts[b])
            b -= 1
    temp.append(alts[a])
    return tuple(temp[::-1]) # reverse


def gen_single_peaked_impartial_culture_strict(nvotes, alts):
    voteset = {}
    for i in range(nvotes):
        tvote = gen_icsp_single_vote(alts)
        voteset[tvote] = voteset.get(tvote, 0) + 1
    return voteset



def create_rankings_single_peaked(num_cands, num_voters, param):
    """create a single-peaked list of rankings
    
    wrapper function to call the preflib function gen_single_peaked_impartial_culture_strict
    """
    
    vote_map = gen_single_peaked_impartial_culture_strict(num_voters, list(range(num_cands)))
    return [vc[0] for vc in vote_map.items()], [vc[1] for vc in vote_map.items()]        

# ##########
# generate profile using the spatial model
# #########
# # TODO: Needs updated

def voter_utility(v_pos, c_pos, beta):
    '''Based on the Rabinowitz and Macdonald (1989) mixed model
    described in Section 3, pp. 745 - 747 of 
    "Voting behavior under the directional spatial model of electoral competition" by S. Merrill III 
    
    beta = 1 is the proximity model
    beta = 0 is the directional model
    '''
    return 2 * np.dot(v_pos, c_pos) - beta*(np.linalg.norm(v_pos)**2 + np.linalg.norm(c_pos)**2)

def create_prof_spatial_model(num_voters, cmap, params):
    num_dim = params[0] # the first component of the parameter is the number of dimensions
    beta = params[1] # used to define the mixed model: beta = 1 is proximity model (i.e., Euclidean distance)
    num_cands = len(cmap.keys())  
    mean = [0] * num_dim # mean is 0 for each dimension
    cov = np.diag([1]*num_dim)  # diagonal covariance
    
    # sample candidate/voter positions using a multivariate normal distribution
    cand_positions = np.random.multivariate_normal(np.array(mean), cov, num_cands)
    voter_positions = np.random.multivariate_normal(np.array(mean), cov, num_voters)
    
    # generate the rankings and counts for each ranking
    ranking_counts = dict()
    for v,v_pos in enumerate(voter_positions):
        v_utils = {voter_utility(v_pos,c_pos,beta): c for c,c_pos in enumerate(cand_positions)}
        ranking = tuple([v_utils[_u] for _u in sorted(v_utils.keys(),reverse=True)])
        if ranking in ranking_counts.keys():
            ranking_counts[ranking] += 1
        else:
            ranking_counts.update({ranking:1})
    
    # list of tuples where first component is a ranking and the second is the count
    prof_counts = ranking_counts.items()
    
    return [rc[0] for rc in prof_counts], [rc[1] for rc in prof_counts]


# #########
# functions to generate profiles
# #########

# dictionary of all the avialble probability models with default parameters
prob_models = {
    "IC": {"func": create_rankings_urn, "param": 0}, # IC model is Mallows with phi=1.0
    "IAC": {"func": create_rankings_urn, "param": 1}, # IAC model is urn with alpha=1
    "MALLOWS": {"func": create_rankings_mallows, "param": 0.8}, 
    "MALLOWS_2REF": {"func": create_rankings_mallows_two_rankings, "param": 0.8}, 
    "URN": {"func": create_rankings_urn, "param": 10},
    "SinglePeaked": {"func": create_rankings_single_peaked, "param": None},
    #"SPATIAL": {"func": create_prof_spatial_model, "param": (3, 1.0)},

}


def generate_profile(num_cands, num_voters, probmod="IC", probmod_param=None):
    '''generate a profile with num_cands candidates and num_voters voters using 
    the probmod probabilistic model (with parameter probmod_param)
    
    Parameters
    ----------
    num_cands: int 
        number of candidates
    num_voters: int
        number of voters
    probmod: str
        name of the probability model to use, default is IC (impartial culture)
        other options are IAC (impartial anonymous culture), URN (urn model with default alpha=10),
        MALLOWS (Mallows with default phi=0.8), MALLOWS_2REF (Mallows with two reference rankings that 
        are reverses of each other and default phi=0.8), SinglePeaked (single peaked profile), and 
        SPATIAL (default is the proximity model with 2 dimensions)
    probmod_param: number
        alternative parameter for the different models (e.g., different dispersion parameter for Mallows)
    '''
    # candidates names must be 0,..., num_cands - 1
    create_rankings = prob_models[probmod]["func"]
    probmod_param = prob_models[probmod]["param"] if  probmod_param is None else probmod_param 
    
    # use preflib tools to generate the rankings
    rankings, rcounts = create_rankings(num_cands, num_voters, probmod_param) 
    
    return Profile(rankings, num_cands, rcounts = rcounts)
