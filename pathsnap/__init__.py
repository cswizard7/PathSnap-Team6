"""PathSnap: map-matching of noisy GPS trajectories onto a road network.

Pipeline: raw trip files -> local metric projection -> candidate road segments
-> candidate features -> classifier probabilities -> HMM/Viterbi decoding.
"""
