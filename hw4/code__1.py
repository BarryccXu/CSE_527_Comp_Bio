from Bio import SeqIO
import numpy as np

def read_fna(file_path):
    data_id = []
    data_seq = []
    for seq_record in SeqIO.parse(file_path, "fasta"):
        tmp_id = seq_record.id
        data_id.append(tmp_id)
        tmp_seq = str(seq_record.seq)
        data_seq.append(tmp_seq)
    return data_id, data_seq

def viterbi(obs, states, start_p, trans_p, emit_p):
    # reference: wikipedia
    V = [{}]
    path = {}
    # Initialize base cases (t == 0)
    for y in states:
        V[0][y] = np.log(start_p[y]) + np.log(emit_p[y][obs[0]])
        path[y] = [y]       
    # Run Viterbi for t > 0
    for t in range(1,len(obs)):
        V.append({})
        newpath = {}
        for y in states:
            (prob, state) = max([(V[t-1][y0] + np.log(trans_p[y0][y]) + np.log(emit_p[y][obs[t]]), y0) for y0 in states])
            V[t][y] = prob
            newpath[y] = path[state] + [y]
        # Don't need to remember the old paths
        path = newpath
    (prob, state) = max([(V[len(obs) - 1][y], y) for y in states])
    return (prob, path[state])

def get_num_states(state):
    

if __name__ == "__main__":
    data_id, data_seq = read_fna("NC_011297.fna")
    sq = data_seq[0]
    states = ('0', '1')
    start_p = {'0': 0.996, '1': 0.004}
    trans_p = {
       '0' : {'0': 0.999, '1': 0.001},
       '1' : {'0': 0.01, '1': 0.99},
       }
    emit_p = {
       '0' : {'A': 0.291, 'T': 0.291, 'G': 0.209, 'C':0.209 },
       '1' : {'A': 0.169, 'T': 0.169, 'G': 0.331, 'C':0.331},
       }
    
    prob, state = viterbi(sq[0:10000], states, start_p, trans_p, emit_p)
