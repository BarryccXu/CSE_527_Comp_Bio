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

def HMM_2_states(sq, initi_prob, trans_prob, emiss_prob):
    base_1 = set(['A', 'T'])
    base_2 = set(['C', 'G'])
    state = ""
    #trace_back = np.zeros([2, len(sq)], dtype = int)
    #initialization
    if(sq[0] in base_1):
        pre_v1 = np.log(initi_prob[0]) + np.log(emiss_prob[0,0])
        pre_v2 = np.log(initi_prob[1]) + np.log(emiss_prob[1,0])
    elif(sq[0] in base_2):
        pre_v1 = np.log(initi_prob[0]) + np.log(emiss_prob[0,1])
        pre_v2 = np.log(initi_prob[1]) + np.log(emiss_prob[1,1])
    if(pre_v1 > pre_v2):
        state = state + '1'
    else:
        state = state + '2'
    cur_v1 = 0; cur_v2 = 0
    for i in range(1, len(sq)):
        list_1 = [pre_v1 + np.log(trans_prob[0,0]), pre_v2 + np.log(trans_prob[1,0])]
        list_2 = [pre_v1 + np.log(trans_prob[0,1]), pre_v2 + np.log(trans_prob[1,1])]
        max_v_to_1 = max(list_1)
        max_v_to_2 = max(list_2)
        if(sq[i] in base_1):
            cur_v1 = np.log(emiss_prob[0,0]) + max_v_to_1
            cur_v2 = np.log(emiss_prob[1,0]) + max_v_to_2
        elif(sq[i] in base_2):
            cur_v1 = np.log(emiss_prob[0,1]) + max_v_to_1
            cur_v2 = np.log(emiss_prob[1,1]) + max_v_to_2
        if(cur_v1 > cur_v2):
            state = state + '1'
        else:
            state = state + '2'
        pre_v1 = cur_v1
        pre_v2 = cur_v2

    return state

def segmentation(sq, state):
    res = dict()
    i = 0
    idx_1 = 0
    idx_2 = 0
    while(i < len(state)):
        if(state[i] == '2'):
            idx_1 = i
            while(state[i] == '2'):
                i += 1
            idx_2 = i
            res[(idx_1, idx_2)] = sq[idx_1 : idx_2]
        else:
            i += 1
    return res

def update_emiss_prob(sq, seg):
    g_c = ""
    for _, v in seg.items():
        g_c += v
    trans_01 = len(seg) / (len(sq) - len(g_c))
    trans_10 = len(seg) / len(g_c)
    table = np.array([[1-trans_01, trans_01], [trans_10, 1-trans_10]])
    return table

def loopy_passing(sq, initi_prob, trans_prob, emiss_prob, iter_time = 10):
    state = HMM_2_states(sq, initi_prob, trans_prob, emiss_prob)
    seg = segmentation(sq, state)
    
    i = 0
    while(i < iter_time):
        state = HMM_2_states(sq, initi_prob, trans_prob, emiss_prob)
        seg = segmentation(sq, state)
        trans_prob = update_emiss_prob(sq, seg)
        i += 1
        print("------------------------------------------")
        print("interate time: ", str(i))
        print("Num of segmentation:", len(seg))
        print("New trans_prob: ", trans_prob)
    return trans_prob
            
    
    
if __name__ == "__main__":
    data_id, data_seq = read_fna("NC_011297.fna")
    sq = data_seq[0]
    initi_prob = np.array([0.996, 0.004])
    trans_prob = np.array([[0.999, 0.001], [0.01, 0.99]])
    emiss_prob = np.array([[0.291, 0.209], [0.169, 0.331]])
    
    table = loopy_passing(sq, initi_prob, trans_prob, emiss_prob, iter_time = 10)
