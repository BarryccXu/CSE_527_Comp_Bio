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
    #state = ""
    trace= np.zeros([2, len(sq)], dtype = int)
    #initialization
    if(sq[0] in base_1):
        pre_v1 = np.log(initi_prob[0]) + np.log(emiss_prob[0,0])
        pre_v2 = np.log(initi_prob[1]) + np.log(emiss_prob[1,0])
    elif(sq[0] in base_2):
        pre_v1 = np.log(initi_prob[0]) + np.log(emiss_prob[0,1])
        pre_v2 = np.log(initi_prob[1]) + np.log(emiss_prob[1,1])
    cur_v1 = 0; cur_v2 = 0
    for i in range(1, len(sq)):
        list_1 = [pre_v1 + np.log(trans_prob[0,0]), pre_v2 + np.log(trans_prob[1,0])]
        list_2 = [pre_v1 + np.log(trans_prob[0,1]), pre_v2 + np.log(trans_prob[1,1])]
        trace[0,i] = np.argmax(list_1)
        trace[1,i] = np.argmax(list_2)
        max_v_to_1 = max(list_1)
        max_v_to_2 = max(list_2)
        if(sq[i] in base_1):
            cur_v1 = np.log(emiss_prob[0,0]) + max_v_to_1
            cur_v2 = np.log(emiss_prob[1,0]) + max_v_to_2
        elif(sq[i] in base_2):
            cur_v1 = np.log(emiss_prob[0,1]) + max_v_to_1
            cur_v2 = np.log(emiss_prob[1,1]) + max_v_to_2
        pre_v1 = cur_v1
        pre_v2 = cur_v2
    
    last_state = 0
    if(cur_v1 > cur_v2):
        last_state = 1
    else:
        last_state = 2

    return trace, last_state

def trace_back(sq, trace, last_state):
    t = 0
    state = [last_state]
    if(last_state == 1):
        t = 0
    elif(last_state == 2):
        t = 1
    for i in range(len(sq) - 1, 0, -1):
        #if(i%10000 == 0):
         #   print(i,'/',len(sq))
        if(t == 0):
            state.append('1')
            t = trace[0,i]
        elif(t == 1):
            state.append('2')
            t = trace[1,i]
    return state[::-1]


def segmentation(sq, state):
    res_1 = dict()
    res_2 = dict()
    i = 0
    idx_1 = 0
    idx_2 = 0
    state.append('3') # a trick to avoid the last one 
    while(i < len(state)):
        if(state[i] == '1'):
            idx_1 = i
            while(state[i] == '1'):
                i += 1
            idx_2 = i
            res_1[(idx_1, idx_2)] = sq[idx_1 : idx_2]
            
        elif(state[i] == '2'):
            idx_1 = i
            while(state[i] == '2'):
                i += 1
            idx_2 = i
            res_2[(idx_1, idx_2)] = sq[idx_1 : idx_2]
        else:
            break
    state.pop()
    return res_1, res_2

def update_emiss_prob(seg_1, seg_2):
    a_t = ""
    g_c = ""
    for _, v in seg_1.items():
        a_t += v
    for _, v in seg_2.items():
        g_c += v
    # it only works for the first and the last segment is in state 1
    trans_01 = len(seg_2) / (len(a_t))
    trans_10 = len(seg_2) / len(g_c)
    table = np.array([[1-trans_01, trans_01], [trans_10, 1-trans_10]])
    return table

def loopy_passing(sq, initi_prob, trans_prob, emiss_prob, iter_time = 10):
    i = 0
    while(i < iter_time):
        trace, last_state = HMM_2_states(sq, initi_prob, trans_prob, emiss_prob)
        state = trace_back(sq, trace, last_state)
        seg_1, seg_2 = segmentation(sq, state)
        trans_prob = update_emiss_prob(seg_1, seg_2)
        i += 1
        print("------------------------------------------")
        print("interate time: ", str(i))
        print("Num of segmentation(A-T):", len(seg_1))
        print("Num of segmentation(C-G):", len(seg_2))
        print("New trans_prob: \n", np.around(trans_prob, decimals = 4))
    return trans_prob, seg_1, seg_2
            
    
    
if __name__ == "__main__":
    data_id, data_seq = read_fna("NC_011297.fna")
    sq = data_seq[0]
    initi_prob = np.array([0.996, 0.004])
    trans_prob = np.array([[0.999, 0.001], [0.01, 0.99]])
    emiss_prob = np.array([[0.291, 0.209], [0.169, 0.331]])
    
    #trace, last_state = HMM_2_states(sq, initi_prob, trans_prob, emiss_prob)
    #state = trace_back(sq, trace, last_state)
    #res_1, res_2 = segmentation(sq, state)
    trans_prob_new, seg_1, seg_2 = loopy_passing(sq, initi_prob, trans_prob, emiss_prob, iter_time = 10)
