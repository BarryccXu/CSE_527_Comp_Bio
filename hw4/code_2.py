import numpy as np
import random
from Bio.SubsMat.MatrixInfo import blosum62
from Bio import SeqIO

def get_full_blosum62(matrix = blosum62):
    #extend blosum matrix    
    mat = dict()
    for k,v in matrix.items():
        mat[k] = matrix[k]
        if(k[0] != k[1]):
            mat[k[::-1]] = matrix[k]
    return mat

def nw_global_sequence_alignment(seq1, seq2, mat, gap = -4):
    F = np.ndarray((1+len(seq1), 1+len(seq2)), dtype = int)
    #dynamic matrix initialization
    F[:,0] = np.arange(0, (1+len(seq1))*gap, gap)
    F[0,:] = np.arange(0, (1+len(seq2))*gap, gap)
    tb = np.copy(F)
    for r in range(1, 1+len(seq1)):
        for c in range(1, 1+len(seq2)):
            key = (seq1[r-1].upper(), seq2[c-1].upper())
            s = 0
            if(key in mat):
                s = mat[key]
            #direction[left, left_top_corner, top] 
            d = [ F[r, c-1] + gap, F[r-1, c-1] + s, F[r-1, c] + gap ]
            F[r, c] = np.max(d)
            tb[r, c] = np.argmax(d)                   
    return F, tb, F[len(seq1), len(seq2)]

def trace_back(seq_1, seq_2, tb):
    #create stack to store two sequences
    stk1 = []; stk2 = []
    for s in seq_1:
        stk1.append(s)
    for s in seq_2:
        stk2.append(s)
    r, c = F.shape
    r = r - 1
    c = c - 1
    s1 = ""; s2 = ""
    while(r > 0 or c > 0):
        if(tb[r,c] == 0):
            s1 += '_'
            s2 += stk2.pop()
            c -= 1
        elif(tb[r,c] == 1):
            s1 += stk1.pop()
            s2 += stk2.pop()
            r -= 1
            c -= 1
        elif(tb[r,c] == 2):
            s1 += stk1.pop()
            s2 += '_'
            r -= 1
    #reverse string
    s1 = s1[::-1]; s2 = s2[::-1]
    return s1, s2

def compute_p_value(s, z):
    # compute p-value
    norm = (z - np.mean(s)) / np.std(s)
    p = np.exp(- np.power(norm, 2) / 2) / np.sqrt(2 * np.pi)
    return p
    
def permutation(seq1, seq2, score, mat, gap = -4, count = 1000):
    score_list = list()
    for i in range(count):
        shuffled = list(seq2)
        random.shuffle(shuffled)
        shuffled = ''.join(shuffled)
        _, _, s = nw_global_sequence_alignment(seq1, shuffled, mat, gap)
        score_list.append(s)
    p = compute_p_value(score_list, score)
    return p
    
def load_data(path):
    ids = list()
    seqs = list()
    for seq_record in SeqIO.parse(path, "fasta"):
        ID = (seq_record.id).split('|')
        ids.append(ID)
        SEQ = (seq_record.seq).tostring()
        seqs.append(SEQ)
    return ids, seqs

def sw_local_sequence_alignment(seq1, seq2, mat, gap = -8):
    F = np.ndarray((1+len(seq1), 1+len(seq2)), dtype = int)
    #dynamic matrix initialization
    F[0,:] = np.zeros(len(F[0,:]))
    F[:,0] = np.zeros(len(F[:,0]))
    tb = np.copy(F)
    for r in range(1, 1+len(seq1)):
        for c in range(1, 1+len(seq2)):
            key = (seq1[r-1].upper(), seq2[c-1].upper())
            s = 0
            if(key in mat):
                s = mat[key]
            #direction[left, left_top_corner, top] 
            d = [ F[r, c-1] + gap, F[r-1, c-1] + s, F[r-1, c] + gap ]
            F[r, c] = np.max(np.max(d), 0)
            tb[r, c] = np.argmax(d)                   
    return F, tb, F[len(seq1), len(seq2)]
	
def trace_back_local(seq1, seq2, F, tb):
    s1 = ""; s2 = ""
    #get max value
    maxvalue = np.max(F)
    r, c = np.unravel_index(F.argmax(), F.shape)
    while(r > 0 or c > 0):
        if(F[r,c] <= 0):
            break
        if(tb[r,c] == 0):
            s1 += '_'
            s2 += seq2[c-1]
            c -= 1
        elif(tb[r,c] == 1):
            s1 += seq1[r-1]
            s2 += seq2[c-1]
            r -= 1
            c -= 1
        elif(tb[r,c] == 2):
            s1 += seq1[r-1]
            s2 += '_'
            r -= 1
    #reverse string
    s1 = s1[::-1]; s2 = s2[::-1]
    return s1, s2, maxvalue

if __name__ == "__main__":
    mat = get_full_blosum62()
    F, tb, score = nw_global_sequence_alignment("deadly", "ddgearlyk", mat) 
    print("F matrix: \n", F) 
    print("score: \n", score)     
    s1, s2 = trace_back("deadly", "ddgearlyk", tb) 
    print("Two sequences: \n")
    print(s1)
    print(s2)
    p = permutation("deadly", "ddgearlyk", score, mat, gap=-4, count=1000)
    print("P_value: ", p)
    F, tb, score = sw_local_sequence_alignment("SRGMIEVGNQWT", "RGMVVGRW", mat, gap=-8)
    s1, s2, maxvalue = trace_back_local("SRGMIEVGNQWT", "RGMVVGRW", F, tb)
    print("Best alignment: \n")
    print(s1)
    print(s2)    
    print("score: \n", score) 
    
    #read data
    ids, seqs = load_data("data.fasta")   
    score_list = list()
    #compute scores
    for i in range(1, len(seqs)):
        _, _, score = nw_global_sequence_alignment(seqs[0], seqs[i], mat)
        score_list.append(score)
    print(score_list)
    
    p_list = list()
    for i in range(1, len(seqs)):
        p = permutation(seqs[0], seqs[i], score_list[i-1], count = 1000)
        p_list.append(p)
    print(p_list)
    
    
    



