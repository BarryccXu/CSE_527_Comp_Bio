# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 20:57:40 2017

@author: BarryXU
"""
from Bio.SubsMat.MatrixInfo import blosum62
import numpy as np
import random as rd
from Bio import SeqIO
#import seaborn as sns
def NW_gsa(sq_1, sq_2, gap = -4, matrix = blosum62):
    #create an int array 
    F = np.ndarray((1+len(sq_1), 1+len(sq_2)), dtype = int)
    #initialization for first row and first column
    F[0,:] = np.arange(0, gap*(1+len(sq_2)), gap)
    F[:,0] = np.arange(0, gap*(1+len(sq_1)), gap)
    trace = np.copy(F) #this matrix is used to trace back
    #loop for dynamic programming
    for row in range(1, 1+len(sq_1)):
        for col in range(1, 1+len(sq_2)):
            pair = tuple([sq_1[row-1].upper(), sq_2[col-1].upper()])
            #pair = tuple(sorted(pair, reverse=True))
            #whether the pair of protein is in the matrix
            score = 0
            if(pair in matrix):
                score = matrix[pair]
            elif(pair[::-1] in matrix):
                score = matrix[pair[::-1]]
            else:
                score = 0
            #direction[left, left_top_corner, top] 
            direction = [ gap + F[row, col-1], score + F[row-1, col-1], gap + F[row-1, col] ]
            F[row, col] = max(direction)
            trace[row, col] = np.argmax(direction)                   
    return F, F[len(sq_1), len(sq_2)],trace

def trace_back_global(sq_1, sq_2, trace):
    str_1 = ""
    str_2 = ""
    row, col = F.shape
    r = row - 1
    c = col - 1
    while(r > 0 or c > 0):
        if(trace[r,c] == 0):
            str_1 += '_'
            str_2 += sq_2[c-1]
            c -= 1
        elif(trace[r,c] == 1):
            str_1 += sq_1[r-1]
            str_2 += sq_2[c-1]
            r -= 1
            c -= 1
        elif(trace[r,c] == 2):
            str_1 += sq_1[r-1]
            str_2 += '_'
            r -= 1
    #return reversed string
    return str_1[::-1], str_2[::-1]

def trace_back_local(sq_1, sq_2, F, trace):
    str_1 = ""
    str_2 = ""
    #find the max value in the F-matrix
    max_value = np.max(F)
    r, c = np.unravel_index(F.argmax(), F.shape)
    while(r > 0 or c > 0):
        if(F[r,c] <= 0):
            break
        if(trace[r,c] == 0):
            str_1 += '_'
            str_2 += sq_2[c-1]
            c -= 1
        elif(trace[r,c] == 1):
            str_1 += sq_1[r-1]
            str_2 += sq_2[c-1]
            r -= 1
            c -= 1
        elif(trace[r,c] == 2):
            str_1 += sq_1[r-1]
            str_2 += '_'
            r -= 1
    #return reversed string
    return str_1[::-1], str_2[::-1], max_value

def permt(sq_1, sq_2, score, gap = -4, matrix = blosum62, permt_time = 1000):
    '''
    sq_2 will be permuted
    '''
    def p_value(scores, z):
        '''
        inter function for compute p_value
        '''
        mean = np.mean(scores)
        std = np.std(scores)
        norm_z = (z - mean) / std
        p_value = np.exp(-np.power(norm_z, 2) / 2) / np.sqrt(2 * np.pi)
        return p_value
    scores = []
    for i in range(permt_time):
        sq_2_tmp = "".join(rd.sample(sq_2, len(sq_2)))
        _, s, _ = NW_gsa(sq_1, sq_2_tmp, gap, matrix)
        scores.append(s)
    return p_value(scores, score)
    
def read_fasta(file_path):
    data_id = []
    data_seq = []
    for seq_record in SeqIO.parse(file_path, "fasta"):
        tmp_id = (seq_record.id).split('|')
        data_id.append(tmp_id)
        tmp_seq = (seq_record.seq).tostring()
        data_seq.append(tmp_seq)
    return data_id, data_seq

if __name__ == "__main__":
    sq_1 = "deadly"
    sq_2 = "ddgearlyk"
    F, score, trace = NW_gsa(sq_1, sq_2)       
    str_1, str_2 = trace_back_global(sq_1, sq_2, trace)        
    p_value = permt(sq_1, sq_2, score, gap=-4, matrix=blosum62, permt_time=1000)
    data_id, data_seq = read_fasta("data.fasta")
    sq_1_local = "SRGMIEVGNQWT"
    sq_2_local = "RGMVVGRW"
    F, _, trace = NW_gsa(sq_1_local, sq_2_local, matrix = blosum62, gap = -8)
    str_1_local, str_2_local, max_value = trace_back_local(sq_1_local, sq_2_local, F, trace)
    
    
    

