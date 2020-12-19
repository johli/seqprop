from itertools import combinations
from itertools import chain
import numpy as np
import pandas as pd
import numpy as np
from collections import defaultdict
from itertools import combinations
from itertools import chain
import itertools
import pickle
from feature import *
import os
import warnings

def usage2signal(usage,all_fail_prob=0):
    if type(usage)==tuple or type(usage)==list:
        usage=np.array(usage)
    signal=np.zeros_like(usage,dtype=np.float32)
    if np.sum(usage)==0:
        return signal
    usage=usage*(1-all_fail_prob)/np.sum(usage)
    # de-normalization
    for i in range(len(usage)-1,-1,-1):
        prev_fail_prob=np.prod(1-signal[i+1:])
        signal[i]=usage[i]/prev_fail_prob if prev_fail_prob!=0 else np.nan # be careful of np.nan
    return signal
def signal2usage(signal):
    if type(signal)==tuple or type(signal)==list:
        signal=np.array(signal)
    usage=np.zeros_like(signal,dtype=np.float32)
    if np.sum(signal)==0:
        return usage
    for i in range(len(signal)-1,-1,-1):
        usage[i]=np.prod(1-signal[i+1:])*signal[i]
    usage=usage/np.sum(usage) # re-normalization
    return usage,np.prod(1-signal)

def truncate(sequences,final_length):
    truncated=[]
    for i,s in enumerate(sequences):
        if len(s)>=final_length:
            truncated.append(s[:final_length])
        else:
            raise ValueError('sequence at position %d is too short'%(i))
    return truncated

def pad(sequences,final_length,pad_character='N'):
    padded=[]
    if pad_character not in 'NO?':
        raise ValueError('Bad Character: '+pad_character)
    for i,s in enumerate(sequences):
        if len(s)<=final_length:
            left_pad_len=int((final_length-len(s))/2)
            right_pad_len=final_length-len(s)-left_pad_len
            padded.append(pad_character*(left_pad_len)+s+pad_character*(right_pad_len))
        else:
            raise ValueError('sequence at position %d is too long'%(i))
    return padded

def value_fill(processed_sequences,final_length,fill_vector=np.array([.25,.25,.25,.25])):
    N,W,C=processed_sequences.shape
    if final_length>=W:
        return processed_sequences
    left_fill_len=int((W-final_length)/2)
    right_fill_len=W-final_length-left_fill_len
    processed_sequences[:,:left_fill_len,:]=fill_vector.reshape([1,1,C])
    processed_sequences[:,-right_fill_len:,:]=fill_vector.reshape([1,1,C])
    return processed_sequences

def dna_one_hot(s):
    encoding_dict = {'A': np.array([1.0,0.0,0.0,0.0]),
                     'T': np.array([0.0,1.0,0.0,0.0]),
                     'U': np.array([0.0,1.0,0.0,0.0]),
                     'C': np.array([0.0,0.0,1.0,0.0]),
                     'G': np.array([0.0,0.0,0.0,1.0]),
                     'N': np.array([0.25,0.25,0.25,0.25]),
                     'O': np.array([0.0,0.0,0.0,0.0])}
    try:
        np_arr=np.array(list(map(lambda c:encoding_dict[c],s)))
    except KeyError:
        print(s)
        raise
    return np_arr


        

def dna_one_hot_reverse(m):
    s=''
    for n in m:
        if np.allclose(n,[1,0,0,0]):
            s+='A'
        elif np.allclose(n,[0,1,0,0]):
            s+='T'
        elif np.allclose(n,[0,0,1,0]):
            s+='C'
        elif np.allclose(n,[0,0,0,1]):
            s+='G'
        else:
            s+='N'
    return s


def generate_dataset(sequence_table,gene_ids,data_source,processed_sequences1,processed_sequences2=None,min_diff=0.05,gene_of_pas_number='all',filter_illegal_genes=True):
    if data_source in ['bl_usage','sp_usage']:
        warnings.warn("Deprecated data_source option: %s"%data_source,UserWarning)
        X1_indices=[]
        X2_indices=[]
        Y=[]
        for i,gene_id in enumerate(gene_ids):
            print('%5.2f%%'%((i+1)/len(gene_ids)*100),end='\r')
            gene_indices=sequence_table.index[sequence_table.index.str.startswith(gene_id)]
            if type(gene_of_pas_number)==int and len(gene_indices)!=gene_of_pas_number:
                continue
            for pas1,pas2 in combinations(gene_indices,2):
                if np.random.rand(1).squeeze()>0.5:
                    pas1,pas2=pas2,pas1
                diff=sequence_table[data_source][pas1]-sequence_table[data_source][pas2]
                if np.abs(diff)>min_diff:
                    pas_index1=sequence_table.index.get_loc(pas1)
                    pas_index2=sequence_table.index.get_loc(pas2)
                    X1_indices.append(pas_index1)
                    X2_indices.append(pas_index2)
                    Y.append(np.array((1,0)) if diff>0 else np.array((0,1)))
        X1=processed_sequences1[X1_indices,...]
        X2=processed_sequences1[X2_indices,...]
        return(X1,X2,np.array(Y))
    elif data_source in ['bl_signal','sp_signal']:
        warnings.warn("Deprecated data_source option: %s"%data_source,UserWarning)
        X1_indices=[]
        X2_indices=[]
        Y=[]
        for i,gene_id in enumerate(gene_ids):
            print('%5.2f%%'%((i+1)/len(gene_ids)*100),end='\r')
            gene_indices=sequence_table.index[sequence_table.index.str.startswith(gene_id)]
            if type(gene_of_pas_number)==int and len(gene_indices)!=gene_of_pas_number:
                continue
            for pas1,pas2 in combinations(gene_indices,2):
                if np.random.rand(1).squeeze()>0.5:
                    pas1,pas2=pas2,pas1
                diff=sequence_table[data_source][pas1]-sequence_table[data_source][pas2]
                if  ~np.isnan(diff) and np.abs(diff)>min_diff:
                    pas_index1=sequence_table.index.get_loc(pas1)
                    pas_index2=sequence_table.index.get_loc(pas2)
                    X1_indices.append(pas_index1)
                    X2_indices.append(pas_index2)
                    Y.append(np.array((1,0)) if diff>0 else np.array((0,1)))
        X1=processed_sequences1[X1_indices,...]
        X2=processed_sequences1[X2_indices,...]
        return (X1,X2,np.array(Y))
    elif data_source=='comparison':
        warnings.warn("Deprecated data_source option: %s"%data_source,UserWarning)
        indices=[]
        Y=[]
        for i, gene_id in enumerate(gene_ids):
            print('%5.2f%%'%((i+1)/len(gene_ids)*100),end='\r')
            pas_indices=sequence_table.index[sequence_table.index.str.startswith(gene_id)]
            if type(gene_of_pas_number)==int and len(pas_indices)!=gene_of_pas_number:
                continue
            for pas in pas_indices:
                diff=sequence_table['bl_usage'][pas]-sequence_table['sp_usage'][pas]
                if np.abs(diff)>min_diff:
                    index=sequence_table.index.get_loc(pas)
                    indices.append(index)
                    Y.append(np.array((1,0)) if diff>0 else np.array((0,1)))
        X1=processed_sequences1[indices,...]
        X2=processed_sequences2[indices,...]
        return (X1,X2,np.array(Y))
    elif data_source in ('bl_gene_usage','sp_gene_usage'):
        if data_source=='bl_gene_usage':
            column_usage='bl_usage'
            column_signal="bl_signal"
        elif data_source=='sp_gene_usage':
            column_usage='sp_usage'
            column_signal="sp_signal"
        X_indices=[]
        Y_usage=[]
        Y_signal=[]
        pas_numbers=[]
        for i,gene_id in enumerate(gene_ids):  
            print('%5.2f%%'%((i+1)/len(gene_ids)*100),end='\r')
            table_int_indices=sequence_table.index.str.startswith(gene_id).nonzero()[0]
            table_indices=sequence_table.index[table_int_indices]
            if type(gene_of_pas_number)==int and len(table_indices)!=gene_of_pas_number:
                continue
            usages=sequence_table[column_usage].loc[table_indices].values
            signals=sequence_table[column_signal].loc[table_indices].values
            if not filter_illegal_genes or np.abs(sum(usages)-1)<0.1: # exclude those genes that does not express
                X_indices.append(table_int_indices)
                pas_numbers.append(len(table_int_indices))
                Y_usage.append(usages)
                Y_signal.append(signals)
        X_indices=np.hstack(X_indices)
        X_indices_original=sequence_table.index[X_indices]
        X=processed_sequences1[X_indices,...]
        Y_usage=np.hstack(Y_usage).astype(np.float32)
        Y_signal=np.hstack(Y_signal).astype(np.float32)
        pas_numbers=np.array(pas_numbers)
        return (X,X_indices_original,Y_usage,Y_signal,pas_numbers)
            # Possiblly the pas should be randomly shuffled
    elif data_source in ('bl_gene_comparison','sp_gene_comparison'):
        if data_source=='bl_gene_comparison':
            column_usage='bl_usage'
            column_signal="bl_signal"
        elif data_source=='sp_gene_comparison':
            column_usage='sp_usage'
            column_signal="sp_signal"
        X_indices=[]
        Y_comparison_usage=[]
        Y_comparison_signal=[]
        pas_numbers=[]
        for i,gene_id in enumerate(gene_ids):
            print('%5.2f%%'%((i+1)/len(gene_ids)*100),end='\r')
            table_int_indices=sequence_table.index.str.startswith(gene_id).nonzero()[0]
            usages=sequence_table[column_usage].iloc[table_int_indices].values
            signals=sequence_table[column_signal].iloc[table_int_indices].values
        
            if type(gene_of_pas_number)==int and len(usages)!=gene_of_pas_number:
                continue
            if not filter_illegal_genes or np.abs(sum(usages)-1)<0.1: # exclude those genes that does not express
                X_indices.append(table_int_indices)
                pas_numbers.append(len(table_int_indices))
                Y_comparison_usage.append(usages)
                Y_comparison_signal.append(signals)
        X_indices=np.hstack(X_indices)
        X_indices_original=sequence_table.index[X_indices]
        X_comparison=processed_sequences1[X_indices,...]
        Y_comparison_usage=np.hstack(Y_comparison_usage).astype(np.float32)
        Y_comparison_signal=np.hstack(Y_comparison_signal).astype(np.float32)
        pas_numbers=np.array(pas_numbers)
        pas_numbers_cumsum=np.concatenate([[0],pas_numbers.cumsum()])
        X1_indices_comparison=[]
        X2_indices_comparison=[]
        Y_comparison=[]
        for i,pas_num in enumerate(pas_numbers):
            for pas1,pas2 in combinations(np.arange(pas_numbers_cumsum[i],pas_numbers_cumsum[i+1]),2):
                if np.random.rand()>0.5:
                    pas1,pas2=pas2,pas1
                diff=Y_comparison_usage[pas1]-Y_comparison_usage[pas2]
                if np.abs(diff)>min_diff:
                    X1_indices_comparison.append(pas1)
                    X2_indices_comparison.append(pas2)
                    Y_comparison.append(np.array((1,0)) if diff>0 else np.array((0,1)))
        Y_comparison=np.array(Y_comparison)
        X1_indices_comparison=np.array(X1_indices_comparison)
        X2_indices_comparison=np.array(X2_indices_comparison)
        return X_comparison,X_indices_original,Y_comparison,Y_comparison_usage,Y_comparison_signal,X1_indices_comparison,X2_indices_comparison,pas_numbers
    else:
        raise ValueError('unknown data_source')
def prepare_data_dict(sequence_table,train_range,test_range,bl_processed,sp_processed,gene_ids,gene_of_pas_number_list,strain):
    warnings.warn("Deprecated function",UserWarning)
    assert strain in ['sp','bl']
    if strain=='bl':
        processed=bl_processed
    elif strain=='sp':
        processed=sp_processed
    train_gene_ids=[gene_ids[i] for i in train_range]
    test_gene_ids=[gene_ids[i] for i in test_range]
    X_train,Y_train,pas_numbers_train=generate_dataset(sequence_table,train_gene_ids,'%s_gene_usage'%(strain),processed)
    X_dev,Y_dev,pas_numbers_dev=generate_dataset(sequence_table,test_gene_ids,'%s_gene_usage'%(strain),processed)
    X_comparison_train,Y_comparison_train,X1_indices_comparison_train,X2_indices_comparison_train,pas_numbers_train=generate_dataset(sequence_table,train_gene_ids,'%s_gene_comparison'%(strain),processed,min_diff=0.15)
    X_comparison_dev,Y_comparison_dev,X1_indices_comparison_dev,X2_indices_comparison_dev,pas_numbers_dev=generate_dataset(sequence_table,test_gene_ids,'%s_gene_comparison'%(strain),processed,min_diff=0.15)
    X1_comparison_train,X2_comparison_train,Y_comparison_train_2=generate_dataset(sequence_table,train_gene_ids,'%s_usage'%(strain),processed,min_diff=0.15)
    X1_comparison_dev,X2_comparison_dev,Y_comparison_dev_2=generate_dataset(sequence_table,test_gene_ids,'%s_usage'%(strain),processed,min_diff=0.15)

    data_dict={'X_train':X_train,'Y_train':Y_train,'pas_numbers_train':pas_numbers_train,
               'X_dev':X_dev,'Y_dev':Y_dev,'pas_numbers_dev':pas_numbers_dev,
               'X1_indices_comparison_train':X1_indices_comparison_train,'X2_indices_comparison_train':X2_indices_comparison_train,'Y_comparison_train':Y_comparison_train,
               'X_comparison_train':X_comparison_train,'pas_numbers_train':pas_numbers_train,
               'X1_indices_comparison_dev':X1_indices_comparison_dev,'X2_indices_comparison_dev':X2_indices_comparison_dev,'Y_comparison_dev':Y_comparison_dev,
               'X_comparison_dev':X_comparison_dev,'pas_numbers_dev':pas_numbers_dev
               }
    data_dict_frey={'X_train':X_train,'Y_train':Y_train,'pas_numbers_train':pas_numbers_train,
               'X_dev':X_dev,'Y_dev':Y_dev,'pas_numbers_dev':pas_numbers_dev,
               'X1_comparison_train':X1_comparison_train,'X2_comparison_train':X2_comparison_train,'Y_comparison_train':Y_comparison_train_2,
               'X1_comparison_dev':X1_comparison_dev,'X2_comparison_dev':X2_comparison_dev,'Y_comparison_dev':Y_comparison_dev_2}
    for pas_number in gene_of_pas_number_list:
        X_dev_pas_number,Y_dev_pas_number,pas_numbers_dev_pas_number=generate_dataset(sequence_table,test_gene_ids,'%s_gene_usage'%(strain),processed,gene_of_pas_number=pas_number)
        data_dict.update({'X_dev_%dpas'%(pas_number):X_dev_pas_number,'Y_dev_%dpas'%(pas_number):Y_dev_pas_number,'pas_numbers_dev_%dpas'%(pas_number):pas_numbers_dev_pas_number})
        data_dict_frey.update({'X_dev_%dpas'%(pas_number):X_dev_pas_number,'Y_dev_%dpas'%(pas_number):Y_dev_pas_number,'pas_numbers_dev_%dpas'%(pas_number):pas_numbers_dev_pas_number})
    return data_dict,data_dict_frey
# TODO: the way to control filter_genes should be changed
def prepare_data_dict_onefold(sequence_table,gene_range,bl_processed,sp_processed,gene_ids,gene_of_pas_number_list,strain,filter_genes):
    if strain=='bl':
        processed=bl_processed
    elif strain=='sp':
        processed=sp_processed
    else:
        assert False
    fold_gene_ids=[gene_ids[i] for i in gene_range]
    print("\nbatch")
    X,X_indices_original,Y_usage,Y_signal,pas_numbers=generate_dataset(sequence_table,fold_gene_ids,'%s_gene_usage'%(strain),processed,filter_illegal_genes=filter_genes)
    print()
    print("comparison")
    X_comparison,X_comparison_indices_original,Y_comparison,Y_comparison_usage,Y_comparison_signal,X1_indices_comparison,X2_indices_comparison,pas_numbers_comparison=generate_dataset(sequence_table,fold_gene_ids,'%s_gene_comparison'%(strain),processed,min_diff=0.15)
    print()
    data_dict={'X':X,'X_indices_original':X_indices_original,'Y_usage':Y_usage,"Y_signal":Y_signal,'pas_numbers':pas_numbers,
               'X1_indices_comparison':X1_indices_comparison,'X2_indices_comparison':X2_indices_comparison,"Y_comparison":Y_comparison,'Y_comparison_usage':Y_comparison_usage,"Y_comparison_signal":Y_comparison_signal,
               'X_comparison':X_comparison,'X_comparison_indices_original':X_comparison_indices_original, 'X1_comparison':X_comparison[X1_indices_comparison],'X2_comparison':X_comparison[X2_indices_comparison],
               'pas_numbers_comparison':pas_numbers_comparison
              }
    for pas_number in gene_of_pas_number_list:
        print("\n%d pas"%(pas_number))
        X_pas_number,X_indices_original_pas_number,Y_usage_pas_number,Y_signal_pas_number,pas_numbers_pas_number=generate_dataset(sequence_table,fold_gene_ids,'%s_gene_usage'%(strain),processed,gene_of_pas_number=pas_number)
        data_dict.update({'X_%dpas'%(pas_number):X_pas_number,'Y_usage_%dpas'%(pas_number):Y_usage_pas_number,"Y_signal_%dpas"%(pas_number):Y_signal_pas_number,'pas_numbers_%dpas'%(pas_number):pas_numbers_pas_number,
        'X_indices_original_%dpas'%(pas_number):X_indices_original_pas_number})
        
    return data_dict

def prepare_fold_files(data_source,generation,strain,attributes,num_folds,fold_size,pas_numbers_list,base_dir,final_length=None,filter_genes=True):
    # fold_size=1651
    # pas_numbers_list=[2,3,4]
    # TODO: sequence length problem
    assert generation in ['parental','f1'] and strain in ['bl','sp'] and attributes in ['sequences','features']
    sequence_table,bl_processed,sp_processed,gene_ids=data_source['%s_sequence_table'%(generation)],data_source['%s_bl_processed_%s'%(generation,attributes)],data_source['%s_sp_processed_%s'%(generation,attributes)],data_source['%s_gene_ids'%(generation)]
    if final_length is not None:
        bl_processed=value_fill(bl_processed.copy(),final_length)
        sp_processed=value_fill(sp_processed.copy(),final_length)
    fold_nums=[]
    data_dict_folds=[]
    for i in range(num_folds):
        start=fold_size*i
        end=min(fold_size*(i+1),len(gene_ids))
        fold_nums.append(range(start,end))
    for i in range(num_folds):
        print("fold %d"%(i))
        data_dict_fold=prepare_data_dict_onefold(sequence_table,fold_nums[i],bl_processed,sp_processed,gene_ids,pas_numbers_list,strain,filter_genes)
        print()
        data_dict_folds.append(data_dict_fold)
    if final_length is None:
        file_name='data_dict_folds-%s-%s-%s.pkl'%(generation,strain,attributes)
    else:
        file_name='data_dict_folds-%s-%s-%s-%dnt.pkl'%(generation,strain,attributes,final_length)
        
    file_name=os.path.join(base_dir,file_name)
    with open(file_name,'wb') as f:
        pickle.dump(data_dict_folds,f)

def sequence_process(sequences,length):
    sequences=pad(sequences,length)
    processed_sequences=np.array([dna_one_hot(s) for s in sequences],dtype=np.float32)
    return processed_sequences

def feature_process(sequences,length):
    sequences=pad(sequences,length)
    features=[]
    for i,s in enumerate(sequences):
        print('%5.2f%%'%(i/len(sequences)*100),end='\r')
        features.append(get_all_features(s))
    processed_features=np.array(features,dtype=np.float32)
    return processed_features
def blsp_combine(parental_data_dict_BL,parental_data_dict_SP):
    parental_data_dict_combined={}
    for k in parental_data_dict_BL:
        if 'indices_comparison' in k:
            if 'train' in k:
                parental_data_dict_combined[k]=np.concatenate([parental_data_dict_BL[k],len(parental_data_dict_BL['X_comparison_train'])+parental_data_dict_SP[k]],axis=0)
            elif 'dev' in k:
                parental_data_dict_combined[k]=np.concatenate([parental_data_dict_BL[k],len(parental_data_dict_BL['X_comparison_dev'])+parental_data_dict_SP[k]],axis=0)
        else:
            parental_data_dict_combined[k]=np.concatenate([parental_data_dict_BL[k],parental_data_dict_SP[k]],axis=0)
    return parental_data_dict_combined
def blsp_folds_combine(BL_folds,SP_folds):
    assert len(BL_folds)==len(SP_folds)
    combined_folds=[]
    for i in range(len(BL_folds)):
        d=dict()
        for k in BL_folds[i]:
            if 'indices_comparison' in k:
                d[k]=np.concatenate([BL_folds[i][k],len(BL_folds[i]['X_comparison'])+SP_folds[i][k]],axis=0)
            else:
                d[k]=np.concatenate([BL_folds[i][k],SP_folds[i][k]],axis=0)
        combined_folds.append(d)
    return combined_folds
def fold_combine(folds,suffix):
    fold_combined={}
    assert len(folds)>0
    pas_numbers=set()
    for k in folds[0].keys():
        if 'indices_comparison' in k:
            concat_list=[]
            base_number=0
            for i in range(len(folds)):
                concat_list.append(folds[i][k]+base_number)
                base_number+=len(folds[i]['X'])
            fold_combined[k+'_%s'%(suffix)]=np.concatenate(concat_list,axis=0)
        else:
            match=re.match(r'(X|Y_usage|Y_signal|pas_numbers)_(\d+pas)',k)
            if match:
                pas_numbers.add(match.group(2))
                fold_combined['%s_%s_%s'%(match.group(1),suffix,match.group(2))]=np.concatenate([folds[i][k] for i in range(len(folds))])
            else:
                fold_combined[k+'_%s'%(suffix)]=np.concatenate([folds[i][k] for i in range(len(folds))])
    pas_numbers.add("all")
    # TODO: build max_pred dict
    for pas_number in pas_numbers:
        if pas_number=="all":
            X_key="X"+"_%s"%(suffix)
            Y_key="Y_usage"+"_%s"%(suffix)
            pas_numbers_key="pas_numbers"+"_%s"%(suffix)
            X_max_pred,Y_max_pred,pas_numbers_max_pred=filter_genes_max_pred(fold_combined[X_key],fold_combined[Y_key],fold_combined[pas_numbers_key])
            fold_combined["X_max_pred_%s"%(suffix)]=X_max_pred
            fold_combined["Y_usage_max_pred_%s"%(suffix)]=Y_max_pred
            fold_combined["pas_numbers_max_pred_%s"%(suffix)]=pas_numbers_max_pred
        else:
            X_key="X_%s_%s"%(suffix,pas_number)
            Y_key="Y_usage_%s_%s"%(suffix,pas_number)
            pas_numbers_key="pas_numbers_%s_%s"%(suffix,pas_number)
            X_max_pred,Y_max_pred,pas_numbers_max_pred=filter_genes_max_pred(fold_combined[X_key],fold_combined[Y_key],fold_combined[pas_numbers_key])
            fold_combined["X_max_pred_%s_%s"%(suffix,pas_number)]=X_max_pred
            fold_combined["Y_usage_max_pred_%s_%s"%(suffix,pas_number)]=Y_max_pred
            fold_combined["pas_numbers_max_pred_%s_%s"%(suffix,pas_number)]=pas_numbers_max_pred
    return fold_combined
def filter_genes_max_pred(X,Y,pas_numbers,min_diff=0.15):
    assert len(X)==len(Y)==np.sum(pas_numbers)
    start_idx=0
    xy_indices=[]
    pas_numbers_indices=[]
    for i,pas_number in enumerate(pas_numbers):
        gene_usage=Y[start_idx:start_idx+pas_number]
        argsort=np.argsort(gene_usage)
        if gene_usage[argsort[-1]]-gene_usage[argsort[-2]]>=0.15:
            xy_indices.append(list(range(start_idx,start_idx+pas_number)))
            pas_numbers_indices.append(i)
        start_idx+=pas_number
    xy_indices=np.hstack(xy_indices)
    return X[xy_indices],Y[xy_indices],pas_numbers[pas_numbers_indices]
def get_usage_dstatistics(X,Y,pas_numbers):
    ranges=[]
    largest_two_diff=[]
    start_idx=0
    for i,pas_number in enumerate(pas_numbers):
        usages=Y[start_idx:start_idx+pas_numbers[i]]
        ra=usages.ptp()
        argsort=usages.argsort()
        ltd=usages[argsort[-1]]-usages[argsort[-2]]
        ranges.append(ra)
        largest_two_diff.append(ltd)
        start_idx+=pas_numbers[i]
    return dict(ranges=np.array(ranges),largest_two_diff=np.array(largest_two_diff))
def get_zhihao_data(dataset,base_dir):
    if dataset=="dragon":
        MOTIF_VARIANTS = [
                    'AATAAA',
                    'ATTAAA',
                    'AAAAAG',
                    'AAGAAA',
                    'TATAAA',
                    'AATACA',
                    'AGTAAA',
                    'ACTAAA',
                    'GATAAA',
                    'CATAAA',
                    'AATATA',
                    'AATAGA'
                    ]
        foldset=list(range(1,6))
        labelset=["pos","neg"]

        string_sequences=[]
        labels=[]
        motifs=[]
        fold_nums=[]
        original_file_names=[]

        for motif in MOTIF_VARIANTS:
            for fold in foldset:
                for label in labelset:
                    if label=="pos":
                        folder_name="positive5fold"
                        file_name=os.path.join(base_dir,folder_name,"%s_fold_%d.txt"%(motif,fold))
                        short_file_name=os.path.join(folder_name,"%s_fold_%d.txt"%(motif,fold))
                    elif label=="neg":
                        folder_name="negatives5fold"
                        file_name=os.path.join(base_dir,folder_name,"neg%s_fold_%d.txt"%(motif,fold))
                        short_file_name=os.path.join(folder_name,"neg%s_fold_%d.txt"%(motif,fold))
                    else:
                        assert False
                    with open(file_name) as f:
                        for line in f:
                            line=line.rstrip()
                            string_sequences.append(line)
                            labels.append(label)
                            motifs.append(motif)
                            fold_nums.append(fold)
                            original_file_names.append(short_file_name)
        data_frame=pd.DataFrame({"sequence":string_sequences,"label":labels,"motif":motifs,"fold":fold_nums,"filename":original_file_names})
        return data_frame
    elif dataset=="omni":
        MOTIF_VARIANTS = [
                    'AATAAA',
                    'ATTAAA',
                    'AAAAAG',
                    'AAGAAA',
                    'TATAAA',
                    'AATACA',
                    'AGTAAA',
                    'ACTAAA',
                    'GATAAA',
                    'CATAAA',
                    'AATATA',
                    'AATAGA'
                    ]
        labelset=["pos","neg"]

        string_sequences=[]
        labels=[]
        motifs=[]
        original_file_names=[]

        for motif in MOTIF_VARIANTS:
            for label in labelset:
                if label=="pos":
                    folder_name="positive"
                    file_name=os.path.join(base_dir,folder_name,"%s.txt"%(motif))
                    short_file_name=os.path.join(folder_name,"%s.txt"%(motif))
                elif label=="neg":
                    folder_name="negative"
                    file_name=os.path.join(base_dir,folder_name,"%s.txt"%(motif))
                    short_file_name=os.path.join(folder_name,"%s.txt"%(motif))
                else:
                    assert False
                with open(file_name) as f:
                    for line in f:
                        line=line.rstrip()
                        string_sequences.append(line)
                        labels.append(label)
                        motifs.append(motif)
                        original_file_names.append(short_file_name)
        data_frame=pd.DataFrame({"sequence":string_sequences,"label":labels,"motif":motifs,"filename":original_file_names})
        return data_frame        

    else:
        assert False
def restore_signal2usage(data_dict):
    import re
    from copy import copy
    data_dict=copy(data_dict)
    for k in data_dict:
        mat=re.search(r'(.*)signal(.*)',k)
        if mat:
            usage_k=mat.group(1)+"usage"+mat.group(2)
            data_dict[k]=data_dict[usage_k]
    return data_dict
def map_index(pas):
    gene,pas_number=pas.split(':')
    pas_number=int(pas_number)
    assert pas_number<100
    new_index="%s:%02d"%(gene,pas_number)
    return new_index
 
