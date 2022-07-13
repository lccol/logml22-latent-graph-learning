import numpy as np
import pandas as pd
from pathlib import Path
from scipy.ndimage import zoom
from tqdm.auto import tqdm

try:
    import pickle5 as pickle
except ImportError as e:
    import pickle
from skimage import transform
import multiprocessing as mp

channel_stoi_default = {"i": 0, "ii": 1, "v1":2, "v2":3, "v3":4, "v4":5, "v5":6, "v6":7, "iii":8, "avr":9, "avl":10, "avf":11, "vx":12, "vy":13, "vz":14}

def stratify(data, classes, ratios, samples_per_group=None):
    """Stratifying procedure. Modified from https://vict0rs.ch/2018/05/24/sample-multilabel-dataset/ (based on Sechidis 2011)
    data is a list of lists: a list of labels, for each sample (possibly containing duplicates not multi-hot encoded).
    
    classes is the list of classes each label can take
    ratios is a list, summing to 1, of how the dataset should be split
    samples_per_group: list with number of samples per patient/group
    """
    np.random.seed(0) # fix the random seed

    # data is now always a list of lists; len(data) is the number of patients; data[i] is the list of all labels for patient i (possibly multiple identical entries)

    if(samples_per_group is None):
        samples_per_group = np.ones(len(data))
        
    #size is the number of ecgs
    size = np.sum(samples_per_group)

    # Organize data per label: for each label l, per_label_data[l] contains the list of patients
    # in data which have this label (potentially multiple identical entries)
    per_label_data = {c: [] for c in classes}
    for i, d in enumerate(data):
        for l in d:
            per_label_data[l].append(i)

    # In order not to compute lengths each time, they are tracked here.
    subset_sizes = [r * size for r in ratios] #list of subset_sizes in terms of ecgs
    per_label_subset_sizes = { c: [r * len(per_label_data[c]) for r in ratios] for c in classes } #dictionary with label: list of subset sizes in terms of patients

    # For each subset we want, the set of sample-ids which should end up in it
    stratified_data_ids = [set() for _ in range(len(ratios))] #initialize empty

    # For each sample in the data set
    print("Starting fold distribution...")
    size_prev=size+1 #just for output
    while size > 0:
        if(int(size_prev/1000) > int(size/1000)):
            print("Remaining entries to distribute:",size,"non-empty labels:", np.sum([1 for l, label_data in per_label_data.items() if len(label_data)>0]))
        size_prev=size
        # Compute |Di| 
        lengths = {
            l: len(label_data)
            for l, label_data in per_label_data.items()
        } #dictionary label: number of ecgs with this label that have not been assigned to a fold yet
        try:
            # Find label of smallest |Di|
            label = min({k: v for k, v in lengths.items() if v > 0}, key=lengths.get)
        except ValueError:
            # If the dictionary in `min` is empty we get a Value Error. 
            # This can happen if there are unlabeled samples.
            # In this case, `size` would be > 0 but only samples without label would remain.
            # "No label" could be a class in itself: it's up to you to format your data accordingly.
            break
        # For each patient with label `label` get patient and corresponding counts
        unique_samples, unique_counts = np.unique(per_label_data[label],return_counts=True)
        idxs_sorted = np.argsort(unique_counts, kind='stable')[::-1]
        unique_samples = unique_samples[idxs_sorted] # this is a list of all patient ids with this label sort by size descending
        unique_counts =  unique_counts[idxs_sorted] # these are the corresponding counts
        
        # loop through all patient ids with this label
        for current_id, current_count in zip(unique_samples,unique_counts):
            
            subset_sizes_for_label = per_label_subset_sizes[label] #current subset sizes for the chosen label

            # Find argmax clj i.e. subset in greatest need of the current label
            largest_subsets = np.argwhere(subset_sizes_for_label == np.amax(subset_sizes_for_label)).flatten()
            
            # if there is a single best choice: assign it
            if len(largest_subsets) == 1:
                subset = largest_subsets[0]
            # If there is more than one such subset, find the one in greatest need of any label
            else:
                largest_subsets2 = np.argwhere(np.array(subset_sizes)[largest_subsets] == np.amax(np.array(subset_sizes)[largest_subsets])).flatten()
                subset = largest_subsets[np.random.choice(largest_subsets2)]

            # Store the sample's id in the selected subset
            stratified_data_ids[subset].add(current_id)

            # There is current_count fewer samples to distribute
            size -= samples_per_group[current_id]
            # The selected subset needs current_count fewer samples
            subset_sizes[subset] -= samples_per_group[current_id]

            # In the selected subset, there is one more example for each label
            # the current sample has
            for l in data[current_id]:
                per_label_subset_sizes[l][subset] -= 1
               
            # Remove the sample from the dataset, meaning from all per_label dataset created
            for x in per_label_data.keys():
                per_label_data[x] = [y for y in per_label_data[x] if y!=current_id]
              
    # Create the stratified dataset as a list of subsets, each containing the orginal labels
    stratified_data_ids = [sorted(strat) for strat in stratified_data_ids]
    #stratified_data = [
    #    [data[i] for i in strat] for strat in stratified_data_ids
    #]

    # Return both the stratified indexes, to be used to sample the `features` associated with your labels
    # And the stratified labels dataset

    #return stratified_data_ids, stratified_data
    return stratified_data_ids

def resample_data(sigbufs, channel_labels, fs, target_fs, channels=8, channel_stoi=None,skimage_transform=True,interpolation_order=3):
    channel_labels = [c.lower() for c in channel_labels]
    #https://github.com/scipy/scipy/issues/7324 zoom issues
    factor = target_fs/fs
    timesteps_new = int(len(sigbufs)*factor)
    if(channel_stoi is not None):
        data = np.zeros((timesteps_new, channels), dtype=np.float32)
        for i,cl in enumerate(channel_labels):
            if(cl in channel_stoi.keys() and channel_stoi[cl]<channels):
                if(skimage_transform):
                    data[:,channel_stoi[cl]]=transform.resize(sigbufs[:,i],(timesteps_new,),order=interpolation_order).astype(np.float32)
                else:
                    data[:,channel_stoi[cl]]=zoom(sigbufs[:,i],timesteps_new/len(sigbufs),order=interpolation_order).astype(np.float32)
    else:
        if(skimage_transform):
            data=transform.resize(sigbufs,(timesteps_new,channels),order=interpolation_order).astype(np.float32)
        else:
            data=zoom(sigbufs,(timesteps_new/len(sigbufs),1),order=interpolation_order).astype(np.float32)
    return data

def load_dataset(target_root,filename_postfix="",df_mapped=True):
    target_root = Path(target_root)
    # if(df_mapped):
    #     df = pd.read_pickle(target_root/("df_memmap"+filename_postfix+".pkl"))
    # else:
    #     df = pd.read_pickle(target_root/("df"+filename_postfix+".pkl")
    
    ### due to pickle 5 protocol error

    if(df_mapped):
        df = pickle.load(open(target_root/("df_memmap"+filename_postfix+".pkl"), "rb"))
    else:
        df = pickle.load(open(target_root/("df"+filename_postfix+".pkl"), "rb"))


    if((target_root/("lbl_itos"+filename_postfix+".pkl")).exists()):#dict as pickle
        infile = open(target_root/("lbl_itos"+filename_postfix+".pkl"), "rb")
        lbl_itos=pickle.load(infile)
        infile.close()
    else:#array
        lbl_itos = np.load(target_root/("lbl_itos"+filename_postfix+".npy"))


    mean = np.load(target_root/("mean"+filename_postfix+".npy"))
    std = np.load(target_root/("std"+filename_postfix+".npy"))
    return df, lbl_itos, mean, std

def save_dataset(df,lbl_itos,mean,std,target_root,filename_postfix="",protocol=4):
    target_root = Path(target_root)
    df.to_pickle(target_root/("df"+filename_postfix+".pkl"), protocol=protocol)


def dataset_get_stats(df, col="data", simple=True):
    '''creates (weighted) means and stds from mean, std and length cols of the df'''
    if(simple):
        return df[col+"_mean"].mean(), df[col+"_std"].mean()
    else:
        #https://notmatthancock.github.io/2017/03/23/simple-batch-stat-updates.html
        #or https://gist.github.com/thomasbrandon/ad5b1218fc573c10ea4e1f0c63658469
        def combine_two_means_vars(x1,x2):
            (mean1,var1,n1) = x1
            (mean2,var2,n2) = x2
            mean = mean1*n1/(n1+n2)+ mean2*n2/(n1+n2)
            var = var1*n1/(n1+n2)+ var2*n2/(n1+n2)+n1*n2/(n1+n2)/(n1+n2)*np.power(mean1-mean2,2)
            return (mean, var, (n1+n2))

        def combine_all_means_vars(means,vars,lengths):
            inputs = list(zip(means,vars,lengths))
            result = inputs[0]

            for inputs2 in inputs[1:]:
                result= combine_two_means_vars(result,inputs2)
            return result

        means = list(df[col+"_mean"])
        vars = np.power(list(df[col+"_std"]),2)
        lengths = list(df[col+"_length"])
        mean,var,length = combine_all_means_vars(means,vars,lengths)
        return mean, np.sqrt(var)

def dataset_add_std_col(df, col="data", axis=(0), data_folder=None):
    '''adds a column with mean'''
    df[col+"_std"]=df[col].apply(lambda x: np.std(np.load(x if data_folder is None else data_folder/x, allow_pickle=True),axis=axis))


def dataset_add_mean_col(df, col="data", axis=(0), data_folder=None):
    '''adds a column with mean'''
    df[col+"_mean"]=df[col].apply(lambda x: np.mean(np.load(x if data_folder is None else data_folder/x, allow_pickle=True),axis=axis))

def dataset_add_length_col(df, col="data", data_folder=None):
    '''add a length column to the dataset df'''
    df[col+"_length"]=df[col].apply(lambda x: len(np.load(x if data_folder is None else data_folder/x, allow_pickle=True)))

def npys_to_memmap(npys, target_filename, max_len=0, delete_npys=True):
    memmap = None
    start = []#start_idx in current memmap file
    length = []#length of segment
    filenames= []#memmap files
    file_idx=[]#corresponding memmap file for sample
    shape=[]

    for idx,npy in tqdm(list(enumerate(npys))):
        data = np.load(npy, allow_pickle=True)
        if(memmap is None or (max_len>0 and start[-1]+length[-1]>max_len)):
            if(max_len>0):
                filenames.append(target_filename.parent/(target_filename.stem+"_"+str(len(filenames)+".npy")))
            else:
                filenames.append(target_filename)

            if(memmap is not None):#an existing memmap exceeded max_len
                shape.append([start[-1]+length[-1]]+[l for l in data.shape[1:]])
                del memmap
            #create new memmap
            start.append(0)
            length.append(data.shape[0])
            memmap = np.memmap(filenames[-1], dtype=data.dtype, mode='w+', shape=data.shape)
        else:
            #append to existing memmap
            start.append(start[-1]+length[-1])
            length.append(data.shape[0])
            memmap = np.memmap(filenames[-1], dtype=data.dtype, mode='r+', shape=tuple([start[-1]+length[-1]]+[l for l in data.shape[1:]]))

        #store mapping memmap_id to memmap_file_id
        file_idx.append(len(filenames)-1)
        #insert the actual data
        memmap[start[-1]:start[-1]+length[-1]]=data[:]
        memmap.flush()
        if(delete_npys is True):
            npy.unlink()
    del memmap

    #append final shape if necessary
    if(len(shape)<len(filenames)):
        shape.append([start[-1]+length[-1]]+[l for l in data.shape[1:]])
    #convert everything to relative paths
    filenames= [f.name for f in filenames]
    #save metadata
    np.savez(target_filename.parent/(target_filename.stem+"_meta.npz"),start=start,length=length,shape=shape,file_idx=file_idx,dtype=data.dtype,filenames=filenames)

def npys_to_memmap_batched(npys, target_filename, max_len=0, delete_npys=True, batch_length=900000):
    memmap = None
    start = np.array([0])#start_idx in current memmap file (always already the next start- delete last token in the end)
    length = []#length of segment
    filenames= []#memmap files
    file_idx=[]#corresponding memmap file for sample
    shape=[]#shapes of all memmap files

    data = []
    data_lengths=[]
    dtype = None

    for idx,npy in tqdm(list(enumerate(npys))):

        data.append(np.load(npy, allow_pickle=True))
        data_lengths.append(len(data[-1]))

        if(idx==len(npys)-1 or np.sum(data_lengths)>batch_length):#flush
            data = np.concatenate(data)
            if(memmap is None or (max_len>0 and start[-1]>max_len)):#new memmap file has to be created
                if(max_len>0):
                    filenames.append(target_filename.parent/(target_filename.stem+"_"+str(len(filenames))+".npy"))
                else:
                    filenames.append(target_filename)

                shape.append([np.sum(data_lengths)]+[l for l in data.shape[1:]])#insert present shape

                if(memmap is not None):#an existing memmap exceeded max_len
                    del memmap
                #create new memmap
                start[-1] = 0
                start = np.concatenate([start,np.cumsum(data_lengths)])
                length = np.concatenate([length,data_lengths])

                memmap = np.memmap(filenames[-1], dtype=data.dtype, mode='w+', shape=data.shape)
            else:
                #append to existing memmap
                start = np.concatenate([start,start[-1]+np.cumsum(data_lengths)])
                length = np.concatenate([length,data_lengths])
                shape[-1] = [start[-1]]+[l for l in data.shape[1:]]
                memmap = np.memmap(filenames[-1], dtype=data.dtype, mode='r+', shape=tuple(shape[-1]))

            #store mapping memmap_id to memmap_file_id
            file_idx=np.concatenate([file_idx,[(len(filenames)-1)]*len(data_lengths)])
            #insert the actual data
            memmap[start[-len(data_lengths)-1]:start[-len(data_lengths)-1]+len(data)]=data[:]
            memmap.flush()
            dtype = data.dtype
            data = []#reset data storage
            data_lengths = []

    start= start[:-1]#remove the last element
    #cleanup
    for npy in npys:
        if(delete_npys is True):
            npy.unlink()
    del memmap

    #convert everything to relative paths
    filenames= [f.name for f in filenames]
    #save metadata
    np.savez(target_filename.parent/(target_filename.stem+"_meta.npz"),start=start,length=length,shape=shape,file_idx=file_idx,dtype=dtype,filenames=filenames)
    
def reformat_as_memmap(df, target_filename, data_folder=None, annotation=False, max_len=0, delete_npys=True,col_data="data",col_label="label", batch_length=0):
    npys_data = []
    npys_label = []

    for id,row in df.iterrows():
        npys_data.append(data_folder/row[col_data] if data_folder is not None else row[col_data])
        if(annotation):
            npys_label.append(data_folder/row[col_label] if data_folder is not None else row[col_label])
    if(batch_length==0):
        npys_to_memmap(npys_data, target_filename, max_len=max_len, delete_npys=delete_npys)
    else:
        npys_to_memmap_batched(npys_data, target_filename, max_len=max_len, delete_npys=delete_npys,batch_length=batch_length)
    if(annotation):
        if(batch_length==0):
            npys_to_memmap(npys_label, target_filename.parent/(target_filename.stem+"_label.npy"), max_len=max_len, delete_npys=delete_npys)
        else:
            npys_to_memmap_batched(npys_label, target_filename.parent/(target_filename.stem+"_label.npy"), max_len=max_len, delete_npys=delete_npys, batch_length=batch_length)

    #replace data(filename) by integer
    df_mapped = df.copy()
    df_mapped["data_original"]=df_mapped.data
    df_mapped["data"]=np.arange(len(df_mapped))

    df_mapped.to_pickle(target_filename.parent/("df_"+target_filename.stem+".pkl"))
    return df_mapped

def prepare_data(data_path, denoised=False, target_fs=100, strat_folds=10, channels=8, channel_stoi=channel_stoi_default, target_folder=None, skimage_transform=True, recreate_data=True):
    '''prepares the Zheng et al 2020 dataset'''
    target_root = Path(".") if target_folder is None else target_folder
    target_root.mkdir(parents=True, exist_ok=True)

    if(recreate_data is True):
        #df_attributes = pd.read_excel("./AttributesDictionary.xlsx")
        #df_conditions = pd.read_excel("./ConditionNames.xlsx")
        #df_rhythm = pd.read_excel("./RhythmNames.xlsx")
        df = pd.read_excel(data_path/"Diagnostics.xlsx")
        df["id"]=df.FileName
        df["data"]=df.FileName.apply(lambda x: x+".npy")
        df["label_condition_txt"]=df.Beat.apply(lambda x: [y for y in x.split(" ") if x!="NONE"])
        df["label_rhythm_txt"]=df.Rhythm.apply(lambda x: x.split(" "))
        df["label_txt"]=df.apply(lambda row: row["label_condition_txt"]+row["label_rhythm_txt"],axis=1)
        df["sex"]=df.Gender.apply(lambda x:x.lower())
        df["age"]=df.PatientAge
        df.drop(["Gender","PatientAge","Rhythm","Beat","FileName"],inplace=True,axis=1)

        #map to numerical indices
        lbl_itos={}
        lbl_stoi={}
        lbl_itos["all"] = np.unique([item for sublist in list(df.label_txt) for item in sublist])
        lbl_stoi["all"] = {s:i for i,s in enumerate(lbl_itos["all"])}
        df["label"] = df["label_txt"].apply(lambda x: [lbl_stoi["all"][y] for y in x])
        lbl_itos["condition"] = np.unique([item for sublist in list(df.label_condition_txt) for item in sublist])
        lbl_stoi["condition"] = {s:i for i,s in enumerate(lbl_itos["condition"])}
        df["label_condition"] = df["label_condition_txt"].apply(lambda x: [lbl_stoi["condition"][y] for y in x])
        lbl_itos["rhythm"] = np.unique([item for sublist in list(df.label_rhythm_txt) for item in sublist])
        lbl_stoi["rhythm"] = {s:i for i,s in enumerate(lbl_itos["rhythm"])}
        df["label_rhythm"] = df["label_rhythm_txt"].apply(lambda x: [lbl_stoi["rhythm"][y] for y in x])
        df["dataset"]="Zheng2020"

        for id,row in tqdm(list(df.iterrows())):
            fs = 500.

            df_tmp = pd.read_csv(data_path/("ECGDataDenoised")/(row["id"]+".csv"))
            channel_labels = list(df_tmp.columns)
            sigbufs = np.array(df_tmp)*0.001 #assuming data is given in muV

            data = resample_data(sigbufs=sigbufs,channel_stoi=channel_stoi,channel_labels=channel_labels,fs=fs,target_fs=target_fs,channels=channels,skimage_transform=skimage_transform)
            assert(target_fs<=fs)
            np.save(target_root/(row["id"]+".npy"),data)

        stratified_ids = stratify(list(df["label_txt"]), lbl_itos["all"], [1./strat_folds]*strat_folds)
        df["strat_fold"]=-1
        idxs = np.array(df.index.values)
        for i,split in enumerate(stratified_ids):
            df.loc[idxs[split],"strat_fold"]=i

        #add means and std
        dataset_add_mean_col(df,data_folder=target_root)
        dataset_add_std_col(df,data_folder=target_root)
        dataset_add_length_col(df,data_folder=target_root)

        #save means and stds
        mean, std = dataset_get_stats(df)

        #save
        save_dataset(df, lbl_itos, mean, std, target_root)
    else:
        df, lbl_itos, mean, std = load_dataset(target_root,df_mapped=False)
    return df, lbl_itos, mean, std