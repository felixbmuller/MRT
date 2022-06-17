import torch
import numpy as np
import torch_dct as dct

from MRT.Models import Transformer
from utils import N_JOINTS

def init_model(model_path: str='./saved_model/39.model', device="cpu", train_mode=False):
    """Load model from checkpoint.

    Parameters
    ----------
    model_path : str, optional
        path to load the model from, by default './saved_model/39.model'
    device : str, optional
        gpu or cpu, by default "cpu"
    train_mode : bool, optional
        whether to set the model to train or eval mode, by default False

    Returns
    -------
    MRT.Models.Transformer
        model
    """

    model = Transformer(d_word_vec=128, d_model=128, d_inner=1024,
            n_layers=3, n_head=8, d_k=64, d_v=64,device=device).to(device)

    model.load_state_dict(torch.load(model_path,map_location=device)) 
    model.train(train_mode)

    return model

def infer(model, input_seq, output_seq, device="cpu"):
    """perform inference using the Tranformer model

    Parameters
    ----------
    model : Transformer
        model to use
    input_seq : np.ndarray
        input
    output_seq : np.ndarray
        ground truth
    device : str, optional
        gpu or cpu, by default "cpu"

    Returns
    -------
    results: Tensor
        prediction
    output_seq: Tensor
        ground truth
    """

    input_seq=torch.tensor(input_seq,dtype=torch.float32).to(device)
    output_seq=torch.tensor(output_seq,dtype=torch.float32).to(device)
    use=[input_seq.shape[1]]
    
    input_=input_seq.view(-1,15,input_seq.shape[-1])
    # same shape as before (3, 15, 45)

    output_=output_seq.view(output_seq.shape[0]*output_seq.shape[1],-1,input_seq.shape[-1])
    # shape: (138, 1, 45)

    input_ = dct.dct(input_)
    output__ = dct.dct(output_[:,:,:])
    
    
    rec_=model.forward(input_[:,1:15,:]-input_[:,:14,:],dct.idct(input_[:,-1:,:]),input_seq,use)
    
    rec=dct.idct(rec_)

    results=output_[:,:1,:]
    for i in range(1,16):
        results=torch.cat([results,output_[:,:1,:]+torch.sum(rec[:,:i,:],dim=1,keepdim=True)],dim=1)
        # why is output fused into result here? Isn't output the GT
    results=results[:,1:,:]

    new_input_seq=torch.cat([input_seq,results.reshape(input_seq.shape)],dim=-2)
    new_input=dct.dct(new_input_seq.reshape(-1,30,45))
    
    new_rec_=model.forward(new_input[:,1:,:]-new_input[:,:-1,:],dct.idct(new_input[:,-1:,:]),new_input_seq,use)
    
    
    new_rec=dct.idct(new_rec_)
    
    new_results=new_input_seq.reshape(-1,30,45)[:,-1:,:]
    for i in range(1,16):
        new_results=torch.cat([new_results,new_input_seq.reshape(-1,30,45)[:,-1:,:]+torch.sum(new_rec[:,:i,:],dim=1,keepdim=True)],dim=1)
    new_results=new_results[:,1:,:]
    
    results=torch.cat([results,new_results],dim=-2)

    rec=torch.cat([rec,new_rec],dim=-2)

    results=output_[:,:1,:]

    for i in range(1,16+15):
        results=torch.cat([results,output_[:,:1,:]+torch.sum(rec[:,:i,:],dim=1,keepdim=True)],dim=1)
    results=results[:,1:,:]

    new_new_input_seq=torch.cat([input_seq,results.unsqueeze(0)],dim=-2)
    new_new_input=dct.dct(new_new_input_seq.reshape(-1,45,45))
    
    new_new_rec_=model.forward(new_new_input[:,1:,:]-new_new_input[:,:-1,:],dct.idct(new_new_input[:,-1:,:]),new_new_input_seq,use)


    new_new_rec=dct.idct(new_new_rec_)
    rec=torch.cat([rec,new_new_rec],dim=-2)

    results=output_[:,:1,:]

    for i in range(1,31+15):
        results=torch.cat([results,output_[:,:1,:]+torch.sum(rec[:,:i,:],dim=1,keepdim=True)],dim=1)
    results=results[:,1:,:]

    return results, output_seq

def split_extend_sequence(seq):
    """split an output sequence (ground truth or predicted) into one second parts and separate coordinates for all joints.

    Parameters
    ----------
    seq : Tensor
        sequence to split

    Returns
    -------
    segment1, segment2, segment3
    """

    segment1=seq[:,:15,:].view(seq.shape[0],-1,N_JOINTS,3)
    segment2=seq[:,:30,:].view(seq.shape[0],-1,N_JOINTS,3)
    segment3=seq[:,:45,:].view(seq.shape[0],-1,N_JOINTS,3)

    return segment1, segment2, segment3

def calc_loss(results, output_seq, MPJPE=True, mean=True):
    """Calculate loss

    Parameters
    ----------
    results : Tensor
        prediction
    output_seq : Tensor
        ground truth
    MPJPE : bool, optional
        whether to calculate MPJPE or pose with align, by default True
    mean : bool, optional
        return mean loss or all individual losses, by default True

    Returns
    -------
    loss1, loss2, loss3
    """

    prediction_1, prediction_2, prediction_3 = split_extend_sequence(results)
    gt_1, gt_2, gt_3 = split_extend_sequence(output_seq)

    if MPJPE:
        #MPJPE
        loss1=torch.sqrt(((prediction_1 - gt_1) ** 2).sum(dim=-1)).mean(dim=-1).mean(dim=-1).numpy().tolist()
        loss2=torch.sqrt(((prediction_2 - gt_2) ** 2).sum(dim=-1)).mean(dim=-1).mean(dim=-1).numpy().tolist()
        loss3=torch.sqrt(((prediction_3 - gt_3) ** 2).sum(dim=-1)).mean(dim=-1).mean(dim=-1).numpy().tolist()
            
        #pose with align
    else:
        loss1=torch.sqrt(((prediction_1 - prediction_1[:,:,0:1,:] - gt_1 + gt_1[:,:,0:1,:]) ** 2).sum(dim=-1)).mean(dim=-1).mean(dim=-1).numpy().tolist()
        loss2=torch.sqrt(((prediction_2 - prediction_2[:,:,0:1,:] - gt_2 + gt_2[:,:,0:1,:]) ** 2).sum(dim=-1)).mean(dim=-1).mean(dim=-1).numpy().tolist()
        loss3=torch.sqrt(((prediction_3 - prediction_3[:,:,0:1,:] - gt_3 + gt_3[:,:,0:1,:]) ** 2).sum(dim=-1)).mean(dim=-1).mean(dim=-1).numpy().tolist()

    if mean:
        loss1 = np.mean(loss1)
        loss2 = np.mean(loss2)
        loss3 = np.mean(loss3)

    return loss1, loss2, loss2

