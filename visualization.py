import numpy as np
import torch

import matplotlib.pyplot as plt
import matplotlib.animation as animation

N_JOINTS = 15
body_edges = np.array(
[[0,1], [1,2],[2,3],[0,4],
[4,5],[5,6],[0,7],[7,8],[7,9],[9,10],[10,11],[7,12],[12,13],[13,14]]
)

def plot_scene_gif(save_path : str, 
            results, 
            input_seq, 
            output_seq, 
            plot_predicted:bool=True, 
            plot_actual:bool=True,
            title:str = "",
            interval:int=67):
    """Plot one pose forecasting scene as gif. The input is green, prediction 
    is blue and actual (ground truth) is yellow. 

    Adapted from MRT/test.py

    Parameters
    ----------
    save_path: str
        where to store the gif
    results : ndarray(#people, #frames=45, #N_JOINTS*3=45)
        prediction of the neural network.
    input_seq : ndarray(#people, #frames=15, #N_JOINTS*3=45)
        input sequence for the neural network.
    output_seq : ndarray(#people, #frames=46, #N_JOINTS*3=45)
        output sequence of the neural network.
    plot_predicted : bool, optional
        plot network predictions, by default True
    plot_actual : bool, optional
        plot actual trajectory, by default True
    title : str
        title to add
    interval : int
        interval between two frames in the gif
    """

    if isinstance(results, torch.Tensor):
        results = results.detach().cpu().numpy()
    if isinstance(input_seq, torch.Tensor):
        input_seq = input_seq.detach().cpu().numpy()
    if isinstance(output_seq, torch.Tensor):
        output_seq = output_seq.detach().cpu().numpy()

    rec=results[:,:,:]
    rec=rec.reshape(results.shape[0],-1,N_JOINTS,3)
        
    input_seq=input_seq.reshape(results.shape[0],15,N_JOINTS,3)
    pred=np.concatenate([input_seq,rec],axis=1)
    output_seq=output_seq.reshape(results.shape[0],-1,N_JOINTS,3)[:,1:,:,:]
    all_seq=np.concatenate([input_seq,output_seq],axis=1)

    fig = plt.figure(figsize=(10, 4.5))
    fig.tight_layout()
    ax = fig.add_subplot(111, projection='3d')

    length_=45+15

    p_x=np.linspace(-10,10,15)
    p_y=np.linspace(-10,10,15)

    def animate(i):
        """animate frame i"""
            
        ax.clear()

        # draw grid
        for x_i in range(p_x.shape[0]):
            temp_x=[p_x[x_i],p_x[x_i]]
            temp_y=[p_y[0],p_y[-1]]
            z=[0,0]
            ax.plot(temp_x,temp_y,z,color='black',alpha=0.1)

        for y_i in range(p_x.shape[0]):
            temp_x=[p_x[0],p_x[-1]]
            temp_y=[p_y[y_i],p_y[y_i]]
            z=[0,0]
            ax.plot(temp_x,temp_y,z,color='black',alpha=0.1)

        for j in range(pred.shape[0]): # for each person

            if plot_predicted:        
                xs=pred[j,i,:,0]
                ys=pred[j,i,:,1]
                zs=pred[j,i,:,2]
                    
                alpha=1
                # plot predicted joints as dots
                ax.plot( zs,xs, ys, 'y.',alpha=alpha)
                    
            if plot_actual:
                x=all_seq[j,i,:,0]
                y=all_seq[j,i,:,1]
                z=all_seq[j,i,:,2]
                    
                    
                # plot actual joints as dots
                ax.plot( z,x, y, 'y.')


            plot_edge=True
            if plot_edge:
                for edge in body_edges:

                    if plot_predicted:
                        x=[pred[j,i,edge[0],0],pred[j,i,edge[1],0]]
                        y=[pred[j,i,edge[0],1],pred[j,i,edge[1],1]]
                        z=[pred[j,i,edge[0],2],pred[j,i,edge[1],2]]
                        if i>=15:
                            # blue = prediction
                            ax.plot(z,x, y, zdir='z',c='blue',alpha=alpha)
                                    
                        else:
                            # green = input
                            ax.plot(z,x, y, zdir='z',c='green',alpha=alpha)
                            
                    if plot_actual:
                        x=[all_seq[j,i,edge[0],0],all_seq[j,i,edge[1],0]]
                        y=[all_seq[j,i,edge[0],1],all_seq[j,i,edge[1],1]]
                        z=[all_seq[j,i,edge[0],2],all_seq[j,i,edge[1],2]]
                            
                        if i>=15:
                            # yellow = baseline
                            ax.plot( z,x, y, 'yellow',alpha=0.8)
                        else:
                            ax.plot( z, x, y, 'green')
                            
                    
        ax.set_xlim3d([-3 , 3])
        ax.set_ylim3d([-3 , 3])
        ax.set_zlim3d([0,3])
        ax.set_axis_off()
        plt.title(title + "; frame " + str(i),y=-0.1)
        
    ani = animation.FuncAnimation(fig, animate, frames=length_, interval=interval, repeat=True)

    ani.save(save_path)