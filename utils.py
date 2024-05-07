from pathlib import Path
import re
import os
from PIL import Image,ImageSequence
import numpy as np
import cv2 as cv
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt


def get_end(x):
    return int(re.findall('.......ome',x.stem)[0][:6])

def get_ch(x):
    return int(re.findall('Ch.',x.stem)[0][2:])

def get_cycle(x):
    return int(re.findall('Cycle.....',x.as_posix())[0][5:])

def combine_tiffs(path, get_z = get_end, get_ch = get_ch, get_frame_cycle = get_cycle):
    """
    create a single multi-page tiff of all tiffs belonging to the same channel 
    and same z level in a given data directory
    
    path: path to the data directory
    ch: number of the channel
    get_frame_method: function for finding the frame number in the file name
    """
    files = list(path.iterdir())
    files = list(filter(lambda x: x.is_file() & (x.suffix == '.tif'), files))

    zs = np.unique(list(map(get_z, files)))
    channels = np.unique(list(map(get_ch, files)))

    ps = {f'z{z}': {f'ch{c}': None for c in channels} for z in zs}
    for c in channels:
        for z in zs:
            p = list(filter(lambda x: (get_z(x) == z) & (get_ch(x) == c), files))
            p = sorted(p, key = get_frame_cycle)
            imlist = [Image.open(i) for i in p]
            mov_dir = path/f'ch{c}_z_{z}_movie'
            if not mov_dir.is_dir(): os.mkdir(mov_dir)
            sp = mov_dir/'movie.ome.tif'
            if not sp.exists():
                imlist[0].save(sp.as_posix(), save_all = True, append_images = imlist[1:])
            ps[f'z{z}'][f'ch{c}'] = sp
    return ps


def draw_masks(im,ms,mask,show_plot=True):
    """
    function to remove any unwanted neurons
    """
    cv.namedWindow('image')
    down=[False]
    disp_im=im+ms
    color=(65*np.random.rand(1,3)).astype(np.uint8)
    
    def cb(e,x,y,f,z):
        if e==cv.EVENT_LBUTTONDOWN:
            down[0]=True
        if e==cv.EVENT_LBUTTONUP:
            down[0]=False
        if down[0]:
            mask[y,x]=1
            cv.circle(mask,(x,y),1,1,3)
            cv.circle(ms,(x,y),1,tuple(color[0].tolist()),3)
            cv.circle(disp_im,(x,y),1,tuple(color[0].tolist()),3)
            cv.imshow('image',disp_im)
            
    cv.setMouseCallback('image',cb)
    while True:
        cv.imshow('image',disp_im)
        if cv.waitKey(20) & 0xFF == 27:
            break
    
    cv.destroyAllWindows()
    if show_plot:
        _,ax=plt.subplots(1,2)
        ax[0].imshow(mask)
        ax[1].imshow(im)
    return ms,mask.astype(bool)


def remove_neurons(a,im,ms):
    """
    function to remove any unwanted neurons
    """
    cv.namedWindow('image')
    neurons=[]
    def cb(e,x,y,f,z):
        if e==cv.EVENT_RBUTTONDOWN:
            if a[x*im.shape[0]+y].sum()==1:
                neuron=np.argmax(a[x*im.shape[0]+y])
                ms[a[:,neuron].reshape(im.shape[:2], order='F')]=0
                neurons.append(neuron)
                cv.imshow('image',im+ms)

    cv.setMouseCallback('image',cb)
    while True:
        cv.imshow('image',im+ms)
        if cv.waitKey(20) & 0xFF == 27:
            break
    
    cv.destroyAllWindows()
    return neurons,ms