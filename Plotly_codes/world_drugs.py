# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 23:32:55 2018

@author: ALLAN
"""

import plotly.plotly as py
import plotly
import pandas as pd
import itertools
import os
import numpy as np
import numpy.matlib
import sklearn.preprocessing
from scipy.spatial.distance import cdist,squareform

def plot_all(data,layout):
    fig = dict( data=data, layout=layout )
    plotly.offline.plot(fig, validate=False, filename='drugs_in_the_world.html')

def plot_retrieval(lat,lon,text, querry_name, classes, features,im_filenames, n_retrievals=3, distance = 'euclidean',standardize = 'False'):
    
    Y = np.array(classes)
    Y = Y-min(Y)+1
    # Numero de amostras
    #Nobj = len(Y)
    
    X=np.array(features)
    im_filenames=np.array(im_filenames)
    #Y=data[:,0]
    #nclasses=max(Y)-min(Y)+1
    #nfigs_classe=np.zeros(int(nclasses))
    
    #ymin=min(Y)
    
    #for i in range(int(nclasses)):
    #    nfigs_classe[i]=sum((Y==ymin+i).astype(int))
    # Buffer para saida dos resultados
    #X = data[:,1:]
    
    #normalizar
    if(standardize == 'True'):
        X = sklearn.preprocessing.scale(X)
    
    querry_idx=np.where(im_filenames == querry_name)
    
    # Calcula matriz de distancias
    print(np.array([X[querry_idx]]).shape)
    distance_values=cdist(X,np.array(X[querry_idx]))
    retrieval_idx= np.argsort(distance_values, axis=None)[:n_retrievals]
    

    #Marker colors
    querry_class = Y[querry_idx]
    colors = 255*np.array([[0,0,1],[0,1,0],[1,0,0]])
    
    rtr_color_array = np.zeros([n_retrievals,3])
    bin_index = Y[retrieval_idx]==querry_class
    print(bin_index)
    rtr_color_array[bin_index,:] = colors[1,:]
    rtr_color_array[~bin_index,:] = colors[2,:]
    rtr_color_array[0,:] = colors[0,:]
    rtr_color_list = rtr_color_array.tolist()
    rtr_color_list = ['rgb('+(str(x)[1:-1])+')' for x in rtr_color_list]
    
    # text_list = [None] * n_retrievals
    # for i in range(n_retrievals):
    #     text_list = im_filenames[retrieval_idx] +

    data = [ dict(
            type = 'scattergeo',
            #locationmode = 'USA-states',
            #lon = db[tuple(im_filenames_int)][5],
            lon = [lon[item] for item in retrieval_idx],
            #lat = db[tuple(im_filenames_int)][6],
            lat = [lat[item] for item in retrieval_idx],
            text = [text[item] for item in retrieval_idx],
            mode = 'markers',
            marker = dict(
                size = 15,
                opacity = 0.8,
                #reversescale = True,
                #autocolorscale = False,
                symbol = 'circle',
                color=rtr_color_list
            ))]

    layout = dict(
            title = 'Drugs in the world',
            colorbar = True,
            geo = dict(
                #scope='usa',
                #projection=dict( type='albers usa' ),
                showland = True,
                landcolor = "rgb(250, 250, 250)",
                subunitcolor = "rgb(217, 217, 217)",
                countrycolor = "rgb(217, 217, 217)",
                countrywidth = 0.5,
                subunitwidth = 0.5
            ),
        )

    fig = dict( data=data, layout=layout )
    plotly.offline.plot(fig, validate=False, filename='drugs_in_the_world.html')

if __name__ == '__main__':
    df = pd.read_csv('metadadosdrogasposicaogeografica2.csv', encoding = "ISO-8859-1", sep=';')
    df.head()
    df['Nome']= df['Nome'].astype(str)

    #df['text'] = df['sample_name'] + '; ' + df['location'] + '; ' + 'Lat: ' + df['Latitude'].astype(str) + '; ' + 'Long: ' + df['Longitude'].astype(str)
    df['text'] = df['Nome'] + '; ' + df['sample_name'] + '; ' + df['Substance'] + '; ' + df['location']
    
    db=df.set_index('Nome').T.to_dict('list')

    # scl = [ [0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],\
    #     [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"] ]

    data = [ dict(
            type = 'scattergeo',
            #locationmode = 'USA-states',
            lon = df['Longitude'],
            lat = df['Latitude'],
            text = df['text'],
            mode = 'markers',
            marker = dict(
                size = 8,
                opacity = 0.8,
                reversescale = True,
                autocolorscale = False,
                symbol = 'square',
                line = dict(
                    width=1,
                    color='rgba(102, 102, 102)'
                )
                # ,
                # colorscale = scl,
                # cmin = df['Nome'].min(),
                # color = df['Nome'],
                # cmax = df['Nome'].max(),
                # colorbar=dict(
                    # title="Names"
                # )
            ))]

    layout = dict(
            title = 'Drugs in the world',
            colorbar = True,
            geo = dict(
                #scope='usa',
                #projection=dict( type='albers usa' ),
                showland = True,
                landcolor = "rgb(250, 250, 250)",
                subunitcolor = "rgb(217, 217, 217)",
                countrycolor = "rgb(217, 217, 217)",
                countrywidth = 0.5,
                subunitwidth = 0.5
            ),
        )

    #plot_all(data,layout)
    
    classes = list(itertools.chain(*(pd.read_csv('ilicitas_lenet/labels_cnn_training.csv', sep=',').values.tolist())))
    features = pd.read_csv('ilicitas_lenet/feature_vectors_cnn_training.csv', sep=',').values.tolist()
    im_filenames = list(itertools.chain(*(pd.read_csv('ilicitas_lenet/image_filenames_cnn_training.csv', sep=',').values.tolist())))
    im_filenames = [os.path.basename(item)[:-7] for item in im_filenames]
    
    ##Lost data: "4065"
    db['4065']=[' Mortal Kombat 2', 'MDMA', 'Aug 13, 2013', 'Aug 13, 2013', 'Lanarkshire, Scotland', 55.673, 3.782,'']

    plot_retrieval([db[item][5] for item in im_filenames],
                   [db[item][6] for item in im_filenames],
                   [db[item][7] for item in im_filenames], 
                   '3558', classes, features, im_filenames,30)
