# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 16:38:28 2018

@author: ALLAN
"""
import plotly
import numpy as np
import sklearn.preprocessing
from scipy.spatial.distance import cdist

class Maps(object):
    def __init__(self, title='', result_filename='map_result'):
        self._title=title
        self._result_filename=result_filename
    
    def set_title(self, title=''):
        self._title=title
    
    def set_result_filename(self, result_filename='map_result'):
        self._result_filename=result_filename

    def plot_all(self, lat, lon, text, marker_size = 15, marker_opacity = 0.8, marker_symbol = 'circle'):
        """Plot all data with the same color
    	 
    		Parameters
            ----------
    		lat : list
    			It has the latitude for each marker
    		lon : list cujo i-esimo elemento 
        		It has the longitude for each marker
    		text : list
    			It has the text that will be shown for each marker
    		marker_size : float, optional, default False
    			Control marker opacity
    		marker_opacity : float, optional, default False
    			Control marker opacity
    		marker_symbol : float, optional, default False
    			Control marker symbol
        """   # noqa
    
        #Configuration of variables data and layout that were necessary
        #for plotly function: plotly.offline.plot
        self._data = [ dict(
                type = 'scattergeo',
                #locationmode = 'USA-states',
                lon = lon,
                lat = lat,
                text = text,
                mode = 'markers',
                marker = dict(
                    size = marker_size,
                    opacity = marker_opacity,
                    reversescale = True,
                    autocolorscale = False,
                    symbol = marker_symbol,
                    line = dict(
                        width=1,
                        color='rgba(102, 102, 102)'
                    )
                ))]
    
        self._layout = dict(
                title = self._title,
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
    
    def plot_retrieval(self, lat, lon, text, query_name, classes, features,
                       im_filenames, n_retrievals=3, distance = 'euclidean',
                       standardize = 'False', marker_size = 15, 
                       marker_opacity = 0.8, marker_symbol = 'circle'):
        """Plot the map of the locations of the retrieved images from a query
        image. The query image has been marked blue, the images retrieved from
        the same class as the query image are in green and the images of
        different classes in red.
    	 
    		Parameters
            ----------
    		lat : list
    			It has the latitude for each marker
    		lon : list cujo i-esimo elemento 
        		It has the longitude for each marker
    		text : list
    			It has the text that will be shown for each marker
    		query_name : list
    			Name of the image query
       		classes : integer list
        			Classes of each data point corresponding to each name
                    of im_filenames
       		features : integer list
        			Features of the each data point corresponding to each name
                    of im_filenames
       		im_filenames : integer list
        			List of the image names
       		n_retrievals : integer
        			Number of retrievals
            distancia : str or function, optional
                Distance to be used in retrieval. The distance can be:
    			'braycurtis', 'canberra', 'chebyshev', 'cityblock',
    			'correlation', 'cosine', 'dice', 'euclidean', 'hamming',
    			'jaccard', 'kulsinski', 'mahalanobis', 'matching',
    			'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
    			'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule'.
            standardize : boolean, optional, default False
                Normalizes with media 0 and variance 1  for True
    		marker_size : float, optional, default False
    			Control marker opacity
    		marker_opacity : float, optional, default False
    			Control marker opacity
    		marker_symbol : float, optional, default False
    			Control marker symbol
        """   # noqa
        
        Y = np.array(classes)
        Y = Y-min(Y)+1
        
        # Samples number
        #Nobj = len(Y)
        
        X=np.array(features)
        im_filenames=np.array(im_filenames)
        
        #normalize
        if(standardize == 'True'):
            X = sklearn.preprocessing.scale(X)
        
        query_idx=np.where(im_filenames == query_name)
        
        # Compute distance matrix
        distance_values=cdist(X,np.array(X[query_idx]))
        retrieval_idx= np.argsort(distance_values, axis=None)[:n_retrievals]
        
    
        #Marker colors
        query_class = Y[query_idx]
        colors = 255*np.array([[0,0,1],[0,1,0],[1,0,0]])
        
        rtr_color_array = np.zeros([n_retrievals,3])
        bin_index = Y[retrieval_idx]==query_class
    
        rtr_color_array[bin_index,:] = colors[1,:]
        rtr_color_array[~bin_index,:] = colors[2,:]
        rtr_color_array[0,:] = colors[0,:]
        rtr_color_list = rtr_color_array.tolist()
        rtr_color_list = ['rgb('+(str(item)[1:-1])+')' for item in rtr_color_list]
        
        #Configuration of variables data and layout that were necessary
        #for plotly function: plotly.offline.plot
        self._data = [ dict(
                type = 'scattergeo',
                #locationmode = 'USA-states',
                #lon = db[tuple(im_filenames_int)][5],
                lon = [lon[item] for item in retrieval_idx],
                #lat = db[tuple(im_filenames_int)][6],
                lat = [lat[item] for item in retrieval_idx],
                text = [text[item] for item in retrieval_idx],
                mode = 'markers',
                marker = dict(
                    size = marker_size,
                    opacity = marker_opacity,
                    #reversescale = True,
                    #autocolorscale = False,
                    symbol = marker_symbol,
                    color=rtr_color_list
                ))]
    
        self._layout = dict(
                title = self._title,
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
        
    def plot_classes(self, lat, lon, text, classes, classes_colors=None, 
                     marker_size = 15, marker_opacity = 0.8,
                     marker_symbol = 'circle'):
        """Plot elements of the same class with the same color
    	 
    		Parameters
            ----------
    		lat : list
    			It has the latitude for each marker
    		lon : list cujo i-esimo elemento 
        		It has the longitude for each marker
    		text : list
    			It has the text that will be shown for each marker
       		classes : integer list
        			It has the classes of each data point
       		classes_colors : list of lists, optional, default None
        			It has the rgb colors of each class
    		marker_size : float, optional, default False
    			Control marker opacity
    		marker_opacity : float, optional, default False
    			Control marker opacity
    		marker_symbol : float, optional, default False
    			Control marker symbol
        """   # noqa
        #the classes index: Y begin in zero
        Y = np.array(classes)
        Y = Y-min(Y)
        
        # Samples number
        Nobj = len(Y)
        
        #Configuration of variables data and layout that were necessary
        #for plotly function: plotly.offline.plot
    
        #if it colors were passed as argument 
        if classes_colors is not None:
            #Marker colors
            colors = np.array(classes_colors)
            color_array = np.zeros([Nobj,3])
            #auxiliar indexes
            idx = range(Nobj)
            color_array[idx,:] = colors[Y[idx],:]
        
            color_list = color_array.tolist()
            color_list = ['rgb('+(str(item)[1:-1])+')' for item in color_list]
            
            self._data = [ dict(
                    type = 'scattergeo',
                    #locationmode = 'USA-states',
                    lon = lon,
                    lat = lat,
                    text = text,
                    mode = 'markers',
                    marker = dict(
                        size = marker_size,
                        opacity = marker_symbol,
                        #reversescale = True,
                        #autocolorscale = False,
                        symbol = marker_symbol,
                        color=color_list
                    ))]
        #if not
        else:
            scl = [ [0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],\
                     [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"] ]
    
            self._data = [ dict(
                    type = 'scattergeo',
                    #locationmode = 'USA-states',
                    lon = lon,
                    lat = lat,
                    text = text,
                    mode = 'markers',
                    marker = dict(
                        size = marker_size,
                        opacity = marker_opacity,
                        #reversescale = True,
                        #autocolorscale = False,
                        symbol = marker_symbol,
                        colorscale = scl,
                        cmin = 0,
                        color=Y,
                        cmax = Y.max()
                    ))]
            
        #common configuration for both paths
        self._layout = dict(
                title = self._title,
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
    
    def show_map(self):
        #Create map with plotly function: plotly.offline.plot
        fig = dict( data=self._data, layout=self._layout )
        plotly.offline.plot(fig, validate=False, filename=self._result_filename)