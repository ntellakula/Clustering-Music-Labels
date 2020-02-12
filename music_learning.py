#Neccesary packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#import the data
source = pd.read_csv("C:/Users/NTellaku/Documents/Python Scripts/Music Learning/data.csv")
source2 = pd.read_csv("C:/Users/NTellaku/Documents/Python Scripts/Music Learning/data_2genre.csv")

#columns in the csvs - are they the same?
source.columns == source2.columns

#concatenate the two data frames
all_genres = pd.concat([source, source2], ignore_index = True)

#what genres are music are included in the data?
all_genres.label.unique()

#notice the labels of 1 & 2. data states they are pop & classical respectively
all_genres.label = all_genres.label.replace(1, "pop")
all_genres.label = all_genres.label.replace(2, "classical")

#what genres are music are included in the data?
all_genres.label.unique()



#Some exploratory on the genre i like best - Metal
metal_music = all_genres[all_genres.label == "metal"]

#Density Plot of all the variables
metal_features = metal_music.drop(columns = ["filename", "label"])

#create overarching figure
fig = plt.figure(figsize = (40, 30))
for i in range(metal_features.shape[1]):
   
    #Set up the plot and add the subplots
    ax = plt.subplot(4, 7, i + 1)
    
    #Draw the plot
    sns.distplot(metal_features.iloc[:, i],
                 hist = False,
                 kde = True,
                 kde_kws = {'shade': True, 'linewidth': 1},
                 color = "#4287f5")
    #Add labels
    plt.xlabel(metal_features.columns[i])

#add a big axis, hide frame
fig.add_subplot(111, frameon = False)
#hide tick and tick label of the big axis
plt.tick_params(labelcolor = 'none',
                top = False, bottom = False,
                left = False, right = False)   
plt.ylabel("Probability Density", fontsize = "xx-large")
plt.annotate("Density Plots of Features", ha = "left", va = "top",
            xy = (0.5, 0), fontsize = "xx-large",
            xytext = (0.5, -0.08))
fig.tight_layout(pad = 8.0)
plt.show()


#Density Plot of the tempo for all the individual genres
#Not a great idea, but why not just for fun
#Have to do this while iterating through a loop
genres_list = all_genres.label.unique()
fig = plt.figure(figsize = (10, 8))
for j in range(len(genres_list)):
    genre = all_genres[all_genres.label == genres_list[j]]
    
    #Draw the plot
    sns.distplot(genre.tempo,
                 hist = False,
                 kde = True,
                 label = genres_list[j].capitalize(),
                 kde_kws = {"shade": True, "linewidth": 1})
#Make the plot look better
plt.title("Tempo of All Genres")
plt.xlabel("Tempo")
plt.ylabel("Probability Density")

#Fix legend to have thicker bars
leg = plt.legend(ncol = 2)
for legobj in leg.legendHandles:
    legobj.set_linewidth(4.0)
plt.show()

#check for correlation among all the features of the data set, and visualize it
#save correlations as a separate object
metal_corr = metal_features.corr()

#unpivot the correlation data frame to have [x, y] pairs with values
corr_unpivot = pd.melt(metal_corr.reset_index(), id_vars = "index")
#Mapping the color to values
#range of the color values
color_min, color_max, n_colors = [-1, 1, corr_unpivot.shape[0]]
palette = sns.diverging_palette(10, 150, n = n_colors)

#map the color to the value
def value_to_color(val):
    val_position = float((val - color_min)) / (color_max - color_min)
    ind = int(val_position * (n_colors - 1))
    return palette[ind]

#Set up a function to add square markers
#borrowed from:
#https://towardsdatascience.com/better-heatmaps-and-correlation-matrix-plots-in-python-41445d0f2bec
def heatmap(x, y, value, size):
    #split the plot into a 1x15 column grid
    plot_grid = plt.GridSpec(1, 15, hspace = 0.2, wspace = 0.1)
    ax = plt.subplot(plot_grid[:, :-1]) #first 14 columns
    
    #Mapping from column names to integer coordinates
    x_labels = [v for v in sorted(x.unique())]
    y_labels = [v for v in sorted(y.unique())]
    x_to_num = {p[1]: p[0] for p in enumerate(x_labels)}
    y_to_num = {p[1]: p[0] for p in enumerate(y_labels)}
    
    #size of scatter plot and shape
    size_scale = 230
    ax.scatter(x = x.map(x_to_num),     #x mapping
               y = y.map(y_to_num),     #y mapping
               s = size * size_scale,   #Vector of size of squares
               marker = "s",            #Force the square shape
               c = value.apply(value_to_color)) #color with mapping
    
    #Show column labels on the axes
    ax.set_xticks([x_to_num[v] for v in x_labels])
    ax.set_yticks([y_to_num[v] for v in y_labels])
    ax.set_xticklabels(x_labels, rotation = 45,
                       horizontalalignment = "right")
    ax.set_yticklabels(y_labels)
    
    ax.grid(False, "major")
    ax.grid(True, "minor", color = "white")
    
    #move each correlation point to center of square
    ax.set_xticks([t + 0.5 for t in ax.get_xticks()], minor = True)
    ax.set_yticks([t + 0.5 for t in ax.get_yticks()], minor = True)
    ax.set_facecolor('#f0f4fa')
    
    #Removes "extra" square space that shows up
    ax.set_xlim([-0.5, max([v for v in x_to_num.values()]) + 0.5])
    ax.set_ylim([-0.5, max([v for v in y_to_num.values()]) + 0.5])
    
    #remove the frame
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    #remove the actual ticks, keep the labels
    ax.tick_params(axis = u'both', which = u'both', length = 0)
    
    #color legend
    ax = plt.subplot(plot_grid[:, -1]) #rightmost column for the gradient
    
    col_x = [0] * len(palette) #x-coords
    bar_y = np.linspace(color_min, color_max, n_colors) #y-coords
    bar_height = bar_y[1] - bar_y[0]
    ax.barh(y = bar_y,
            width = [5] * len(palette), #5 units wide bar
            left = col_x,               #bars start at 0
            height = bar_height,
            color = palette,
            linewidth = 0)
    
    #specs for the bar graph
    ax.set_xlim(1, 2)           #bars go from 0 to 5, this crops the plot
    ax.grid(False)              #hide the grid
    ax.set_facecolor("white")   #white background
    ax.set_xticks([])           #no horizontal ticks
    ax.set_yticks(np.linspace(min(bar_y), max(bar_y), 3)) #min, mid, max ticks
    ax.yaxis.tick_right()       #vertical ticks on the right
    
    #remove the frame
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    
#Size and correlation graphic
plt.figure(figsize = (9, 9))
plt.suptitle("Correlations Between Metal Music Features")
plt.gcf().subplots_adjust(bottom = 0.2,
                          left = 0.2)
heatmap(corr_unpivot["index"],
        corr_unpivot["variable"],
        corr_unpivot["value"],
        size = corr_unpivot["value"].abs())

#great visual, now actually classify which ones matter
#tempo and beats, mse and mfcc1, zero_crossing_rate with spectral_centroid
#zero_crossing_rate oppositely with mfcc2, kinda with rolloff
#spectral_centroid with rolloff, spectral_bandwidth, oppositely with mfcc2
#spectral_bandwith with rolloff, mfcc3, oppoisitely with mfcc2
#rolloff oppositely with mfcc2