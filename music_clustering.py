#Music Learning Implementation of Data

#Necessary packages
from sklearn.manifold import TSNE
from sklearn.preprocessing import scale
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from scipy.spatial.distance import pdist

#For T-SNE
#Implement the T-SNE clustering model
cluster_palette = sns.color_palette("bright", 10)

#strip the labels so we know the features
music_features = all_genres.drop(columns = ["filename", "label"])

#load the model
tsne = TSNE()

#fit the model
genres_tsne = tsne.fit_transform(music_features)

#draw a scatterplot of the resulting features
ax = sns.scatterplot(x = genres_tsne[:, 0],
                     y = genres_tsne[:, 1],
                     hue = all_genres.label,
                     size = all_genres.label)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width  * 0.85, box.height])
ax.legend(loc = "center right", bbox_to_anchor = (1.3, 0.5), ncol = 1)
plt.title("T-SNE Representation")
plt.show()
#kind of difficult to read and peruse. all we can observe is how far away
#a cluster of pop music is from everything else
#can visually observe a tight cluster of classical music,
#tight cluster of metal music, both of which are on opposite sides

### Hierarchial Clustering
#Scale all the data due to different units of measurement
X = scale(music_features, axis = 0)

#create the clustering values
#euclidean distance for simplicity
#average method due to highest cophenetic correlation coefficient
Z = linkage(X, method = "average")

#Check clustering status: How close is the cophenetic correlation coefficient?
c, coph_dists = cophenet(Z, pdist(X))

#which clusters were merged in the ith iteration?
Z[0] #index 182, 1182, distance is 0, cluster of 2
Z[1] #index 756, 1056, distance is 0, cluster of 2
#first cluster of 3
cluster_of_3 = Z[Z[:, 3] == 3]
#index 1013 and 1238, 1238 > 1200, so index 38
Z[38] #index 713, 738, distance is 0, cluster of 2
#cluster of 3 becomes 1013, 713, 738 indices


###Plotting the dendrogram
#Calculate the full dendrogram
plt.figure(figsize = (25, 10))
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Song Index")
plt.ylabel("Euclidean Distance")
dendrogram(Z,
           leaf_rotation = 90.,     #rotate x-axis labels
           leaf_font_size = 8.,     #font size
           labels = all_genres.filename.str.slice(0, -3).to_list())
plt.show()

#Dendrogram Truncation
plt.figure(figsize = (25, 10))
plt.title("Hierarchial Clustering Dendrogram (Truncated)")
plt.xlabel("Song Index")
plt.ylabel("Euclidean Distance")
dendrogram(Z,
           leaf_rotation = 90.,
           leaf_font_size = 12.,
           labels = all_genres.filename.str.slice(0, -3).to_list(),
           truncate_mode = "lastp",     #how many clusters?
           p = 12,                      #this many clusters
           show_contracted = True)      #show the counts
plt.gcf().subplots_adjust(bottom = 0.2)
plt.show()
#black dots show where prior clusters and merges occured
#some high black dots show that we may have missed some significant info

#https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/
#to add information and other fancy things to dendrogram for easy viewing
