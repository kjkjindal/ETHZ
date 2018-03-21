import numpy as np
import h5py
import itertools
import csv
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
import os
import gc
import matplotlib.patches as mpatches
# import seaborn as sns



#helpers
def gene_extractor(target_list, data, labels):
    data_return = []
    label_return = []
    for i in range(len(data)):
        if (labels[i][0] == target_list[0] and labels[i][1] == target_list[1] and labels[i][2] == target_list[2]):
            data_return.append(data[i])
            label_return.append(labels[i])

    return data_return, label_return

def iter_gen(x,y,z):
    if(x == y and  y == z):
        return [[x,y,z]]
    return ([x,y,z], [z,x,y], [y,z,x])

def shift(l,n):
    return l[n:] + l[:n]


def create_paths(root = '.'):

    paths = []
    paths.append(root + '/log_mmd.csv')
    paths.append(root + '/log_G_loss.csv')
    paths.append(root + '/log_D_loss.csv')
    paths.append(root + '/log_med_gene1.csv')
    paths.append(root + '/log_med_gene2.csv')
    paths.append(root + '/log_med_gene3.csv')
    paths.append(root + '/log_med_avg.csv')

    return paths


#logging function for saving model results
def save_data(data, paths = None, root = '.'):

    '''
        data order: lists of mmds, G_loss, D_loss, gene1 median, gene2 median, gene3 median, average median
        returns: nothing
    '''
    if(paths == None):
        paths = []
        paths = create_paths(root)

    for i,j in zip(paths,data):
        with open(i, 'w', newline = '') as writeList:
                fileWrite=csv.writer(writeList)

                if j:
                    for k in j:
                        fileWrite.writerow([k])



#loader for loading model results
def load_data(paths = None, root = '.'):

    '''
        paths: list of path strings in order: mmds, losses(G,D), gene1 median, gene2 median, gene3 median, average median
        returns: data in that order
    '''

    if (paths == None):
        paths = []
        paths = create_paths(root)

    data_list = []

    for i in paths:
        temp_list = []
        with open(i, 'r') as file:
            reader = list(csv.reader(file))
            for j in reader:
                temp_list.append(float(j[0]))
            data_list.append(temp_list)


    return tuple(data_list)


def make(path):
    if not os.path.exists(path):
        os.makedirs(path)
    
    return path
'''
-----------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------
'''


def unique_triplets(n):
    '''
        Returns all unique edge triplets, assuming 0-n unique edges
    '''
    triplets = []
    for  i in range(n):
        triplets.append([i,i,i])
        for j in range(i,n):
            for k in range(i+1,n):
                triplets.append([i,j,k])

    return triplets



#function to extract required triplets from raw data
def extract_data(x,y,z,data,label):
    
    #get valid iterations of x, y, z
    a = iter_gen(x,y,z)
    
    #get corresponding expression data
    genes = []
    c = []
    labels = []
    l = []
    for i in a:
        c,l = gene_extractor(i, data, label)
        genes.extend(c)
        labels.extend(l)

    if((not c) or (not l)):
        return []
        
    genes = np.array(genes)
    
    #create final dataset (reorder shuffled labels)
    gene_final = []
    label_final = []
    
    for j in range(len(genes)):
        temp_gene = list(genes[j])
        temp_label = list(labels[j])
        while (temp_label[0] != x or temp_label[1] != y or temp_label[2] != z):
            temp_label = shift(temp_label, 1)
            temp_gene = shift(temp_gene, 1)


        gene_final.append(temp_gene)
        label_final.append(temp_label)
    
    return np.transpose(np.array(gene_final), [0,2,1]), np.array(label_final)


#plotting function for a small generated sample
def plot_sample(samples, r, c):

    '''
        samples: samples to plot
        r: number of rows in the plot
        c: number of columns in the plot
    '''
    fig = plt.figure(figsize = (r,c))
    gs = gridspec.GridSpec(r,c)
    
    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.plot(sample)
        plt.axis([0,9,-15,5])
    
    return fig



#plotting function for plotting model results
def plot_data(data, root = '.', med_baseline = None):

    '''
        data order: lists of mmds, G_loss, D_loss, gene1 median, gene2 median, gene3 median, average median
        root: root directory for saving plots (needs to exist already)
        med_baseline: baseline for median pariwise distance plot

        returns: nothing
    '''

    #plot mmd
    fig = plt.plot(data[0])
    plt.savefig(root + '/mmd.png')
    plt.close()

    #losses
    fig = plt.plot(data[1], label = 'Generator Loss')
    fig = plt.plot(data[2], label = 'Discriminator Loss')
    plt.legend()
    plt.savefig(root + '/losses.png')
    plt.close()

    #plot gene medians with average
    fig = plt.plot(data[3], label = 'Gene 1')
    fig = plt.plot(data[4], label = 'Gene 2')
    fig = plt.plot(data[5], label = 'Gene 3')
    fig = plt.plot(data[6], label = 'Average')
    if (med_baseline != None):
        plt.plot(np.zeros_like(data[6]) + med_baseline, label = 'Baseline')
    plt.legend()
    plt.savefig(root + '/pairwise distance medians.png')
    plt.close()




#function to calculate pair wise median distance
def pair_wise_dist_median(genes):

    sess = tf.Session()
    
    D = []

    for i in genes:
        r = tf.reduce_sum(i*i, 1)
        r = tf.reshape(r, [-1, 1])
        D.append(np.median(sess.run(r - 2*tf.matmul(i, tf.transpose(i)) + tf.transpose(r))))

    sess.close()
    return tuple(D)



def histo_plotter(data1, data2, base_name, bin_seq, ylim, root = '.'):

    '''
        data1: real data shape(None, 10, 3)
        data2: GAN data shape(None, 10, 3)
        root: base directory
        base_name: base file name identifier
        bin_seq: bin sequence

        Returns: nothing
    '''
    a1,b1,c1 = np.squeeze(np.split(data1, 3, axis = 2))
    fig = plt.figure(figsize = (10,3))
    gs = gridspec.GridSpec(1,2)
    ax1 = plt.subplot(gs[0])
    plt.hist(np.reshape(a1, [-1,1]), bins = bin_seq, alpha = 0.5, label = 'gene1_real')
    plt.hist(np.reshape(b1, [-1,1]), bins = bin_seq, alpha = 0.5, label = 'gene2_real')
    plt.hist(np.reshape(c1, [-1,1]), bins = bin_seq, alpha = 0.5, label = 'gene3_real')
    plt.ylim(0,ylim)
    plt.legend()
    
    ax2 = plt.subplot(gs[1])
    a2,b2,c2 = np.squeeze(np.split(data2, 3, axis = 2))
    plt.hist(np.reshape(a2, [-1,1]), bins = bin_seq, alpha = 0.5, label = 'gene1_GAN')
    plt.hist(np.reshape(b2, [-1,1]), bins = bin_seq, alpha = 0.5, label = 'gene2_GAN')
    plt.hist(np.reshape(c2, [-1,1]), bins = bin_seq, alpha = 0.5, label = 'gene3_GAN')
    plt.ylim(0,ylim)
    plt.legend()
    
    plt.savefig((root + '/check{}.png').format(str(base_name).zfill(4)))
    plt.close()  

    #save raw plot data
    real_raw = [np.histogram(i, bins = bin_seq) for i in [a1,b1,c1]]
    gan_raw = [np.histogram(i, bins = bin_seq) for i in [a2,b2,c2]]

    #save x points (once for whole run)
    # print (real_raw[0][1])
    root2 = make(root + '/histo data raw')
    save_data(data = [list(real_raw[0][1])], paths = [root2 + '/bins.csv'])

    #save heights for all
    for i,j in enumerate(real_raw):
        save_data(data = [list(j[0])], paths = [root2 + '/real_hist_epoch{0}_gene{1}.csv'.format(base_name, i+1).zfill(4)])

    for i,j in enumerate(gan_raw):
        save_data(data = [list(j[0])], paths = [root2 + '/gan_hist_epoch{0}_gene{1}.csv'.format(base_name, i+1).zfill(4)])




def time_resolved_histo(data_real, data_gan, root, bins, epoch = 0, num_genes = 3, width = 0.02):
    '''
        data_real: 60k real data points (60k, 10, 3)
        data_gan : 60k generated data points (60k, 10, 3) 
        width    : width of bars in plots
        root     : folder to save raw data and histograms plots
        bins     : bins for histograms
        epoch    : epoch number
        num_genes: number of genes
    '''
    real_genes = np.squeeze(np.split(data_real, 3, 2))
    generated_genes = np.squeeze(np.split(data_gan, 3, 2))
    real_genes_time = [np.squeeze(np.split(i, 10, 1)) for i in real_genes]
    generated_genes_time = [np.squeeze(np.split(i, 10, 1)) for i in generated_genes]

    fig = plt.figure(figsize=(44, 11))
    gs = gridspec.GridSpec(3,10)
    c = 0
    #gene1
    for j in range(num_genes):
        for i in range(10):
            data_real = np.histogram(real_genes_time[j][i], bins)
            data_generated = np.histogram(generated_genes_time[j][i], bins)

            #plot data
            ax = plt.subplot(gs[i + j*10])
            plt.bar(data_generated[1][:-1], data_generated[0], label='generated', alpha=0.5, color='red', width = width)
            plt.bar(data_real[1][:-1], data_real[0], label='real', alpha=0.5, color='blue', width = width)
            ax.set_title('Epoch {} Gene {} Time Point {}'.format(str(epoch), j+1,i+1))
            plt.ylim(0,2000)
            plt.legend(loc = 'upper left')
            plt.grid(True)

            #save data
            if(c == 0):
                save_data(data = [list(data_real[1])], paths = [make(root + '/raw data') + '/histogram bins.csv'])
                c = 1
                
            save_data(data = [list(data_real[0])], paths = [make(root + '/raw data') + '/real data epoch{} gene{} time{}.csv'.format(str(epoch).zfill(4), str(j+1).zfill(2), str(i+1).zfill(2))])
            save_data(data = [list(data_generated[0])], paths = [make(root + '/raw data') + '/generated data epoch{} gene{} time{}.csv'.format(str(epoch).zfill(4), str(j+1).zfill(2), str(i+1).zfill(2))])


    plt.tight_layout()
    plt.savefig(root + '/time resolved histogram epoch{}.png'.format(str(epoch).zfill(4)))
    fig.clf()
    plt.close(fig)
    del data_real
    del data_generated
    gc.collect()


def time_resolved_scatter(data_real, data_gan, root, epoch = 0, num_genes = 3, size = 1, marker = 'o', time_points = 10):
    '''
        data_real: 60k real data points (60k, 10, 3)
        data_gan : 60k generated data points (60k, 10, 3) 
        root     : folder to save scatter plots in
        epoch    : epoch number
        num_genes: number of genes
        size     : size of marks
        marker   : marker type for scatter plot
        time_points : number of time points to plot
    '''

    real_genes = np.squeeze(np.split(data_real, 3, 2))
    generated_genes = np.squeeze(np.split(data_gan, 3, 2))
    real_genes_time = [np.squeeze(np.split(i, 10, 1)) for i in real_genes]
    generated_genes_time = [np.squeeze(np.split(i, 10, 1)) for i in generated_genes]

    fig = plt.figure(figsize=(40, 14))
    gs = gridspec.GridSpec(3,9)
    c = 0

    for i in range(num_genes):
        for j in range(time_points - 1):
            data_real_x = real_genes_time[i][j]
            data_real_y = real_genes_time[i][j+1]

            data_generated_x = generated_genes_time[i][j]
            data_generated_y = generated_genes_time[i][j+1]

            ax = plt.subplot(gs[j + i*9])

            ax.scatter(data_generated_x,
                       data_generated_y,
                       color = 'Red',
                       label = 'generated',
                       marker = marker,
                       s = size,
                       alpha = 0.2)
            
            ax.scatter(data_real_x,
                       data_real_y,
                       color = 'Blue',
                       label = 'real',
                       marker = marker,
                       s = size,
                       alpha = 0.2)
            
            red_patch = mpatches.Patch(color='red', label='Generated Data')
            blue_patch = mpatches.Patch(color='blue', label='Real Data')
            ax.set_title('Epoch {} Gene {}'.format(str(epoch), i+1))
            ax.set_xlabel('time {}'.format(j+1))
            ax.set_ylabel('time {}'.format(j+2))
            ax.grid(True)
            ax.axis('equal')
            ax.set_xlim([-12,2])
            ax.set_ylim([-12,2])
            ax.legend(handles = [red_patch, blue_patch])
                
    plt.tight_layout()
    plt.savefig(root + '/time resolved scatter plot{}.png'.format(str(epoch).zfill(4)))
    fig.clf()
    plt.close(fig)
    del data_real_x, data_real_y, data_generated_x, data_generated_y
    gc.collect()



def tsne_plotter_file(data_real, data_gan, root, size = 1000, marker_size = 0.2, marker = 'o'):
    '''
        data_real: shape (size, 10,3)
        data_gan : shape (size, 10, 3)
        root: folder to save data in
        size: size of real/gan data to be plotted
        marker_size : size of marks
        marker : marker type for scatter plot
    '''

    for i in list_i:
        logger.restore(sess, expt_name + path_check.format(str(1*100).zfill(4)))
        data_gan = sess.run(generator(z_g), feed_dict={z_g :sample(size,100)}).reshape(size,3,10).transpose(0,2,1)
        r = [np.squeeze(np.split(data_real, 3, axis = 2))]
        g = [np.squeeze(np.split(data_gan, 3, axis = 2))]

        tsne = TSNE(2)

        genes = [np.concatenate((np.concatenate((i, np.ones([size,1])), axis=1), np.concatenate((j, np.zeros([size,1])), axis=1)),axis=0) for i,j in zip(r,g)]

        results = [tsne.fit_transform(i) for i in genes]

        fig = plt.figure(figsize = (15, 3))
        gs = gridspec.GridSpec(1,3)

        for j,k in enumerate(results):
            ax = plt.subplot(gs[j])

            for p,q in enumerate(zip(['blue', 'red'], ['real', 'fake'])):
                ax.scatter(k[p*size: p*size + size, 0],
                           k[p*size:p*size + size, 1],
                           c = q[0],
                           label = q[1],
                           marker = marker,
                           s = marker_size)

            ax.set_title('Gene {}'.format(j + 1))
            ax.legend()

            save_data(data = [list(data_real[0])], paths = [make(root + '/raw data') + '/real data epoch{} gene{} time{}.csv'.format(str(epoch).zfill(4), str(j+1).zfill(2), str(i+1).zfill(2))])

        plt.tight_layout()
        plt.savefig(root + '/tsne{}.png'.format(str(1*100).zfill(4)))
        plt.close()



def save_checkpoint(saver, sess, e_metrics, path = '.'):
    '''
        Save a re-loadable checkpoint of the model, complete with model checkpoint, mmds, losses (G,D), medians(1,2,3,avg)
        saver: Tensorflow sessions saver object
        sess: Tensorflow session object
        e_metrics: list of metrics
        path: base path for saving the checkpoint
    '''

    #save model checkpoint
    model_path = path + '/final_checkpoint'
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    saver.save(sess, model_path + '/model.ckpt')


    #save mmds, losses(G,D), medians(1,2,3,avg)
    e_metric_path = path + '/results/end point metrics/end point metrics raw'
    if not os.path.exists(e_metric_path):
        os.makedirs(e_metric_path)

    save_data(e_metrics, root = e_metric_path)


def load_checkpoint(saver, sess, path = '.'):
    '''
        Loads a checkpoint of the model, complete with model checkpoint, mmds, losses (G,D), medians(1,2,3,avg)
        saver: Tensorflow sessions saver object
        sess: Tensorflow session object
        path: base path for loading the checkpoint
    '''

    #load model checkpoint
    model_path = path + '/model_checkpoint'
    saver.restore(sess, model_path + '/model.ckpt')

    #load mmds, losses(G,D), medians(1,2,3,avg)
    e_metric_path = path + '/end point metrics'
    return load_data(root = e_metric_path)

