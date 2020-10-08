
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import graphviz
import itertools
import seaborn as sn


def partition(x):
    """
    Partition the column vector x into subsets indexed by its unique values (v1, ... vk)

    Returns a dictionary of the form
    { v1: indices of x == v1,
      v2: indices of x == v2,
      ...
      vk: indices of x == vk }, where [v1, ... vk] are all the unique values in the vector z.
    """
    ## I had no need of using partition function since my id3 implementation is handling it with help of dataframes and mutual_information function


def entropy(y):
    """
    Compute the entropy of a vector y by considering the counts of the unique values (v1, ... vk), in z

    Returns the entropy of z: H(z) = p(z=v1) log2(p(z=v1)) + ... + p(z=vk) log2(p(z=vk))
    y: series
    """
    total = len(y)
    result = 0  
    for x in np.unique(y):
        cm_term = len(y[y==x])/total
        log_term = np.log2(cm_term)
        result += cm_term*log_term
    
    return -result
        
    
def mutual_information(x, y):
    """
    Compute the mutual information between a data column (x) and the labels (y). The data column is a single attribute
    over all the examples (n x 1). Mutual information is the difference between the entropy BEFORE the split set, and
    the weighted-average entropy of EACH possible split.

    Returns the mutual information: I(x, y) = H(y) - H(y | x)
    
    """
    if len(x) == 0:
        return 0
    
    Hy = entropy(y.iloc[:,0])
    
    
    S = len(y)
    
    sec_term = 0
    
    for values in np.unique(x):
        col = x.columns[0]
        x_dataf = pd.DataFrame(x[col][x[col]==values])
        entxy = entropy(y.loc[x_dataf.index,0])
        sec_term += (len(x_dataf)/S)*entxy

    return Hy - sec_term
        

def id3(x, y, attribute_value_pairs=None, depth=0, max_depth=5):
    """
    Implements the classical ID3 algorithm given training data (x), training labels (y) and an array of
    attribute-value pairs to consider. This is a recursive algorithm that depends on three termination conditions
        1. If the entire set of labels (y) is pure (all y = only 0 or only 1), then return that label
        2. If the set of attribute-value pairs is empty (there is nothing to split on), then return the most common
           value of y (majority label)
        3. If the max_depth is reached (pre-pruning bias), then return the most common value of y (majority label)
    Otherwise the algorithm selects the next best attribute-value pair using INFORMATION GAIN as the splitting criterion
    and partitions the data set based on the values of that attribute before the next recursive call to ID3.

    The tree we learn is a BINARY tree, which means that every node has only two branches. The splitting criterion has
    to be chosen from among all possible attribute-value pairs. That is, for a problem with two features/attributes x1
    (taking values a, b, c) and x2 (taking values d, e), the initial attribute value pair list is a list of all pairs of
    attributes with their corresponding values:
    [(x1, a),
     (x1, b),
     (x1, c),
     (x2, d),
     (x2, e)]
     If we select (x2, d) as the best attribute-value pair, then the new decision node becomes: [ (x2 == d)? ] and
     the attribute-value pair (x2, d) is removed from the list of attribute_value_pairs.

    The tree is stored as a nested dictionary, where each entry is of the form
                    (attribute_index, attribute_value, True/False): subtree
    * The (attribute_index, attribute_value) determines the splitting criterion of the current node. For example, (4, 2)
    indicates that we test if (x4 == 2) at the current node.
    * The subtree itself can be nested dictionary, or a single label (leaf node).
    * Leaf nodes are (majority) class labels

    Returns a decision tree represented as a nested dictionary, for example
    {(4, 1, False):
        {(0, 1, False):
            {(1, 1, False): 1,
             (1, 1, True): 0},
         (0, 1, True):
            {(1, 1, False): 0,
             (1, 1, True): 1}},
     (4, 1, True): 1}
    """

    if len(x) == 0:
        return 0
    if y.nunique()[0]==1:
        return y.iloc[0,0]
    
    try:
        if len(attribute_value_pairs)==0 or depth == max_depth:
            counts = y[0].value_counts().to_dict()
            return list(counts.keys())[0]
    except TypeError:
        attribute_value_pairs = []
        for i in range(0,len(x.columns)):
            dict_xcol = x[i].value_counts().to_dict()
            for k in dict_xcol.keys():
                attribute_value_pairs.append((i,k))
                
    
    remaining_column = []
    for item in attribute_value_pairs:
        remaining_column.append(item[0])
    remaining_column[:] = list(set(remaining_column))
    
    
    max_mutual_i = -1
    for col_attr in remaining_column:
        mutual_i_result = mutual_information(pd.DataFrame(x[col_attr][:]),y)
        if mutual_i_result > max_mutual_i:
            best_attr = col_attr
            max_mutual_i = mutual_i_result
        
                
    max_gain = -1
    for index,data in enumerate(attribute_value_pairs):
        if data[0]==best_attr:            
            col = data[0]
            value = data[1]
            gain = mutual_information(pd.DataFrame(x[col][x[col] == value]),y)
            if gain >= max_gain:
                attr = col
                val = value
                split = index
                max_gain = gain
    
    
    attribute_value_pairs[:] = attribute_value_pairs[0:split] + attribute_value_pairs[split+1:]
    
    dict_decision_tree = {}
    
    
    
    dict_decision_tree[(attr,val,True)] = id3(x[x[attr]==val],y[x[attr]==val],attribute_value_pairs,depth+1,max_depth)

    dict_decision_tree[(attr,val,False)] = id3(x[x[attr]!=val],y[x[attr]!=val],attribute_value_pairs,depth+1,max_depth)
        
    return dict_decision_tree    
        

def predict_example(x, tree):
    
    if type(tree) != dict:
        return tree
    
    key = list(tree.keys())[0]
    node = key[0]
    value = key[1]
    if value==x[node]:
        tree = predict_example(x,tree[(node,value,True)])
    else:
        tree = predict_example(x,tree[(node,value,False)])
    return tree

    """
    Predicts the classification label for a single example x using tree by recursively descending the tree until
    a label/leaf node is reached.

    Returns the predicted label of x according to tree
    """



def compute_error(y_true, y_pred):
    """
    Computes the average error between the true labels (y_true) and the predicted labels (y_pred)

    Returns the error = (1/n) * sum(y_true != y_pred)
    """
    sum_error = 0
    for i in range(0,len(y_pred)):
        if y_true[i]!=y_pred[i]:
            sum_error +=1
    
    return (1/len(y_pred))*sum_error
        
    
    raise Exception('Function not yet implemented!')


def pretty_print(tree, depth=0):
    """
    Pretty prints the decision tree to the console. Use print(tree) to print the raw nested dictionary representation
    DO NOT MODIFY THIS FUNCTION!
    """
    if depth == 0:
        print('TREE')

    for index, split_criterion in enumerate(tree):
        sub_trees = tree[split_criterion]

        # Print the current node: split criterion
        print('|\t' * depth, end='')
        print('+-- [SPLIT: x{0} = {1} {2}]'.format(split_criterion[0], split_criterion[1], split_criterion[2]))

        # Print the children
        if type(sub_trees) is dict:
            pretty_print(sub_trees, depth + 1)
        else:
            print('|\t' * (depth + 1), end='')
            print('+-- [LABEL = {0}]'.format(sub_trees))


def render_dot_file(dot_string, save_file, image_format='png'):
    """
    Uses GraphViz to render a dot file. The dot file can be generated using
        * sklearn.tree.export_graphviz()' for decision trees produced by scikit-learn
        * to_graphviz() (function is in this file) for decision trees produced by  your code.
    DO NOT MODIFY THIS FUNCTION!
    """
    if type(dot_string).__name__ != 'str':
        raise TypeError('visualize() requires a string representation of a decision tree.\nUse tree.export_graphviz()'
                        'for decision trees produced by scikit-learn and to_graphviz() for decision trees produced by'
                        'your code.\n')

    # Set path to your GraphViz executable here
    os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
    graph = graphviz.Source(dot_string)
    graph.format = image_format
    graph.render(save_file, view=True)


def to_graphviz(tree, dot_string='', uid=-1, depth=0):
    """
    Converts a tree to DOT format for use with visualize/GraphViz
    DO NOT MODIFY THIS FUNCTION!
    """

    uid += 1       # Running index of node ids across recursion
    node_id = uid  # Node id of this node

    if depth == 0:
        dot_string += 'digraph TREE {\n'

    for split_criterion in tree:
        sub_trees = tree[split_criterion]
        attribute_index = split_criterion[0]
        attribute_value = split_criterion[1]
        split_decision = split_criterion[2]

        if not split_decision:
            # Alphabetically, False comes first
            dot_string += '    node{0} [label="x{1} = {2}?"];\n'.format(node_id, attribute_index, attribute_value)

        if type(sub_trees) is dict:
            if not split_decision:
                dot_string, right_child, uid = to_graphviz(sub_trees, dot_string=dot_string, uid=uid, depth=depth + 1)
                dot_string += '    node{0} -> node{1} [label="False"];\n'.format(node_id, right_child)
            else:
                dot_string, left_child, uid = to_graphviz(sub_trees, dot_string=dot_string, uid=uid, depth=depth + 1)
                dot_string += '    node{0} -> node{1} [label="True"];\n'.format(node_id, left_child)

        else:
            uid += 1
            dot_string += '    node{0} [label="y = {1}"];\n'.format(uid, sub_trees)
            if not split_decision:
                dot_string += '    node{0} -> node{1} [label="False"];\n'.format(node_id, uid)
            else:
                dot_string += '    node{0} -> node{1} [label="True"];\n'.format(node_id, uid)

    if depth == 0:
        dot_string += '}\n'
        return dot_string
    else:
        return dot_string, node_id, uid
    
  

if __name__ == '__main__':
    # Load the training data
    M = np.genfromtxt('./monks-3.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = pd.DataFrame(M[:, 0])
    Xtrn = pd.DataFrame(M[:, 1:])

    # Load the test data
    M = np.genfromtxt('./monks-3.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = pd.DataFrame(M[:, 0])
    Xtst = pd.DataFrame(M[:, 1:])
    

    # Learn a decision tree of depth 3
    decision_tree = id3(Xtrn, ytrn, max_depth=10)

    # Pretty print it to console
    pretty_print(decision_tree)

    # Visualize the tree and save it as a PNG image
    dot_str = to_graphviz(decision_tree)
    render_dot_file(dot_str, './my_learned_tree_depth='+str(10))
    




################################################################################################################
############ Generating Graph for different datasets and varying depths ############################################################################################
    

    # Load the training data
    M = np.genfromtxt('./monks-1.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = pd.DataFrame(M[:, 0])
    Xtrn = pd.DataFrame(M[:, 1:])

    # Load the test data
    M = np.genfromtxt('./monks-1.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = pd.DataFrame(M[:, 0])
    Xtst = pd.DataFrame(M[:, 1:])
    
    
    
    
    
    M = np.genfromtxt('./monks-2.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = pd.DataFrame(M[:, 0])
    Xtrn = pd.DataFrame(M[:, 1:])

    # Load the test data
    M = np.genfromtxt('./monks-2.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = pd.DataFrame(M[:, 0])
    Xtst = pd.DataFrame(M[:, 1:])
    
    
    
    
    
    
    M = np.genfromtxt('./monks-3.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = pd.DataFrame(M[:, 0])
    Xtrn = pd.DataFrame(M[:, 1:])

    # Load the test data
    M = np.genfromtxt('./monks-3.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = pd.DataFrame(M[:, 0])
    Xtst = pd.DataFrame(M[:, 1:])
    
    
    
    
    
    
    final_trn_err = []
    final_tst_err = []
    depth_arr = [i for i in range(1,11)]
    for dp in range(1,11):
        
        decision_tree = id3(Xtrn, ytrn, max_depth=dp)
        
        ytrn_pred = []
        for i in range(0,len(Xtrn)):
            ytrn_pred.append(predict_example(Xtrn.iloc[i],decision_tree))   
        trn_err = compute_error(list(ytrn.iloc[:][0]), ytrn_pred)
        
        
        
        ytst_pred = []
        for i in range(0,len(Xtst)):
            ytst_pred.append(predict_example(Xtst.iloc[i],decision_tree))
        tst_err = compute_error(list(ytst.iloc[:][0]), ytst_pred)
        
        
        final_trn_err.append(trn_err)
        final_tst_err.append(tst_err)
        
    

    
    
    plt.figure()
    plt.plot(depth_arr,final_trn_err,label="Training error")
    plt.plot(depth_arr,final_tst_err,label="Testing error")
    plt.xlabel("Depth of tree")
    plt.ylabel("Error")
    
    plt.legend()
    plt.show()




###########################################################################################################################
################# Visualizing our results using a confusion matrix ##############################################################################################
   
     
   
    
   
    # Load the training data
    M = np.genfromtxt('./monks-1.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = pd.DataFrame(M[:, 0])
    Xtrn = pd.DataFrame(M[:, 1:])

    # Load the test data
    M = np.genfromtxt('./monks-1.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = pd.DataFrame(M[:, 0])
    Xtst = pd.DataFrame(M[:, 1:])

    ytst_nparray = M[:,0]
    


    for dp in range(1,6,2):
        decision_tree = id3(Xtrn, ytrn, max_depth=dp)
        
        y_pred = []
        for i in range(0,len(Xtst)):
            y_pred.append(predict_example(Xtst.iloc[i],decision_tree))
        
        cf_matrix = confusion_matrix(ytst_nparray,y_pred)
        
        
        tst_error = compute_error(list(ytst.iloc[:][0]), list(y_pred))
        
        
        print('Test Error = {0:4.2f}%.'.format(tst_error * 100))
        
        
        dot_str = to_graphviz(decision_tree)
        render_dot_file(dot_str, './cf_decision_tree of depth='+str(dp))
        
        
        
        plt.figure()

        ax = sn.heatmap(cf_matrix,annot = True,fmt = 'g')
        ax.set_ylim([0,2])
        ax.invert_yaxis()
        print("Confusion Matrix for depth = "+str(dp)+"",confusion_matrix(ytst_nparray,y_pred))




        

###########################################################################################################################################################
################### Visualizing the results generated by scikit-learn's decision tree classifier ###########################################################################################################################
        
    M = np.genfromtxt('./monks-1.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    # Load the test data
    M = np.genfromtxt('./monks-1.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]
    
    


    for dp in range(1,6,2):
        decision_tree = DecisionTreeClassifier(max_depth=dp,criterion='entropy')
        
        decision_tree.fit(Xtrn,ytrn)
        y_pred = decision_tree.predict(Xtst)
        
        tst_error = compute_error(list(ytst), list(y_pred))
        
        
        print('Test Error = {0:4.2f}%.'.format(tst_error * 100))
        
        
        
        cf_matrix = confusion_matrix(ytst,y_pred)
        
        
        plt.figure()
        ax = sn.heatmap(cf_matrix,annot = True,fmt = 'g')
        ax.set_ylim([0,2])
        ax.invert_yaxis()
        print("Confusion Matrix for depth = "+str(dp)+"",confusion_matrix(ytst,y_pred))

        
        
        
################################################################################################################################################################################
############################## Using a different dataset for our model #########################################################################################
        
    M = np.genfromtxt('./tic-tac-toe.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = pd.DataFrame(M[:, 0])
    Xtrn = pd.DataFrame(M[:, 1:])

    # Load the test data
    M = np.genfromtxt('./tic-tac-toe.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = pd.DataFrame(M[:, 0])
    Xtst = pd.DataFrame(M[:, 1:])
    
    
    
    ytst_nparray = M[:,0]#compute error takes a list of numpy array
    
    
    
    for dp in range(1,6,2):
        decision_tree = id3(Xtrn, ytrn, max_depth=dp)
        
        y_pred = []
        for i in range(0,len(Xtst)):
            y_pred.append(predict_example(Xtst.iloc[i],decision_tree))
            
            
        tst_error = compute_error(list(ytst_nparray), list(y_pred))
        
        
        print('Test Error = {0:4.2f}%.'.format(tst_error * 100))
        
            
        
        cf_matrix = confusion_matrix(ytst_nparray,y_pred)
        
        plt.figure()
        
        ax = sn.heatmap(cf_matrix,annot = True,fmt = 'g')
        ax.set_ylim([0,2])
        # ax.invert_yaxis()
        print("Confusion Matrix for depth = "+str(dp)+"",confusion_matrix(ytst_nparray,y_pred))

        dot_str = to_graphviz(decision_tree)
        render_dot_file(dot_str, './e_id3_decision_tree_of_depth='+str(dp))






    



    M = np.genfromtxt('./tic-tac-toe.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    # Load the test data
    M = np.genfromtxt('./tic-tac-toe.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]
    
    


    for dp in range(1,6,2):
        decision_tree = DecisionTreeClassifier(max_depth=dp,criterion='entropy')
        
        decision_tree.fit(Xtrn,ytrn)
        y_pred = decision_tree.predict(Xtst)
        
        tst_error = compute_error(list(ytst), list(y_pred))
        
        
        print('Test Error = {0:4.2f}%.'.format(tst_error * 100))
        
        
        
        cf_matrix = confusion_matrix(ytst,y_pred)
        
        
        plt.figure()
        ax = sn.heatmap(cf_matrix,annot = True,fmt = 'g')
        ax.set_ylim([0,2])
        ax.invert_yaxis()
        print("Confusion Matrix for depth = "+str(dp)+"",confusion_matrix(ytst,y_pred))









