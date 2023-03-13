import torch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_networkx
import numpy as np
import networkx as nx
# from model.gcn import GCN


class App:
    """ 
        Demo the usefulness of topological indices.
        Loads the pretrained torch model.
        Inputs a networkx graph
        classifies the graph
        returns the useful topological indices from the PCA result    
    """
    # constructor and load the model
    def __init__(self):
        # self.model = GCN()
        # self.model.load_state_dict(torch.load('gnn_model_weights.pth'))
        self.model = torch.load(
            '/Users/luca/dev/git/gitlab/ffhs/luca.hostettler/bt-hostettler/src/gnn_model.pth')
        self.model.eval()
        self.class_keys = {0: 'random', 1: 'smallworld',
                           2: 'scalefree', 3: 'complete', 4: 'line', 5: 'tree'}

    # classify the graph
    def classify(self, G):
        node_labels = np.arange(G.number_of_nodes())
        degrees = nx.degree_centrality(G)
        betweenness = nx.betweenness_centrality(G)
        attrs = dict(zip(node_labels, node_labels))
        nx.set_node_attributes(G, attrs, "label")
        nx.set_node_attributes(G, degrees, "degree")
        nx.set_node_attributes(G, betweenness, "betweenness")
        # convert to torch tensor
        x = from_networkx(graph, group_node_attrs=["label", "betweenness", "degree"])
        test_loader = DataLoader([x], batch_size=1, shuffle=False)
        # run the model
        for x in test_loader:
            out = self.model(x, x.edge_index, x.batch)
            # convert to numpy
            pred = out.argmax(dim=1)  # Use the class with highest probabilit
            # return the result
            return pred

    # get the topological indices
    def get_topological_indices(self, graph):
        # classify the graph
        y = self.classify(graph)
        # get the topological indices
        topological_indices = y[0]
        # return the topological indices
        return topological_indices


# main
if __name__ == '__main__':
    # create an instance of the app
    app = App()
    # create a sample graph using networkx
    graph = nx.path_graph(150)
    # test graph classification
    g_class = app.classify(graph)

    print(f"Found graph class {app.class_keys[g_class]}")
    # get the topological indices
    topological_indices = app.get_topological_indices(graph)
    # print the topological indices
    print(topological_indices)
