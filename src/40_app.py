import os
import pickle
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_networkx
import numpy as np
import networkx as nx
from model.gcn import GCN


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
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.model = GCN()
        self.model.load_state_dict(torch.load(
            dir_path + '/data/graph_classification_model_state.pt'))

        self.model.eval()
        self.class_keys = {0: 'random', 1: 'smallworld',
                           2: 'scalefree', 3: 'complete', 4: 'line', 5: 'tree', 6: 'star'}

        # we need to read the dataset from the pickle file
        with open(dir_path + '/data/useful_topological_indices.pickle', 'rb') as f:
            self.useful_topological_indices_per_class = pickle.load(f)

    def graph_tensor_from_networkx(self, G):
        node_labels = np.arange(G.number_of_nodes())
        degrees = nx.degree_centrality(G)
        betweenness = nx.betweenness_centrality(G)
        density = nx.density(G)
        attrs = dict(zip(node_labels, node_labels))
        nx.set_node_attributes(G, attrs, "label")
        nx.set_node_attributes(G, degrees, "degree")
        nx.set_node_attributes(G, betweenness, "betweenness")
        nx.set_node_attributes(G, density, "density")
        # convert to torch tensor
        x = from_networkx(graph, group_node_attrs=[
                          "label", "betweenness", "degree", "density"])
        test_loader = DataLoader([x], batch_size=1, shuffle=False)

        return test_loader

    # classify the graph
    def classify(self, G) -> int:  # type: ignore
        test_loader = self.graph_tensor_from_networkx(G)
        # run the model
        for data in test_loader:
            out = self.model(data.x, data.edge_index, data.batch)
            # Use the class with highest probability
            pred = out.argmax(dim=1)
            # return the result
            return int(pred[0])

    # get the topological indices
    def get_topological_indices(self, graph_class):
        class_name = app.class_keys[graph_class]
        useful_topological_indices = self.useful_topological_indices_per_class[class_name]
        return useful_topological_indices


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
    topological_indices = app.get_topological_indices(g_class)
    # print the topological indices
    print(
        f"""The most useful topological indices for {app.class_keys[g_class]} graphs are 
        - 1. {topological_indices[0]}
        - 2. {topological_indices[1]}
        - 3. {topological_indices[2]}
        - 4. {topological_indices[3]}""")
