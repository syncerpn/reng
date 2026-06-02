import onnx
import networkx as nx
import numpy as np

# Load the ONNX model
model = onnx.load("/data/nghiant/StackingConv_mass/onnx/StackingConv_32x32_1_3_16_3_1.onnx")
graph = model.graph

# Create a directed graph
G = nx.DiGraph()

# Add nodes (operations)
for node in graph.node:
    G.add_node(node.name, op_type=node.op_type)

# Add edges (based on tensor flows)
output_to_node = {}
for node in graph.node:
    for output in node.output:
        output_to_node[output] = node.name

for node in graph.node:
    for input_tensor in node.input:
        if input_tensor in output_to_node:
            G.add_edge(output_to_node[input_tensor], node.name)

# Convert to adjacency matrix
adj_matrix = nx.to_numpy_array(G, dtype=np.int32)

# Optionally, print or save
print("Adjacency matrix shape:", adj_matrix.shape)
print(adj_matrix)
