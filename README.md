#%%
!pip install pydot
#RESNET LIKE ARCHITECTURE
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Conv2D,BatchNormalization,Activation,Add,Flatten,Dense
def residual_block(x,filters,kernel_size=3,stride=1):
    shortcut = x
     # Apply a convolution to the shortcut if dimensions mismatch
    if x.shape[-1] != filters:
        shortcut = Conv2D(filters, kernel_size=1, strides=stride, padding='same')(shortcut)
    x=Conv2D(filters,kernel_size,strides=stride,padding='same')(x)
    x=BatchNormalization()(x)
    x=Activation('relu')(x)

    x=Conv2D(filters,kernel_size,strides=1,padding='same')(x)
    x=BatchNormalization()(x)
    x=Add()([x,shortcut])
    x=Activation('relu')(x)
    return x
input=Input(shape=(64,64,3,))
x=Conv2D(64,(7,7),strides=2,padding='same')(input)
x=BatchNormalization()(x)
x=Activation('relu')(x)
x=residual_block(x,64)
x=residual_block(x,128)
x=Flatten()(x)
outputs=Dense(10,activation='softmax')(x)
model=Model(inputs=input,outputs=outputs)
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import networkx as nx

# Create a directed graph to represent the layers
G = nx.DiGraph()

# Define the layers in sequence
layers = [
    "Input (64,64,3)", "Conv2D (64,7x7)", "BatchNorm", "ReLU",
    "Residual Block 1 (64)", "Residual Block 2 (128)",
    "Flatten", "Dense (10, Softmax)", "Output"
]

# Add nodes (layers)
for i, layer in enumerate(layers):
    G.add_node(i, label=layer)

# Add edges (connections between layers)
for i in range(len(layers) - 1):
    G.add_edge(i, i + 1)

# Draw the model architecture graph
pos = nx.spring_layout(G, seed=42)  # Set layout positioning
labels = {i: G.nodes[i]["label"] for i in G.nodes}

plt.figure(figsize=(10, 6))
nx.draw(G, pos, with_labels=True, labels=labels, node_color="lightblue", edge_color="gray", font_size=10, node_size=2500)
plt.title("ResNet-Like Architecture")
plt.show()



