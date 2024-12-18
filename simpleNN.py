import math
import numpy as np
from sklearn.datasets import make_blobs


class Node(object):
    def __init__(self, value, children=[]):
        self.value = value
        self.backward_grad = lambda: None
        self.grad = 0
        self.children = children
    
    def __add__(self, other):
        if isinstance(other, (int, float)):
            other = Node(other)
        
        if not isinstance(other, Node):
            raise TypeError(f'Unknown type: {type(other)} being added to Node')

        parent = Node(self.value + other.value, [self, other])

        '''
        The following is a python closure, and it stores references to self, other and parent.
        This is similar to the concept of a lambda function in C++, which is actually a class 
        with a single operator() method and it stores references to anything that's captured
        '''
        def backward():
            self.grad += parent.grad
            other.grad += parent.grad
        
        parent.backward_grad = backward      

        return parent
    
    def __sub__(self, other):
        return self + (-1 * other)
    
    def __rsub__(self, other):
        return other + (-1 * self)
    
    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            other = Node(other)

        if not isinstance(other, Node):
            raise TypeError(f'Unknown type: {type(other)} being multiplied to Node')
        
        parent = Node(self.value * other.value, [self, other])

        def backward():
            self.grad += parent.grad * other.value
            other.grad += parent.grad * self.value
               
        parent.backward_grad = backward

        return parent

    def __rmul__(self, other):
        return self.__mul__(other)
    
    def tanh(self):
        e = math.exp(2*self.value)
        t = (e + (-1))/(e + 1)
        parent = Node(t, [self])

        # derivative for tanh is 1 - tanh^2
        # this is easy to prove by substituting y = (exp(2x) + 1)
        def backward():
            self.grad += (1-t**2) * parent.grad

        parent.backward_grad = backward

        return parent

    def relu(self):
        parent = Node(0 if self.value < 0 else self.value, [self])

        def backward():
            if self.value > 0 : 
                self.grad += parent.grad
        
        parent.backward_grad = backward

        return parent

    def calculate_gradient(self):
        # construct a topological graph from the last node and then compute the gradient 1by1
        # order of the graph should be the order in which gradient should be computed, so nodes
        # closer to the output come before the nodes further from the output
        def topological_sort(current_node, visited, output):
            for node in current_node.children:
                if node not in visited:
                    visited.add(node)
                    topological_sort(node, visited, output)
            output.append(current_node)
        graph = []
        topological_sort(self, set(), graph)
        self.grad = 1.0
        for node in graph[::-1]:
            node.backward_grad()

    
    def __repr__(self):
        return f'{self.value}'
    
    def __str__(self):
        return f'{self.value}'


class Neuron(object):
    def __init__(self, input_size):
        self.weights = [Node(np.random.randn()) for _ in range(input_size)]
        self.bias = Node(np.random.randn())
    
    def __call__(self, x):
        return (sum([w*x for w, x in zip(self.weights, x)]) + self.bias).tanh()
    
    def get_params(self):
        return self.weights + [self.bias]


class Layer(object):
    def __init__(self, input_size, output_size):
        self.neurons = [Neuron(input_size) for _ in range(output_size)]
    
    def __call__(self, x):
        return [neuron(x) for neuron in self.neurons]
    
    def get_params(self):
        return [p for neuron in self.neurons for p in neuron.get_params()]


class MLP(object):
    def __init__(self, input_dim, hidden_dims):
        self.layers = []
        dim1 = input_dim
        for dim in hidden_dims:
            self.layers.append(Layer(dim1, dim))
            dim1 = dim
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
        # return [i.tanh() for i in x]

    def get_params(self):
        return [p for layer in self.layers for p in layer.get_params()]
    
    def reset_grad(self):
        for p in self.get_params():
            p.grad = 0


def test_nodes():
    a = Node(4)
    b = Node(6)
    e = Node(2)
    c = a + b
    d = a + e
    z = d * c
    z.grad = 1.0
    z.calculate_gradient()
    
    print(f'a.grad: {a.grad}')
    print(f'b.grad: {b.grad}')
    print(f'c.grad: {c.grad}')
    print(f'd.grad: {d.grad}')
    print(f'e.grad: {e.grad}')


def test_MLP():
    mlp = MLP(3, [4, 4, 1])
    Xs = [
        [2, 1, 0],
        [0, -1, 3],
        [0, .1, 0],
        [4, 0.2, 5]
    ]
    ys = [-1, 1, -1, 1]

    def get_loss():
        y_pred = [mlp(x)[0] for x in Xs]
        return sum(((y1-y2)*(y1-y2) for y1, y2 in zip(ys, y_pred)))

    print(f'No of parameters: {len(mlp.get_params())}')
    learning_rate = 0.05
    for i in range(30):
        loss = get_loss()
        mlp.reset_grad()
        loss.calculate_gradient()
        for p in mlp.get_params():
            p.value -= learning_rate * p.grad
        print(f'loss: {loss}')


def test_MLP_complicated():
    mlp = MLP(2, [32, 32, 1])
    Xs, ys = make_blobs(n_samples=100, centers=2)
    ys = ys*2 - 1 # to convert 0, 1 -> -1, 1
    def get_loss():
        y_pred = [mlp(x)[0] for x in Xs]
        return sum((y1-float(y2)) * (y1-float(y2)) for y1, y2 in zip(y_pred, ys))
    
    learning_rate = 0.002
    for i in range(200):
        loss = get_loss()
        mlp.reset_grad()
        loss.grad = 1.0
        loss.calculate_gradient()
        for p in mlp.get_params():
            p.value -= learning_rate * p.grad
        print(f'loss: {loss}')


def main():
    test_nodes()
    test_MLP()
    test_MLP_complicated()


if __name__ == '__main__':
    main()