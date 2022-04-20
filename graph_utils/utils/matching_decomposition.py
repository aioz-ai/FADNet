import networkx as nx


def matching_decomposition(graph):
    """
    Implementing Misra & Gries edge coloring algorithm;
    The coloring produces uses at most Delta +1 colors, where Delta  is the maximum degree of the graph;
    By Vizing's theorem it uses at most one color more than the optimal for all others;
     See http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.24.4452 for details
    :param graph: nx.Graph()
    :return: - List of matching; each matching is an nx.Graph() representing a sub-graph of "graph"
             - list of laplacian matrices, a laplacian matrix for each matching
    """
    # Initialize the graph with a greedy coloring of less then degree + 1 colors
    nx.set_edge_attributes(graph, None, 'color')

    # edge coloring
    for u, v in graph.edges:
        if u != v:
            graph = color_edge(graph, u, v)

    # matching decomposition
    matching_list = get_matching_list_from_graph(graph)

    # compute laplacian matrices
    laplacian_matrices = [nx.laplacian_matrix(matching, nodelist=graph.nodes(), weight=None).toarray()
                          for matching in matching_list]

    return matching_list, laplacian_matrices


def get_matching_list_from_graph(graph):
    """
    
    :param graph: nx.Graph(); each edge should have an attribute "color"
    :return: List of matching; each matching is an nx.Graph() representing a sub-graph of "graph"
    """
    degree = get_graph_degree(graph)
    colors = [i for i in range(degree + 1)]
    
    matching_list = [nx.Graph() for _ in colors]

    for (u, v, data) in graph.edges(data=True):
        color = data["color"]
        idx = colors.index(color)
        matching_list[idx].add_edges_from([(u, v, data)])

    return matching_list


def color_edge(graph, u, v):
    """
    color edge (u, v) if  uncolored following Misra & Gries procedure;
    :param graph: nx.Graph(); each edge should have an attribute "color"
    :param u: node in "graph"
    :param v: node in "graph"
    :return: nx.Graph() where edge (u, v) has an attribute "color", the generated coloring is valid
    """
    degree = get_graph_degree(graph)
    colors = [i for i in range(degree + 1)]

    if graph.get_edge_data(u, v)["color"] is not None:
        return graph

    else:
        maximal_fan = get_maximal_fan(graph, u, v)

        for color in colors:
            if is_color_free(graph, u, color):
                c = color
                break

        for color in colors:
            if is_color_free(graph, maximal_fan[-1], color):
                d = color
                break

        cd_path = get_cd_path(graph, u, c, d)
        
        sub_fan = get_sub_fan(graph, maximal_fan, u, v, cd_path, d)

        graph = invert_cd_path(graph, cd_path, c, d)

        graph = rotate_fan(graph, sub_fan, u)

        graph.add_edge(u, sub_fan[-1], color=d)

        return graph


def get_maximal_fan(graph, u, v):
    """
    constructs a maximal fan starting from v;
    A fan of a vertex u is a sequence of vertices F[1:k] that satisfies the following conditions:
        1) F[1:k] is a non-empty sequence of distinct neighbors of u
        2) (F[1],u) in  E(G) is uncolored
        3) The color of (F[i+1],u) is free on F[i] for 1 ≤ i < k
    A fan is maximal if it can't be extended;
    :param graph: nx.Graph(); each edge should have an attribute "color"
    :param u: node in "graph"
    :param v: node in "graph"
    :return: list of nodes of "graph" representing a maximal fan starting from "v"
    """
    maximal_fan = [v]

    is_maximal = False

    while not is_maximal:
        is_maximal = True
        for node in graph.neighbors(u):
            edge_color = graph.get_edge_data(u, node)["color"]
            if (node not in maximal_fan) and \
                    is_color_free(graph, maximal_fan[-1], edge_color) and \
                    (edge_color is not None):
                maximal_fan.append(node)
                is_maximal = False
                break

    return maximal_fan


def get_sub_fan(graph, maximal_fan, u, v, cd_path, d):
    """
    constructs a sub-fan of "maximal_fan" such that color `d` is free on its last node;
    :param graph: nx.Graph(); each edge should have an attribute "color"
    :param maximal_fan: maxmial resulting from `get_maximal_fan`
    :param u: node in "graph"
    :param v: node in "graph"
    :param cd_path: nx.Graph() representing a path with edges colored only with c and d
    :param d: integer representing a color
    :return: sub-list of maximal fan such that its last node is free on d 
    """
    sub_fan = [v]
    for node in maximal_fan[1:]:
        if graph.get_edge_data(u, node)['color'] == d:
            break
        else:
            sub_fan.append(node)

    if cd_path.has_node(sub_fan[-1]):
        sub_fan = maximal_fan

    return sub_fan


def rotate_fan(graph, fan, u):
    """

    :param graph: nx.Graph(); each edge should have an attribute "color"
    :param fan: list of nodes of "graph" representing a fan
    :param u: node in "graph"
    :return:
    """
    for idx in range(len(fan)-1):
        current_edge = (u, fan[idx])
        next_edge = (u, fan[idx+1])
        color = graph.get_edge_data(*next_edge)["color"]
        graph.add_edge(*current_edge, color=color)

    graph.add_edge(u, fan[-1], color=None)

    return graph


def is_color_free(graph, node, color):
    """
    check if the color is free on a vertex;
    a color is said to be incident on a vertex if an edge incident on that vertex has that color;
     otherwise, the color is free on that vertex
    :param graph: graph: nx.Graph(); each edge should have an attribute "color"
    :param node: node of "graph"
    :param color: integer smaller then the degree of "graph" or None
    :return: boolean True if "color" is free on "node" and False otherwise
    """
    for neighbor in graph.neighbors(node):
        current_color = graph.get_edge_data(node, neighbor)["color"]

        if current_color == color:
            return False

    return True


def get_cd_path(graph, u, c, d):
    """
    Construct cd-path; a path that includes vertex u, has edges colored only c or d , and is maximal
    :param graph: graph: nx.Graph(); each edge should have an attribute "color"
    :param u: node of "graph"
    :param c:  integer smaller then the degree of "graph" or None; represents a color
    :param d: integer smaller then the degree of "graph" or None; represents a color
    :return: List of nodes of "graph" representing a cd-path
    """
    path = nx.Graph()

    current_color = d
    current_node = u
    is_maximal = False

    while not is_maximal:
        is_maximal = True
        for neighbor in graph.neighbors(current_node):

            try:
                color = graph.get_edge_data(current_node, neighbor)["color"]
            except:
                color = None

            if color == current_color:
                path.add_edge(current_node, neighbor)
                current_node = neighbor
                is_maximal = False
                if current_color == c:
                    current_color = d
                else:
                    current_color = c
                break

    return path


def invert_cd_path(graph, path, c, d):
    """
    Switch the colors of the edges on the cd-path: c to d and d to c.
    :param graph: nx.Graph(); each edge should have an attribute "color"
    :param path: nx.Graph() representing cd-path
    :param c: integer smaller then the degree of "graph" or None; represents a color
    :param d: integer smaller then the degree of "graph" or None; represents a color
    :return: graph with switched colors
    """
    for edge in path.edges:
        current_color = graph.get_edge_data(*edge)["color"]
        if current_color == c:
            graph.add_edge(*edge, color=d)
        if current_color == d:
            graph.add_edge(*edge, color=c)

    return graph


def get_graph_degree(graph):
    """
    get maximal degree of nodes of "graph"
    :param graph: nx.Graph()
    :return: integer representing the degree of the graph
    """
    degrees = graph.degree()

    graph_degree = 0
    for _, degree in degrees:
        if degree > graph_degree:
            graph_degree = degree

    return graph_degree


def is_coloring_valid(graph):
    """
    check if the coloring of a graph is valid,
    i.e., two adjacent edges shouldn't have the same color;
    :param graph: nx.Graph() each edge should have an attribute 'color'
    """
    for u, v, data in graph.edges(data=True):
        color = data['color']

        if color is None: continue

        for _, v_, data_ in graph.edges(u, data=True):
            if v_ != v and data_['color'] == color:
                return False

        for _, u_, data_ in graph.edges(v, data=True):
            if u_ != u and data_['color'] == color:
                return False

    return True 


def is_coloring_correct(graph):
    """
    check if the coloring of a graph is correct,
    i.e., two adjacent edges shouldn't have the same color and all edges are colored;
    :param graph: nx.Graph() each edge should have an attribute 'color'
    """
    if is_coloring_valid(graph): 
        for u, v, data in graph.edges(data=True):
            color = data['color']

            if color is None: continue

            for _, v_, data_ in graph.edges(u, data=True):
                if v_ != v and data_['color'] == color:
                    return False

            for _, u_, data_ in graph.edges(v, data=True):
                if u_ != u and data_['color'] == color:
                    return False

        return True
    else: return False 
