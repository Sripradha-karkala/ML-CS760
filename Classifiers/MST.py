#This file implements prim's algorithm for generating
# maximum spanning tree
# Takes an adjacency matrix as input
# outputs the new adjacency matrix and a list of edges
# @author: Sripradha Karkala
#

def maximum_spanning_tree(matrix):

	vertices = range(len(matrix)) # Holds all the vertices names
	visited = []
	#Add the first vertex
	visited.append(0)
	vertices.remove(0)
	start_vertex = 0
	# Check for zero and stuff
	#Initializing the matrix to -1 indicating no edges initially
	adj_matrix = [[-1 for x in range(len(matrix))] for y in range(len(matrix))]
	edges = []

	while len(vertices) > 0:
		maximum = 0
		max_vertex = -1
		parent = -1
		#print vertices

		for i in visited:
			for j in vertices:
				if maximum < matrix[i][j]:
					maximum = matrix[i][j]
					max_vertex = j
					parent = i
				elif maximum == matrix[i][j] and (max_vertex > j or i < parent):
					maximum = matrix[i][j]
					max_vertex = j
					parent = i
		vertices.remove(max_vertex)
		visited.append(max_vertex)
		edges.append([parent, max_vertex])
		adj_matrix[parent][max_vertex] = maximum
	return adj_matrix, edges
#end of function










