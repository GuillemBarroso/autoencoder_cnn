import os
import numpy as np
import matplotlib.pyplot as plt

def getStringsSpaceSeparated(line):
    return line[:-1].split(" ")
def getStringsTabSeparated(line):
    return line[:-1].split("\t")

class Mesh():
    def __init__(self):
        self.nodes = None
        self.elems = None
        self.edges = None
        self.dir = None
        self.nElems = None
        self.nNodes = None
        self.nBoundEdges = None
        self.nElemsX = -1
        self.nElemsY = -1

    def getMesh(self, dataset):
        # getMesh assumes that the mesh file is called Th.msh and that contains
        # data separated with spaces

        # Read mesh
        name = 'Th.msh'
        self.dir = r'datasets/{}'.format(dataset)
        mesh = open('{}/{}'.format(self.dir, name),"r")
        lines = mesh.readlines()
        line = getStringsSpaceSeparated(lines[0])

        self.nNodes, self.nElems, self.nBoundEdges = int(line[0]), int(line[1]), int(line[2])

        # Loop on nodes
        # nodes = [[x, y, boundID]_1, [x, y, boundID]_2, ...]
        self.nodes = []
        for i in range(1, self.nNodes+1):
            line = getStringsSpaceSeparated(lines[i])
            newLine = [float(line[0]), float(line[1]), int(line[2])]
            self.nodes.append(newLine)

            # Find number of elements in x and y directions
            if newLine[1] == 0.0:
                self.nElemsX += 1
            if newLine[0] == 1.0:
                self.nElemsY += 1

        # Loop on elements
        # elems = [[n1, n2, n3]_1, [n1, n2, n3]_2, ...]
        self.elems = []
        for i in range(self.nNodes+1, self.nNodes+self.nElems+1):
            line = getStringsSpaceSeparated(lines[i])
            newLine = [int(line[0]), int(line[1]), int(line[2])]
            self.elems.append(newLine)

        # Loop on edges
        # edges = [[n1, n2, boundID]_1, [n1, n2, boundID]_2, ...]
        self.edges = []
        for i in range(self.nNodes+self.nElems+1, self.nNodes+self.nElems+self.nBoundEdges+1):
            line = getStringsSpaceSeparated(lines[i])
            newLine = [int(line[0]), int(line[1]), int(line[2])]
            self.edges.append(newLine)

        mesh.close()

class TxtData():
    def __init__(self, mesh):
        self.mesh = mesh
        self.data = None

    def readFiles(self):
        # Read results for all cases
        files = os.listdir(self.mesh.dir)
        dataAux = []
        for file in files:
            if file[-3:] == 'txt':
                values = self.extractValuesFromFile(file)
                dataFile = self.computePixelsFromP0values(values)
                dataAux.append(dataFile)
        self.data = np.asarray(dataAux)
    
    def extractValuesFromFile(self, file):
        aux = open('{}/{}'.format(self.mesh.dir, file),"r")
        lines = aux.readlines()
        values = []
        for line in lines:
            lineSep = getStringsTabSeparated(line)
            for val in lineSep:
                if val:
                    newVal = float(val)
                    values.append(newVal)

        nElemsAux = values[0]
        assert self.mesh.nElems == nElemsAux, 'Different number of elements when reading nodal values from TXT files'
        aux.close()
        return values[1:]
    
    def computePixelsFromP0values(self, values):
        # Create data array with pixel values combining element values in the same pixel
        # Assume ordering of elements coming from FreeFEM and structured mesh 
        # (first row first with all elements ordered from left to right)
        data = np.asarray([ [0]*self.mesh.nElemsX for i in range(self.mesh.nElemsY)], dtype=float).reshape(self.mesh.nElemsY, self.mesh.nElemsX, 1)
                
        count = 0
        for i in range(self.mesh.nElemsY):
            for j in range(self.mesh.nElemsX):
                data[i,j] = 0.5*(values[count] + values[count+1])
                count += 2

        return data