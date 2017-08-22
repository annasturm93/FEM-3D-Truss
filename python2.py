#(Found out the actual issue with the dof's - model was underdefined - now working on beam elements instead of bars!)

import numpy as np
from numpy.linalg import norm
from scipy.linalg import eigh #return eigenvectors and eigenvalues (frequencies and mode shapes)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
np.set_printoptions(threshold=np.nan)


def setup():
    #define the coordinate-system / definieren des Koordinatensystems
    x_einheitsvektor=np.array([1,0,0])
    y_einheitsvektor=np.array([0,1,0])
    z_einheitsvektor=np.array([0,0,1])
    #define the nodes / definieren der Knotenkoordinaten



    nodes              = { 1:[0,10,0], 2:[0,0,0], 3:[10,5,0], 4:[0,10,10]}
    degrees_of_freedom = { 1:[1,2,3], 2:[4,5,6], 3:[7,8,9], 4:[10,11,12] }
    elements 		   = { 1:[1,3], 2:[2,3], 3:[4,3] }
    restrained_dof     = [1, 2, 3, 4,5,6,10,11,12]
    forces             = { 1:[0,0,0], 2:[0,0,0], 3:[0,-200,0],4:[0,0,0] }

	# material properties - AISI 1095 Carbon Steel (Spring Steel)
    rho   = {1:0.284, 2:0.284, 3:0.284}
    E = {1:30.0e6, 2:30.0e6, 3:30.0e6}
	# geometric properties
    A = {1:1.0, 2:2.0, 3:2.0}


##    nodes=              {1:[0   ,0   ,0   ], 2:[0.63,0   ,0   ], 3:[0.63,0   ,0.62],\
##                         4:[0   ,0   ,0.62], 5:[0   ,0.68,0   ], 6:[0.63,0.68,0   ],\
##                         7:[0.63,0.68,0.62], 8:[0   ,0.68,0.62], 9:[0   ,1.36,0   ],\
##                        10:[0.63,1.36,0   ],11:[0.63,1.36,0.62],12:[0   ,1.36,0.62]}
##    degrees_of_freedom= {1:[1 ,2 ,3 ], 2:[4 ,5 ,6 ], 3:[7 ,8 ,9 ],\
##                         4:[10,11,12], 5:[13,14,15], 6:[16,17,18],\
##                         7:[19,20,21], 8:[22,23,24], 9:[25,26,27],\
##                        10:[28,29,30],11:[31,32,33],12:[34,35,36]}
##    elements=           {1:[1 ,2 ], 2:[2 ,3 ], 3:[4 ,3 ], 4:[1 ,4 ],\
##                         5:[2 ,6 ], 6:[3 ,7 ], 7:[4 ,8 ], 8:[1 ,5 ],\
##                         9:[6 ,7 ],10:[5 ,8 ],11:[6 ,10],12:[7 ,11],\
##                        13:[8 ,12],14:[5 ,9 ],15:[9 ,10],16:[10,11],\
##                        17:[12,11],18:[9,12]}
##    restrained_dof=     [1,2,3,4,5,6,13,14,15,16,17,18,25,26,27,28,29,30]
##    forces=             {1:[0   ,0   ,0   ], 2:[0   ,0   ,0   ], 3:[0   ,0   ,0   ],\
##                         4:[0   ,0   ,0   ], 5:[0   ,0   ,0   ], 6:[0   ,0   ,0   ],\
##                         7:[0   ,0   ,-100   ], 8:[0   ,0   ,0   ], 9:[0   ,0   ,0   ],\
##                         10:[0   ,0   ,0   ],11:[0   ,0   ,0   ],12:[0   ,0   ,0   ]} #Bsp!
##
##    #material properties / Materialeigenschaften
##    #AlMgSi 0,7 F28
##    rho=                {1:2700, 2:2700, 3:2700, 4:2700,\
##                         5:2700, 6:2700, 7:2700, 8:2700,\
##                         9:2700,10:2700,11:2700,12:2700,\
##                        13:2700,14:2700,15:2700,16:2700,\
##                        17:2700,18:2700} #kg/m^3
##    E=                  {1:7e10, 2:7e10, 3:7e10, 4:7e10,\
##                         5:7e10, 6:7e10, 7:7e10, 8:7e10,\
##                         9:7e10,10:7e10,11:7e10,12:7e10,\
##                        13:7e10,14:7e10,15:7e10,16:7e10,\
##                        17:7e10,18:7e10} #N/m^2
##    A=                  {1:38e-4, 2:38e-4, 3:38e-4, 4:38e-4,\
##                         5:38e-4, 6:38e-4, 7:38e-4, 8:38e-4,\
##                         9:38e-4,10:38e-4,11:38e-4,12:38e-4,\
##                        13:38e-4,14:38e-4,15:38e-4,16:38e-4,\
##                        17:38e-4,18:38e-4} #m^2 (Wert fuer MK Profil)

    #degrees of freedom and assertions / Freiheitsgrade und Ueberpruefung
    ndofs= 3* len(nodes)
    assert len(rho)==len(elements)==len(E)==len(A)
    assert len(restrained_dof)<ndofs
    assert len(forces)==len(nodes)

    return {'x_einheitsvektor':x_einheitsvektor, 'y_einheitsvektor':y_einheitsvektor, \
    'z_einheitsvektor':z_einheitsvektor,'nodes':nodes, 'degrees_of_freedom':degrees_of_freedom, \
    'elements':elements,'restrained_dofs':restrained_dof,'forces':forces, \
    'ndofs':ndofs,'rho':rho,'E':E,'A':A}

def plot_truss(nodes, elements,areas):
    #plot nodes in 3d / Plotten der Knoten in 3d
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = [i[0] for i in nodes.values()]
    y = [i[1] for i in nodes.values()]
    z = [i[2] for i in nodes.values()]
    size = 400
    ax.scatter(x, y, z, c='r', marker='o', s=size, zorder=5)

    #plot elements in 3d / Plotten der Elemente in 3d
    for element in elements:
        fromPoint = np.array(nodes[elements[element][0]])
        toPoint = np.array(nodes[elements[element][1]])
        x1 = fromPoint[0]
        y1 = fromPoint[1]
        z1 = fromPoint[2]
        x2 = toPoint[0]
        y2 = toPoint[1]
        z2 = toPoint[2]
        ax.plot([x1, x2], [y1,y2],zs=[z1,z2], c='b', zorder=1)
    plt.show()

def points(element,properties):
    elements =properties['elements']
    nodes = properties['nodes']
    degrees_of_freedom = properties['degrees_of_freedom']

    # find nodes that connects element / Finden der Knoten des Elements
    fromNode = elements[element][0]
    toNode= elements[element][1]

    # find coordinates for each node / Finden der Koordinaten fuerr jeden Knoten
    fromPoint=np.array(nodes[fromNode])
    toPoint = np.array(nodes[toNode])

    #find dofs for each node / Finden der Freiheitsgrade fuer jeden Knoten
    dofs = degrees_of_freedom[fromNode]
    dofs.extend(degrees_of_freedom[toNode])
    dofs = np.array(dofs)

    return fromPoint, toPoint, dofs

def direction_cosine(vec1, vec2):
    return np.dot(vec1,vec2) / (norm(vec1)*norm(vec2))

def rotation_matrix(element_vector, x_axis, y_axis, z_axis):
    #find direction cosines / finden der Richtungscosinuswerte
    x_proj = direction_cosine(element_vector,x_axis)
    y_proj = direction_cosine(element_vector,y_axis)
    z_proj = direction_cosine(element_vector,z_axis)
    return np.array([[x_proj,y_proj,z_proj,0,0,0],[0,0,0,x_proj,y_proj,z_proj]])


def get_matrices(properties):
    # construct global mass- and stiffness matrices / Aufbau der globalen Massen-
    # und Steifigkeitsmatrizen
    ndofs             = properties['ndofs']
    nodes             = properties['nodes']
    elements          = properties['elements']
    forces            = properties['forces']
    areas             = properties['A']
    x_einheitsvektor  = properties['x_einheitsvektor']
    y_einheitsvektor  = properties['y_einheitsvektor']
    z_einheitsvektor  = properties['z_einheitsvektor']

    plot_truss(nodes, elements,areas)

    M = np.zeros((ndofs,ndofs))
    K = np.zeros((ndofs,ndofs))

    for element in elements:
        #find geometry of element / finden der Elementengeometrie
        fromPoint, toPoint, dofs = points(element,properties)
        element_vector           = toPoint - fromPoint

        #find element mass and stiffness matrices - goto 7.1.2/
        #finden der Elementmassen- und Steifigkeitsmatrizen - siehe Kapitel
        #7.1.2 in Masterarbeit
        length = norm(element_vector)
        rho    = properties['rho'][element]
        area   = properties['A'][element]
        E      = properties['E'][element]
        Cm     = rho * area * length / 6.0
        Ck     = E * area / length
        m      = np.array([[2,1],[1,2]])
        k      = np.array([[1,-1],[-1,1]])

        #find rotated mass and stiffnes element matrices / finden der rotierten
        #Elementmassen- und Steifigkeitsmatrizen
        tau = rotation_matrix(element_vector, x_einheitsvektor, y_einheitsvektor, z_einheitsvektor)
        m_r = tau.T.dot(m).dot(tau)
        k_r = tau.T.dot(k).dot(tau)

        #change from element to global coordinates / Transformation von Elementen-
        #in Globalkoordinaten
        index = dofs-1
        B = np.zeros((6,ndofs)) #6: u1,v1,w1,u2,v2,w2
        for i in range(6):
            B[i,index[i]] = 1.0
        M_rG = B.T.dot(m_r).dot(B)
        K_rG = B.T.dot(k_r).dot(B)

        M += Cm * M_rG
        K += Ck * K_rG

    #construct the force vector / Aufbau des Kraftvektors
    F = []
    for f in forces.values():
        F.extend(f)
    F = np.array(F)

    #remove the restrained dofs / Beruecksichtigung der Randbedingungen
    remove_indices = np.array(properties['restrained_dofs']) - 1
    for i in [0,1]:
        M = np.delete(M, remove_indices, axis=i)
        K = np.delete(K, remove_indices, axis=i)
    F = np.delete(F, remove_indices)

    return M, K, F

def get_stresses(properties,X):
    x_axis   = properties['x_einheitsvektor']
    y_axis   = properties['y_einheitsvektor']
    z_axis   = properties['z_einheitsvektor']
    elements = properties['elements']
    E        = properties['E']

    #find the stresses in each member / Finden der Verschiebungen
    stresses = []
    for element in elements:
        fromPoint, toPoint, dofs = points(element, properties)
        element_vector = toPoint - fromPoint

        #get rotation matrix / Transformationsmatrix
        tau = rotation_matrix(element_vector, x_axis,y_axis,z_axis)
        global_displacements = np.array([0,0,0,X[0],X[1],X[2]])
        q = tau.dot(global_displacements)

        #calculate the strains and stresses / Berechnen der Belastung und Spannung
        strain = (q[1] - q[0])/norm(element_vector)
        stress = E[element] * strain
        stresses.append(stress)

    return stresses

def show_results(X,stresses,frequencies):
    print('Nodal Displacements: / Knotenverschiebung:',X)
    print('Stresses: / Spannungen:', stresses)
    print('Frequencies: / Frequenzen:',frequencies)
    print('Displacement magnitude: / Verschiebungsgroesse:',norm(X))

    return


def main():
    #problem setup / Problemstellung
    properties = setup()

    #determine the global matrices / erzeugen der globalen Matrizen
    M, K, F = get_matrices(properties)

    #finding the natural frequencies / finden der Eigenfrequenzen
    evals, evecs=eigh(K,M)
    frequencies=np.sqrt(evals)

    #finding the displacements for each element / berechnen der Verschiebungen fuer jedes Element
    X=np.linalg.inv(K).dot(F)

    #finding the stress in each element / berechnen der Spannungen in jedem Element
    stresses=get_stresses(properties,X)

    #output results / Ausgabe der Ergebnisse
    show_results(X,stresses,frequencies)


if __name__ == '__main__':
    main()
