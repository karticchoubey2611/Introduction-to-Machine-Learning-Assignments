import numpy as np
import array

def equationInTwoVariables(eqn1,eqn2):
    B = ((eqn1[0]*eqn2[0]) - (eqn2[2]*eqn1[0]))/((eqn1[1]*eqn2[0])-(eqn2[1]*eqn1[0]))
    A = (eqn1[2] -eqn1[1]*B)/eqn1[0]
    return (A,B)

def equationInThreeVariables(eqn1,eqn2,eqn3):
    var1 = list(np.array(eqn1)*eqn2[0] - np.array(eqn2)*eqn1[0])
    var2 = list(np.array(eqn1)*eqn3[0] - np.array(eqn3)*eqn1[0])
    var1 = var1[1::]
    var2 = var2[1::]
    B,C = equationInTwoVariables(var1,var2)
    A = (eqn1[3] - eqn1[1]*B - eqn1[2]*C)/(eqn1[0])
    return (A,B,C)
    
    

    
    
                