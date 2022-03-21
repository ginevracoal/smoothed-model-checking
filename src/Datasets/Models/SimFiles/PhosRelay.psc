Modelname: PhosRelay
Description: Phosphorelay network

# Set model to run with numbers of individuals
Species_In_Conc: False
Output_In_Conc: False


# Differential Equations as Reactions
R1:
	$pool > B
	k0
	
R2:
	L1+B > L1p+B
	k1*L1*B/N

R3:
	L1p+L2 > L1+L2p
	k2*L1p*L2/N

R4:
	L2p+L3 > L2+L3p
	k3*L2p*L3/N

R5:
	L3p > L3
	k4*L3p/N


	
	
# Parameter values
L1p = 0
L2p = 0
L3p = 0
L1 = 32
L2 = 32
L3 = 32
B = 0
N = 5000

k0 = 0.1
k1 = 1
k2 = 1
k3 = 1
k4 = 1
