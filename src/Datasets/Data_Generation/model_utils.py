def get_model_details(model_name):
    if model_name == "SIR":
        formula = '(F_[100.,120.](I<1.)) & (G_[0.,100.](I > 0))'
        position_dict = {'S':0,'I':1,'R':2}
    elif model_name == "Poisson":
        formula = 'G_[0,1] (N < 4)'
        position_dict = {'N':0}
    elif model_name == "PhosRelay":
        formula = '(G_[0.,600.](L1P >= L3P)) & (F_[600.,1200.](L1P > L3P))'
        #formula = '(G_[0.,200.](L1P >= L3P)) & (G_[600.,1200.](L1P > L3P))'
        position_dict = {'B':0,'L1':1,'L1P':2,'L2':3,'L2P':4,'L3':5,'L3P':6}
    elif model_name == "PrGeEx":    
        #formula = 'F_[0,21000](G_[0,5000]((R<0)&(L>0)))'
        #formula = 'F_[16000,21000]( (D > 0) & (G_[10,2000](D<= 0)) )'
        formula = 'F_[1600,2100]( (D > 0) & (G_[10,200](D<= 0)) )'
        position_dict={'PLac':0, 'PLac':1, 'PLacRNAP':2, 'TrLacZ1':3, 'RbsLacZ':4, 'TrLacZ2':5, 'Ribosome':6, 'RbsRibosome':7, 'TrRbsLacZ':8, 'LacZ':9, 'dgrLacZ':10, 'dgrRbsLacZ':11}
    elif model_name == "PRDeg":
        formula = '(G_[0.,300.](L1P >= L3P)) & (F_[300.,600.](L1P > L3P))'
        position_dict = {'B':0,'L1':1,'L1P':2,'L2':3,'L2P':4,'L3':5,'L3P':6}
    else:
        formula = ''    
        position_dict = {}

    return formula, position_dict


