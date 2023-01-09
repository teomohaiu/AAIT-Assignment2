
def alpha_weight(epoch):
    T1 = 100
    T2 = 700
    af = 3
    if epoch < T1:
        return 0.0
    elif epoch > T2:
        return af
    else:
         return ((epoch-T1) / (T2-T1))*af