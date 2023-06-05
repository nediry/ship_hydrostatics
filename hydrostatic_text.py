import numpy as np
from scipy.integrate import cumtrapz

length = 113
breadth = 17.38
draft = 6.68
offset = np.loadtxt('s60_cb70.txt', dtype=float)
row, col = offset.shape

pn = np.array([0, .5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9.5, 10]) * length / 10
wl0 = np.array([0, .3, 1, 2, 3, 4, 4 * 1.325, 4 * 1.65]) * draft / 4 #PRU
wl = np.array([0, .5, 1, 2, 3, 4, 4 * 1.325, 4 * 1.65]) * draft / 4 #PRU
# wl0 = np.array([0, .3, 1, 2, 3, 4, 5, 6]) * draft / 4  #YTU
# wl = np.array([0, .5, 1, 2, 3, 4, 5, 6]) * draft / 4  #YTU
# wl0 = np.array([0, .3, 1, 2, 3, 4, 4 * 1.35, 4 * 1.7]) * draft / 4 #ITU
# wl = np.array([0, .5, 1, 2, 3, 4, 4 * 1.35, 4 * 1.7]) * draft / 4 #ITU

for i in range(row):
    offset[i, :] = np.interp(wl, wl0, offset[i, :])

offset *= breadth / 2
alan = np.zeros((row, col))   # BON-JEAN ALANLARI
for i in range(row):
    alan[i, 1:] = 2 * cumtrapz(offset[i, :], wl)

# wl2 = np.array([0, .5, 1, 2, 3, 4, 4 * 1.35, 4 * 1.7]) * 30 #ITU
wl2 = np.array([0, .5, 1, 2, 3, 4, 4 * 1.325, 4 * 1.65]) * 32.9416453 #PRU
# wl2 = np.array([0, .5, 1, 2, 3, 4, 5, 6]) * 32.9416453  #YTU
alan_oran = np.max(alan) / 37.3
alan1 = alan / alan_oran
print('_Curve')
for i in range(8):
    print(str(alan1[1, i] + .5 * 37.4074) + ',' + str(wl2[i]))
print('enter')
print('_Curve')
for i in range(8):
    print(str(alan1[1, i] + 9.5 * 37.4074) + ',' + str(wl2[i]))
print('enter')
alan1 = np.delete(alan1, [1, 11], 0)
dx = 0
for i in range(10):
    print('_Curve')
    for j in range(8):
        print(str(alan1[i, j] + dx) + ',' + str(wl2[j]))
    print('enter')
    dx += 37.4074

moment = np.zeros((row, col))  # BON-JEAN MOMENTLERİ
for i in range(col):
    moment[:, i] = offset[:, i] * wl[i]
for i in range(row):
    moment[i, 1:] = 2 * cumtrapz(moment[i, :], wl)

moment_oran = np.max(moment) / 37.2 + 1
moment1 = moment / moment_oran
print('_Curve')
for i in range(8):
    print(str(moment1[1, i] + .5 * 37.4074) + ',' + str(wl2[i]))
print('enter')
print('_Curve')
for i in range(8):
    print(str(moment1[1, i] + 9.5 * 37.4074) + ',' + str(wl2[i]))
print('enter')
moment1 = np.delete(moment1, [1, 11], 0)
dx = 0
for i in range(10):
    print('_Curve')
    for j in range(8):
        print(str(moment1[i, j] + dx) + ',' + str(wl2[j]))
    print('enter')
    dx += 37.4074

hacim = np.zeros(col)  # HACİM HESABI
for i in range(1, col):
    hacim[i] = np.trapz(alan[:, i], pn)

hacim_oran = hacim[-1] / 374.07 + 2
hacim1 = hacim / hacim_oran
print('_Curve')
for i in range(col):
    print(str(hacim1[i]) + ',' + str(wl2[i]))
print('enter')

deplasman = 1.025 * hacim  # DEPLASMAN HESABI

deplasman_oran = deplasman[-1] / 374.07
deplasman1 = deplasman / deplasman_oran
print('_Curve')
for i in range(col):
    print(str(deplasman1[i]) + ',' + str(wl2[i]))
print('enter')

Awp = np.zeros(col)  # SU HATTI ALANI
for i in range(col):
    Awp[i] = 2 * np.trapz(offset[:, i], pn)

Awp_oran = Awp[-1] / 374.07
Awp1 = Awp / Awp_oran
print('_Curve')
for i in range(col):
    print(str(Awp1[i]) + ',' + str(wl2[i]))
print('enter')

LCF = np.zeros(col)  # YÜZME MERKEZİNİN BOYUNA YERİ (kıçtan)
for i in range(col):  # LCF = MxAwp / Awp
    LCF[i] = 2 * np.trapz(offset[:, i] * pn, pn) / Awp[i]

LCF_oran = length / 374.074
LCF1 = LCF / LCF_oran
print('_Curve')
for i in range(col):
    print(str(LCF1[i]) + ',' + str(wl2[i]))
print('enter')

LCB = np.zeros(col)  # HACİM MERKEZİNİZ BOYUNA YERİ (kıçtan)
for i in range(1, col):  # LCB = Mxalan / hacim
    LCB[i] = np.trapz(alan[:, i] * pn, pn) / hacim[i]

#oran = length / 374.074
LCB1 = LCB / LCF_oran
print('_Curve')
for i in range(1, col):
    print(str(LCB1[i]) + ',' + str(wl2[i]))
print('enter')

T1 =  Awp * 1.025 / 100  # 1cm BATMA TONAJI

T1_oran = 1.5 * 37.4074 / np.max(T1)
T2 = T1 * T1_oran
print('_Curve')
for i in range(col):
    print(str(T2[i]) + ',' + str(wl2[i]))
print('enter')

CM = np.zeros(col)  # ORTA KESİT NARİNLİK KATSAYISI
CM[1:] = alan[6, 1:] / (2 * offset[6, 1:] * wl[1:])

oran = 37.407 * .5 / (np.max(CM) + .1)
CM1 = CM * oran
print('_Curve')
for i in range(1, col):
    print(str(CM1[i]) + ',' + str(wl2[i]))
print('enter')

CB = np.zeros(col)
CB[1:] = hacim[1:] / (length * 2 * offset[6, 1:] * wl[1:]) # BLOK KATSAYISI

CB_oran = 37.4074 * .5 / (np.max(CB) + 1)
CB1 = CB * CB_oran
print('_Curve')
for i in range(1, col):
    print(str(CB1[i]) + ',' + str(wl2[i]))
print('enter')

CW = Awp / (length * 2 * offset[6, :])  # SU HATTI NARİNLİK KATSAYISI

oran = 37.4074 * .5 / (np.max(CW) + .1)
CW1 = CW * oran
print('_Curve')
for i in range(1, col):
    print(str(CW1[i] + 37.4074) + ',' + str(wl2[i]))
print('enter')

CP = np.zeros(col)
CP[1:] = CB[1:] / CM[1:]  # PRİZMATİK KATSAYI

oran = 37.407 * .5 / (np.max(CP) + .1)
CP1 = CP * oran
print('_Curve')
for i in range(1, col):
    print(str(CP1[i] + 37.4074 * .5) + ',' + str(wl2[i]))
print('enter')


KB = np.zeros(col) # HACİM MERKEZİNİN DÜŞEY YERİ
for i in range(1, col):
    KB[i] = np.trapz(moment[:, i], pn) / hacim[i]

KB_oran = 1.5 * 37.4074 / np.max(KB)
KB1 = KB * KB_oran
print('_Curve')
for i in range(1, col):
    print(str(KB1[i]) + ',' + str(wl2[i]))
print('enter')

Icl = np.zeros(col)  # ORTA SİMETRİ EKSENİNE GÖRE ATALET MOMENTİ
for i in range(col):
    Icl[i] = (2 / 3) * np.trapz(offset[:, i]**3, pn)
BM = np.zeros(col)  # ENİNE METESANTR YARIÇAPI
BM[1:] = Icl[1:] / hacim[1:]

BM1 = BM * KB_oran
print('_Curve')
for i in range(1, col):
    print(str(KB1[i] + BM1[i]) + ',' + str(wl2[i]))
print('enter')

Im = np.zeros(col)  # MASTORİYE GÖRE ATALET MOMENTİ
mk = np.array([-5, -4.5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 4.5, 5]) * length / 10
for i in range(8):
    Im[i] = 2 * np.trapz(offset[:, i] * mk**2, pn)
# SU HATTI ALANININ YÜZME MERKEZİNDEN GEÇEN EKSENE GÖRE ATALET MOMENTİ
If = Im - Awp * (LCF - length / 2)**2
BMl = np.zeros(col)  # BOYUNA METASANTR YARIÇAPI
BMl[1:] = If[1:] / hacim[1:]

BMl_oran = np.max(np.abs(BMl)) / 374.074
BMl1 = BMl / BMl_oran
print('_Curve')
for i in range(1, col):
    print(str(BMl1[i]) + ',' + str(wl2[i]))
print('enter')

MCT1 = np.zeros(col)
MCT1[1:] = deplasman[1:] * BMl[1:] / (100 * length)  # BİR SANTİM TRİM MOMENTİ

MTC1_oran = 3.5 * 37.4074 / np.max(MCT1)
MCT2 = MCT1 / MTC1_oran
print('_Curve')
for i in range(1, col):
    print(str(MCT2[i]) + ',' + str(wl2[i]))
print('enter')

# ISLAK YÜZEY ALAN EĞRİSİ

del alan1, moment1, hacim1, deplasman1,  CM1, CB1, CW1, CP1, Awp1, BMl1, T2
del i, j, dx, mk, oran, row, col, wl0, wl, wl2, pn, length, Icl, If, Im
del alan, Awp, BM1, deplasman, hacim, moment, offset, T1, BMl, breadth, draft
