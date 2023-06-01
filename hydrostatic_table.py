import numpy as np
from scipy.integrate import cumtrapz

length = 123
breadth = 17.571
draft = 11.949
offset = np.loadtxt('s60_cb70.txt', dtype=float)
row, col = offset.shape

pn = np.array([0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9.5, 10]) * length/10
wl0 = np.array([0, 0.3, 1, 2, 3, 4, 4*1.325, 4*1.65]) * draft/4 #PRU
wl = np.array([0, 0.5, 1, 2, 3, 4, 4*1.325, 4*1.65]) * draft/4 #PRU
# wl0 = np.array([0, 0.3, 1, 2, 3, 4, 5, 6]) * draft/4  #YTU
# wl = np.array([0, 0.5, 1, 2, 3, 4, 5, 6]) * draft/4  #YTU
# wl0 = np.array([0, 0.3, 1, 2, 3, 4, 4*1.35, 4*1.70]) * draft/4 #ITU
# wl = np.array([0, 0.5, 1, 2, 3, 4, 4*1.35, 4*1.70]) * draft/4 #ITU

for i in range(row):
    offset[i, :] = np.interp(wl, wl0, offset[i, :])

print(np.round(offset, 3))

offset *= breadth/2
alan = np.zeros((row, col))   # BON-JEAN ALANLARI
for i in range(row):
    alan[i, 1:] = 2*cumtrapz(offset[i,:], wl)

moment = np.zeros((row, col))  # BON-JEAN MOMENTLERİ
for i in range(col):
    moment[:, i] = offset[:, i] * wl[i]
for i in range(row):
    moment[i, 1:] = 2*cumtrapz(moment[i,:], wl)

hacim = np.zeros(col)  # HACİM HESABI
for i in range(1, col):
    hacim[i] = np.trapz(alan[:, i], pn)

deplasman = 1.025 * hacim  # DEPLASMAN HESABI

AWP = np.zeros(col)  # SU HATTI ALANI
for i in range(col):
    AWP[i] = 2*np.trapz(offset[:, i], pn)

LCF = np.zeros(col)  # YÜZME MERKEZİNİN BOYUNA YERİ (kıçtan)
for i in range(col):  # LCF = MxAwp / Awp
    LCF[i] = np.trapz(offset[:, i] * pn, pn) / AWP[i] - length/2

LCB = np.zeros(col)  # HACİM MERKEZİNİZ BOYUNA YERİ (kıçtan)
for i in range(1, col):  # LCB = Mxalan / hacim
    LCB[i] = np.trapz(alan[:, i] * pn, pn) / hacim[i] - length/2

T1 =  AWP*1.025 / 100  # 1cm BATMA TONAJI

CM = np.zeros(col)  # ORTA KESİT NARİNLİK KATSAYISI
CM[1:] = alan[6, 1:] / (2*offset[6, 1:] * wl[1:])

CB = np.zeros(col)
CB[1:] = hacim[1:] / (length*2*offset[6, 1:] * wl[1:]) # BLOK KATSAYISI

CW = AWP / (length*2 * offset[6, :])  # SU HATTI NARİNLİK KATSAYISI

CP = np.zeros(col)
CP[1:] = CB[1:] / CM[1:]  # PRİZMATİK KATSAYI

KB = np.zeros(col) # HACİM MERKEZİNİN DÜŞEY YERİ
for i in range(1, col):
    KB[i] = np.trapz(moment[:, i], pn) / hacim[i]

Icl = np.zeros(col)  # ORTA SİMETRİ EKSENİNE GÖRE ATALET MOMENTİ
for i in range(col):
    Icl[i] = (2/3) * np.trapz(offset[:, i]**3, pn)
BM = np.zeros(col)  # ENİNE METESANTR YARIÇAPI
BM[1:] = Icl[1:] / hacim[1:]

Im = np.zeros(col)  # MASTORİYE GÖRE ATALET MOMENTİ
for i in range(col):
    Im[i] = np.trapz(offset[:, i] * pn**2, pn)
# SU HATTI ALANININ YÜZME MERKEZİNDEN GEÇEN EKSENE GÖRE ATALET MOMENTİ
If = Im - AWP * (length/2 - LCF)**2
BMl = np.zeros(col)  # BOYUNA METASANTR YARIÇAPI
BMl[1:] = If[1:] / hacim[1:]

MCT1 = np.zeros(col)
MCT1[1:] = deplasman[1:] * BMl[1:] / (100*length)  # BİR SANTİM TRİM MOMENTİ

# ISLAK YÜZEY ALAN EĞRİSİ
def arc_length(x, y):
    npts = len(x)
    arc = np.sqrt((x[1] - x[0])**2 + (y[1] - y[0])**2)
    for k in range(1, npts):
        arc = arc + np.sqrt((x[k] - x[k-1])**2 + (y[k] - y[k-1])**2)
    return arc/2

l = np.zeros((row, col))
for i in range(row):
    for j in range(1, col):
        l[i, j] = round(arc_length(offset[i, j-1:j+1], wl[j-1:j+1]), 3)
S = np.zeros(col)
for i in range(1, col):
    S[i] = S[i-1] + 2*np.trapz(l[:, i], pn)

import pandas as pd
import dataframe_image as dfi
df = pd.DataFrame([hacim, deplasman, LCB, LCF, CB, CM, CP, CW, AWP, T1, MCT1, Icl,
              Im, If, KB, BM, BMl, S], columns=['WL0', 'WL05', 'WL1', 'WL2', 'WL3',
             'WL4', 'WL5', 'WL6'], index=['V', 'dep', 'LCB', 'LCF', 'CB', 'CM', 'CP',
             'CWP', 'AWP', 'T1', 'MT1', 'Icl', 'Im', 'If', 'KB', 'BM', 'BMl', 'S'])
df = df.round(2)
dfi.export(df, 'hydrostatic_table.png')