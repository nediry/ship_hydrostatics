import numpy as np
import pandas as pd
import dataframe_image as dfi
from scipy.integrate import cumtrapz

class ShipHydrostatics:    
    def __init__(self, length, breadth, draft, offset, pn, wl):
        self.length = length
        self.breadth = breadth
        self.draft = draft
        self.offset = offset * self.breadth / 2
        self.pn = pn * self.length
        self.wl = wl * self.draft
        
    def offset_show(self):
        col = np.linspace(0, len(self.wl) - 1, len(self.wl))
        row = np.linspace(0, len(self.pn) - 1, len(self.pn))
        df = pd.DataFrame(self.offset, columns = col, index = row)
        return df
        
    def offset_expand(self, row, col):
        wl_new = np.linspace(0, 6, col) * self.draft / 4
        pn_new = np.linspace(0, 10, row) * self.length / 10
        offset2 = np.zeros((self.offset.shape[0], col))
        for i in range(self.offset.shape[0]):
            offset2[i, :] = np.interp(wl_new, self.wl, self.offset[i, :])
        self.offset = np.zeros((row, col))
        for i in range(self.offset.shape[1]):
            self.offset[:, i] = np.interp(pn_new, self.pn, offset2[:, i])
        self.offset = np.round(self.offset, 3)
        self.pn = pn_new
        self.wl = wl_new
    
    def area_calc(self):
        alan = np.zeros((self.offset.shape))   # BON-JEAN ALANLARI
        for i in range(self.offset.shape[0]):
            alan[i, 1:] = 2 * cumtrapz(self.offset[i, :], self.wl)
        return np.round(alan, 3)
    
    def moment_calc(self):
        moment = np.zeros((self.offset.shape))  # BON-JEAN MOMENTLERİ
        for i in range(self.offset.shape[1]):
            moment[:, i] = offset[:, i] * wl[i]
        for i in range(self.offset.shape[0]):
            moment[i, 1:] = 2*cumtrapz(moment[i,:], wl)
        return np.round(moment, 3)
    
    def volume_calc(self):
        alan = self.area_calc()
        hacim = np.zeros(self.offset.shape[1])  # HACİM HESABI
        for i in range(1, self.offset.shape[1]):
            hacim[i] = np.trapz(alan[:, i], self.pn)
        return np.round(hacim, 3)
    
    def deplasman_calc(self):
        hacim = self.volume_calc()
        deplasman = 1.025 * hacim  # DEPLASMAN HESABI
        return deplasman
    
    def Awp_calc(self):
        Awp = np.zeros(self.offset.shape[1])  # SU HATTI ALANI
        for i in range(self.offset.shape[1]):
            Awp[i] = 2 * np.trapz(self.offset[:, i], self.pn)
        return np.round(Awp, 3)
    
    def LCF_calc(self):
        Awp = self.Awp_calc()
        # YÜZME MERKEZİNİN BOYUNA YERİ (kıçtan)
        LCF = np.zeros(self.offset.shape[1])  
        for i in range(self.offset.shape[1]):  # LCF = MxAwp / Awp
            LCF[i] = np.trapz(self.offset[:, i] * self.pn, self.pn) \
                   / Awp[i] - self.length / 2
        return np.round(LCF, 3)
    
    def LCB_calc(self):
        alan = self.area_calc()
        hacim = self.volume_calc()
        # HACİM MERKEZİNİZ BOYUNA YERİ (kıçtan)
        LCB = np.zeros(self.offset.shape[1])
        for i in range(1, self.offset.shape[1]):  # LCB = Mxalan / hacim
            LCB[i] = np.trapz(alan[:, i] * self.pn, self.pn) \
                   / hacim[i] - self.length / 2
        return np.round(LCB, 3)
    
    def T1_calc(self):
        Awp = self.Awp_calc()
        T1 =  Awp * 1.025 / 100  # 1cm BATMA TONAJI
        return T1
    
    def coef_calcs(self):
        # ORTA KESİT NARİNLİK KATSAYISI
        alan = self.area_calc()
        CM = np.zeros(self.offset.shape[1])
        CM[1:] = alan[6, 1:] / (2 * self.offset[6, 1:] * self.wl[1:])
        
        # BLOK KATSAYISI
        hacim = self.volume_calc()
        CB = np.zeros(self.offset.shape[1])
        CB[1:] = hacim[1:] / (self.length* 2 * self.offset[6, 1:] * self.wl[1:])
        
        # SU HATTI NARİNLİK KATSAYISI
        Awp = self.Awp_calc()
        CW = Awp / (self.length * 2 * self.offset[6, :])
        
        # PRİZMATİK KATSAYI
        CP = np.zeros(self.offset.shape[1])
        CP[1:] = CB[1:] / CM[1:]
        return CM, CB, CW, CP
    
    def KB_calc(self):
        moment = self.moment_calc()
        hacim = self.volume_calc()
        KB = np.zeros(self.offset.shape[1]) # HACİM MERKEZİNİN DÜŞEY YERİ
        for i in range(1, self.offset.shape[1]):
            KB[i] = np.trapz(moment[:, i], self.pn) / hacim[i]
        return KB
    
    def BM_calc(self):
        hacim = self.volume_calc()
        # ORTA SİMETRİ EKSENİNE GÖRE ATALET MOMENTİ
        Icl = np.zeros(self.offset.shape[1])
        for i in range(self.offset.shape[1]):
            Icl[i] = (2 / 3) * np.trapz(self.offset[:, i]**3, self.pn)
        BM = np.zeros(self.offset.shape[1])  # ENİNE METESANTR YARIÇAPI
        BM[1:] = Icl[1:] / hacim[1:]
        return Icl, BM
    
    def BMl_calc(self):
        Im = np.zeros(self.offset.shape[1])  # MASTORİYE GÖRE ATALET MOMENTİ
        for i in range(self.offset.shape[1]):
            Im[i] = np.trapz(self.offset[:, i] * self.pn**2, self.pn)
        # SU HATTI ALANININ YÜZME MERKEZİNDEN GEÇEN EKSENE GÖRE ATALET MOMENTİ
        LCF = self.LCF_calc()
        Awp = self.Awp_calc()
        If = Im - Awp * (self.length / 2 - LCF)**2
        BMl = np.zeros(self.offset.shape[1])  # BOYUNA METASANTR YARIÇAPI
        hacim = self.volume_calc()
        BMl[1:] = If[1:] / hacim[1:]
        return Im, If, BMl
    
    def MTC1_calc(self):
        deplasman = self.deplasman_calc()
        _, _, BMl = self.BMl_calc()
        # BİR SANTİM TRİM MOMENTİ
        MCT1 = np.zeros(self.offset.shape[1])
        MCT1[1:] = deplasman[1:] * BMl[1:] / (100 * self.length)
        return MCT1
    
    def S_calc(self):
        # ISLAK YÜZEY ALAN EĞRİSİ
        def arc_length(x, y):
            npts = len(x)
            arc = np.sqrt((x[1] - x[0])**2 + (y[1] - y[0])**2)
            for k in range(1, npts):
                arc = arc + np.sqrt((x[k] - x[k-1])**2 + (y[k] - y[k-1])**2)
            return arc/2

        l = np.zeros((self.offset.shape))
        for i in range(self.offset.shape[0]):
            for j in range(1, self.offset.shape[1]):
                l[i, j] = round(arc_length(self.offset[i, j-1:j+1],
                                           self.wl[j-1:j+1]), 3)
        S = np.zeros(self.offset.shape[1])
        for i in range(1, self.offset.shape[1]):
            S[i] = S[i-1] + 2*np.trapz(l[:, i], self.pn)
        return S
    
    def save_show_table(self):
        hacim = self.volume_calc()
        deplasman = self.deplasman_calc()
        LCB = self.LCB_calc()
        LCF = self.LCF_calc()
        CM, CB, CW, CP = self.coef_calcs()
        Awp = self.Awp_calc()
        T1 = self.T1_calc()
        MCT1 = self.MTC1_calc()
        Icl, BM = self.BM_calc()
        Im, If, BMl = self.BMl_calc()
        KB = self.KB_calc()
        S = self.S_calc()
        df = pd.DataFrame([hacim, deplasman, LCB, LCF, CB, CM, CP, CW, Awp,
             T1, MCT1, Icl, Im, If, KB, BM, BMl, S], columns=['WL0', 'WL05',
             'WL1', 'WL2', 'WL3', 'WL4', 'WL5', 'WL6'], index=['V', 'dep',
             'LCB', 'LCF', 'CB', 'CM', 'CP', 'CWP', 'AWP', 'T1', 'MT1',
             'Icl', 'Im', 'If', 'KB', 'BM', 'BMl', 'S'])
        df = df.round(2)
        dfi.export(df, 'hidrostatik.png')
        return df

pn = np.array([0, .5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9.5, 10]) / 10
wl = np.array([0, .3, 1, 2, 3, 4, 5, 6]) / 4
offset = np.loadtxt('s60_cb70.txt', dtype=float)

ship1 = ShipHydrostatics(100, 10, 5, offset, pn, wl)

ship1.save_show_table()