import glob, numpy as np, pandas as pd
from scipy.signal import welch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
FS=1000; LSB=0.125  # GAIN_SIXTEEN: 0.125 mV per ADC count
csv=sorted(glob.glob("geophone_2026*.csv"))[-1]
d=pd.read_csv(csv); t=d["time_s"].to_numpy(float); mv=d["amplitude"].to_numpy(float)*1000.0
x=mv-mv.mean()
nlev=int(round((mv.max()-mv.min())/LSB))+1
fig,ax=plt.subplots(2,2,figsize=(13,8))
# (a) full waveform
ax[0,0].plot(t,mv,lw=0.4); ax[0,0].set_title(f"(a) Full waveform — {csv}\nstd={mv.std():.3f} mV, spans only ~{nlev} ADC levels")
ax[0,0].set_xlabel("time (s)"); ax[0,0].set_ylabel("mV")
# (b) zoom showing the quantization staircase
m=(t>=2)&(t<=2.4)
ax[0,1].plot(t[m],mv[m],"-o",ms=3,lw=0.8)
for lv in np.arange(np.floor(mv[m].min()/LSB),np.ceil(mv[m].max()/LSB)+1)*LSB:
    ax[0,1].axhline(lv,color="gray",lw=0.3,alpha=0.5)
ax[0,1].set_title("(b) 0.4 s zoom — gray lines = 0.125 mV ADC steps\n(signal hops between a handful of discrete levels)")
ax[0,1].set_xlabel("time (s)"); ax[0,1].set_ylabel("mV")
# (c) amplitude histogram in mV on the LSB grid -> shows the discrete levels (the 'comb')
edges=np.arange(mv.min()-LSB/2, mv.max()+LSB, LSB)
ax[1,0].hist(mv,bins=edges,color="#d62728")
ax[1,0].set_title("(c) Amplitude histogram in mV, 1 bin = 1 ADC level (0.125 mV)\n-> the 'comb' is just the discrete ADC codes")
ax[1,0].set_xlabel("mV"); ax[1,0].set_ylabel("count")
# (d) PSD
f,P=welch(x,fs=FS,nperseg=8192)
ax[1,1].loglog(f,P,lw=0.9); ax[1,1].set_xlim(1,FS/2)
for ln in (50,100,150): ax[1,1].axvline(ln,color="gray",ls=":",lw=0.8)
ax[1,1].axvspan(6.5,8.5,color="orange",alpha=0.2)
ax[1,1].set_title("(d) PSD — 50/100/150 Hz mains (dotted) + ~7.4 Hz floor drum (orange)")
ax[1,1].set_xlabel("Hz"); ax[1,1].set_ylabel("mV$^2$/Hz")
plt.tight_layout(); plt.savefig("real_recording_proper.png",dpi=130)
print(f"{csv}: {len(mv)} samples, std={mv.std():.3f} mV, range {mv.min():.3f}..{mv.max():.3f} mV = ~{nlev} ADC levels (LSB {LSB} mV)")
print(f"distinct amplitude values: {len(np.unique(np.round(mv/LSB)))}")
print("plot -> real_recording_proper.png")
