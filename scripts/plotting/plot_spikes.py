import glob, numpy as np, pandas as pd
from scipy.signal import welch, find_peaks, spectrogram
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
FS=1000
csv=sorted(glob.glob("geophone_2026*.csv"))[-1]; print("file:", csv)
mv=pd.read_csv(csv)["amplitude"].to_numpy(float)*1000.0
x=mv-mv.mean(); t=np.arange(len(x))/FS
ax=np.abs(x); thr=max(0.6, 5*np.median(ax)/0.6745)
pk,_=find_peaks(ax, height=thr, distance=15)
print(f"spikes detected: {len(pk)}  (rate {len(pk)/(len(x)/FS):.1f}/s)  threshold {thr:.2f} mV")
# ---- Figure 1: overview ----
fig,ax1=plt.subplots(2,2,figsize=(14,8))
ax1[0,0].plot(t,x,lw=0.4); ax1[0,0].plot(t[pk],x[pk],"r.",ms=4)
ax1[0,0].set_title(f"(a) Full waveform — {len(pk)} spikes, baseline std {x.std():.3f} mV")
ax1[0,0].set_xlabel("s"); ax1[0,0].set_ylabel("mV")
# averaged spike (sign-aligned)
W=50; segs=[]
for p in pk:
    if p-W>=0 and p+W<len(x):
        s=x[p-W:p+W]*np.sign(x[p]); segs.append(s)
segs=np.array(segs); tt=np.arange(-W,W)
if len(segs):
    m=segs.mean(0); sd=segs.std(0)
    ax1[0,1].plot(tt,m,lw=1.8,c="firebrick"); ax1[0,1].fill_between(tt,m-sd,m+sd,alpha=.2,color="firebrick")
    for s in segs[:40]: ax1[0,1].plot(tt,s,lw=0.3,c="gray",alpha=.4)
ax1[0,1].axvline(0,ls=":",c="k",lw=.6)
ax1[0,1].set_title("(b) Averaged spike shape (sign-aligned)\nsingle-sample glitch = electrical; ringing = mechanical")
ax1[0,1].set_xlabel("ms from peak"); ax1[0,1].set_ylabel("mV")
# ISI
if len(pk)>2:
    isi=np.diff(pk)/FS*1000
    ax1[1,0].hist(isi,bins=40,color="#4C78A8")
    ax1[1,0].set_title(f"(c) Inter-spike interval (median {np.median(isi):.0f} ms)\nregular=periodic/electrical; broad/random=mechanical")
    ax1[1,0].set_xlabel("ms between spikes"); ax1[1,0].set_ylabel("count")
# amplitude hist
ax1[1,1].hist(x[pk],bins=40,color="#F58518"); ax1[1,1].set_title("(d) Spike amplitude distribution")
ax1[1,1].set_xlabel("mV"); ax1[1,1].set_ylabel("count")
plt.tight_layout(); plt.savefig("spikes_overview.png",dpi=130); plt.close()
# ---- Figure 2: spectrum + spectrogram ----
fig,ax2=plt.subplots(2,1,figsize=(12,8))
f,P=welch(x,fs=FS,nperseg=8192); ax2[0].loglog(f,P,lw=0.9)
for ln in (50,100,150): ax2[0].axvline(ln,ls=":",c="gray",lw=.7)
ax2[0].set_xlim(1,FS/2); ax2[0].set_title("PSD (dotted=50/100/150 Hz mains)"); ax2[0].set_xlabel("Hz"); ax2[0].set_ylabel("mV^2/Hz"); ax2[0].grid(alpha=.3,which="both")
ff,ttt,Sxx=spectrogram(x,fs=FS,nperseg=512,noverlap=384)
ax2[1].pcolormesh(ttt,ff,10*np.log10(Sxx+1e-12),shading="auto",cmap="magma")
ax2[1].set_ylim(0,200); ax2[1].set_title("Spectrogram (vertical streaks=broadband impulses; horizontal lines=tones)")
ax2[1].set_xlabel("s"); ax2[1].set_ylabel("Hz")
plt.tight_layout(); plt.savefig("spikes_spectrum.png",dpi=130); plt.close()
# spike width (samples above half-peak) — electrical glitches are ~1 sample
widths=[]
for p in pk:
    h=ax[p]/2; l=p
    while l>0 and ax[l]>h: l-=1
    r=p
    while r<len(ax)-1 and ax[r]>h: r+=1
    widths.append(r-l)
print(f"median spike width: {np.median(widths):.1f} samples ({np.median(widths):.0f} ms)  [1 sample = electrical glitch]")
print("saved spikes_overview.png, spikes_spectrum.png")
