#!/usr/bin/env python3
"""Моменты появления каждого нового слова вшитых субтитров (30 fps). -> s2/work/<part>.capon.json"""
import subprocess, numpy as np, cv2, json, sys
from pathlib import Path
def onsets(path, band=(0.63, 0.83)):
    W=360
    w,h=map(int,subprocess.run(["ffprobe","-v","error","-select_streams","v:0","-show_entries","stream=width,height","-of","csv=p=0",path],capture_output=True,text=True).stdout.split(","))
    H=int(h*W/w)
    p=subprocess.Popen(["ffmpeg","-v","error","-i",path,"-vf",f"fps=30,scale={W}:{H}","-f","rawvideo","-pix_fmt","bgr24","-"],stdout=subprocess.PIPE)
    prev=None; ev=[]; i=0; area=[]
    while True:
        b=p.stdout.read(W*H*3)
        if len(b)<W*H*3: break
        f=np.frombuffer(b,np.uint8).reshape(H,W,3)[int(H*band[0]):int(H*band[1])]
        m=(cv2.cvtColor(f,cv2.COLOR_BGR2HSV)[...,2]>200)
        if prev is not None:
            d=np.logical_xor(m,prev).mean()
            if d>0.02 and m.mean()>0.004 and (not ev or i/30-ev[-1]>0.2): ev.append(round(i/30,3))
        prev=m; i+=1
    return ev, i/30
out=Path("work"); out.mkdir(exist_ok=True)
args=[a for a in sys.argv[1:] if not a.startswith("--band=")]
band=next((tuple(map(float,a[7:].split(","))) for a in sys.argv[1:] if a.startswith("--band=")),(0.63,0.83))
for p in args:
    ev,dur=onsets(p,band); n=Path(p).stem
    (out/f"{n}.capon.json").write_text(json.dumps({"dur":dur,"onsets":ev}))
    print(n, len(ev), "onsets", round(dur,1),"s")
