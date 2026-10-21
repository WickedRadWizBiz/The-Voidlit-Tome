import time, warnings, statistics
warnings.filterwarnings("ignore")
from voidlit.tome import TomeEngine, load_scenario
from voidlit.playtest import bot_command
import inspect
e = TomeEngine(seed=3)
e.start()
cmds = ["look around", "talk to elara: I trust you", "force the iris door", "search the room", "pry the iris with insight", "go on", "attack the drone", "sneak past", "search", "advance"]
ts=[]; tl=[]
import voidlit.limbic.npc_brain as nb
orig = nb.NPCBrain.perceive
acc=[0.0]
def timed(self, s):
    t0=time.perf_counter(); r=orig(self, s); acc[0]+=time.perf_counter()-t0; return r
nb.NPCBrain.perceive = timed
import voidlit.cortex.director as dr
origc = dr.Director.choose
accd=[0.0]
def timedc(self,*a,**k):
    t0=time.perf_counter(); r=origc(self,*a,**k); accd[0]+=time.perf_counter()-t0; return r
dr.Director.choose = timedc
for i in range(40):
    c = cmds[i % len(cmds)]
    acc[0]=0; accd[0]=0
    t0=time.perf_counter(); f=e.turn(c); dt=time.perf_counter()-t0
    ts.append(dt); tl.append((acc[0], accd[0]))
    if e.state.over: break
print("turns", len(ts), "median turn ms", round(statistics.median(ts)*1e3,1), "max", round(max(ts)*1e3,1))
print("median limbic ms", round(statistics.median([a for a,_ in tl])*1e3,1), "median director ms", round(statistics.median([b for _,b in tl])*1e3,2))
