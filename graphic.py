import numpy as np
import matplotlib.pyplot as plt

def graphic_ve_cum(ve_cum:list, threshold:float, new_q:int):
    fig, graf = plt.subplots(figsize = (6, 4), constrained_layout=True)
    graf.plot(np.linspace(1, len(ve_cum), 900), ve_cum*100, color="b")
    graf.axhline(threshold*100, linestyle="--", color="k")
    graf.axvline(new_q, linestyle="--", color="k")
    graf.set(xlabel="Número de autovalores", ylabel="Variância Explicada Acumulada (%)")
    graf.set(xlim=(-5, len(ve_cum)), ylim=(min(ve_cum)*100, 102))
    graf.grid(True)
    plt.show()