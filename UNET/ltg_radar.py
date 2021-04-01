from matplotlib.pyplot import get_cmap
from matplotlib.pyplot import cm
from matplotlib.colors import ListedColormap
from matplotlib.colors import BoundaryNorm

rgb_colors = []
rgb_colors.append((49,0,241))     #1 indigo
rgb_colors.append((66,255,35))    #2 ltgreen
rgb_colors.append((50,202,25))    #3 green
rgb_colors.append((33,146,14))    #4 dkgreen
rgb_colors.append((249,252,45))   #5 yellow
rgb_colors.append((224,191,35))   #6 yellow-orange
rgb_colors.append((247,148,32))   #7 orange
rgb_colors.append((243,0,25))     #8 ltred

colors = []
for atup in rgb_colors:
    colors.append('#%02x%02x%02x'%atup)

cm.register_cmap(cmap=ListedColormap(colors,'ltg'))

cmap = get_cmap('radar')
cmap.set_under('#e3e3e3')  #(227,227,227)

bounds = [0.1,2.5,5,7.5,10,20,30,40,50]
norm = BoundaryNorm(bounds,cmap.N)
