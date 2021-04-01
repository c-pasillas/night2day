from matplotlib.pyplot import get_cmap
from matplotlib.pyplot import cm
from matplotlib.colors import ListedColormap
from matplotlib.colors import BoundaryNorm

rgb_colors = []
rgb_colors.append((253,253,253))     #200
rgb_colors.append((151,72,196))     #205
rgb_colors.append((242,0,251))     #210
rgb_colors.append((180,0,15))     #215
rgb_colors.append((203,0,19))     #220
rgb_colors.append((243,0,25))     #225
rgb_colors.append((247,148,32))     #230
rgb_colors.append((224,191,35))     #235
rgb_colors.append((249,252,45))     #240
rgb_colors.append((33,146,14))     #245
rgb_colors.append((50,202,25))     #250
rgb_colors.append((66,255,35))     #255
rgb_colors.append((49,0,241))     #260
rgb_colors.append((63,155,242))   #265
rgb_colors.append((76,235,230))   #270
rgb_colors.append((227,227,227))  #275

colors = []
for atup in rgb_colors:
    colors.append('#%02x%02x%02x'%atup)

cm.register_cmap(cmap=ListedColormap(colors,'ir'))

#cmap = get_cmap('radar')
#cmap.set_over(colors[-1])
#
## testing this for ref3d plots
#cmap.set_under(colors[0])
#
##bounds = [-30,-25,-20,-15,-10,-5,0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80]
#bounds = [0,5,10,15,20,25,30,35,40,45,50,55,60]
#
#ticklabels = [str(a) for a in bounds]
#
#norm = BoundaryNorm(bounds,cmap.N)
