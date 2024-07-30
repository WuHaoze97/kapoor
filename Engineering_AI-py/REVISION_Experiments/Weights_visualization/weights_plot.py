# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.3
#   kernelspec:
#     display_name: pytorch
#     language: python
#     name: pytorch
# ---

import numpy as np
import matplotlib.pyplot as plt
import torch

# +
loaded_weights = torch.load('final_weights_new.pth')

loaded_wt_1 = loaded_weights['wt_1'].numpy()
loaded_wt_2 = loaded_weights['wt_2'].numpy()
loaded_wt_3 = loaded_weights['wt_3'].numpy()
loaded_wt_4 = loaded_weights['wt_4'].numpy()
loaded_wt_5 = loaded_weights['wt_5'].numpy()
loaded_wt_6 = loaded_weights['wt_6'].numpy()
loaded_wt_7 = loaded_weights['wt_7'].numpy()
loaded_wt_8 = loaded_weights['wt_8'].numpy()
loaded_wt_9 = loaded_weights['wt_9'].numpy()
loaded_wt_10 = loaded_weights['wt_10'].numpy()
##loaded_wt_6 = loaded_weights['wt_6'].numpy()
# -

print(loaded_wt_1)

# +
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# # %matplotlib notebook
plt.rcParams['figure.figsize'] = [6, 6]

font_path = 'times-new-roman.ttf'
ticks_font = FontProperties(fname=font_path, size=14)
plt.xlabel("Epochs", fontsize=20, fontproperties=ticks_font)
plt.ylabel("Weights", fontsize=20, fontproperties=ticks_font)

# Set Times New Roman font for ticks
plt.xticks(fontsize=20, fontproperties=ticks_font)
plt.yticks(fontsize=20, fontproperties=ticks_font)
#plt.plot(x_test, u_test, label="Ground Truth",lw=2)
# plt.scatter(x_test[::100], u_test[::100], label="Ground Truth", color='blue', lw=2)
# plt.plot(x_test, u_test_pred.detach(), label="Prediction",color='red', lw=2)
plt.plot(loaded_wt_1, label='$w_1$')
plt.plot(loaded_wt_2, label='$w_2$')
plt.plot(loaded_wt_3, label='$w_3$')
plt.plot(loaded_wt_4, label='$w_4$')
plt.plot(loaded_wt_5, label='$w_5$')
plt.plot(loaded_wt_6, label='$w_6$')

# plt.xlim([-5, 10000])
# plt.ylim([0.1, 1000])

# Increase the size of the legend
legend_font = FontProperties(family='Times New Roman', style='normal', size=50)

plt.legend(loc='lower right', frameon=False, prop={'size': 17})  # Adjust 'size' as needed

plt.savefig('fig_A_W_epochs.pdf', dpi=500, bbox_inches="tight", format='pdf', backend='cairo')

plt.show()


# +
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# # %matplotlib notebook
plt.rcParams['figure.figsize'] = [6, 6]

font_path = 'times-new-roman.ttf'
ticks_font = FontProperties(fname=font_path, size=14)
plt.xlabel("Epochs", fontsize=20, fontproperties=ticks_font)
plt.ylabel("Weights", fontsize=20, fontproperties=ticks_font)

# Set Times New Roman font for ticks
plt.xticks(fontsize=20, fontproperties=ticks_font)
plt.yticks(fontsize=20, fontproperties=ticks_font)
#plt.plot(x_test, u_test, label="Ground Truth",lw=2)
# plt.scatter(x_test[::100], u_test[::100], label="Ground Truth", color='blue', lw=2)
# plt.plot(x_test, u_test_pred.detach(), label="Prediction",color='red', lw=2)
plt.plot(loaded_wt_1, label='$w_1$')
plt.plot(loaded_wt_2, label='$w_2$')
plt.plot(loaded_wt_3, label='$w_3$')
plt.plot(loaded_wt_4, label='$w_4$')
plt.plot(loaded_wt_5, label='$w_5$')
plt.plot(loaded_wt_6, label='$w_6$')

# plt.xlim([-5, 10000])
# plt.ylim([0.1, 1000])

# Increase the size of the legend
legend_font = FontProperties(family='Times New Roman', style='normal', size=50)

# # Set x-axis and y-axis limits to define the zoomed-in portion
plt.xlim([1400, 2600])
plt.ylim([0, 1])
plt.legend(loc='lower right', frameon=False, prop={'size': 17})  # Adjust 'size' as needed


plt.savefig('fig1_A_W_epochs.pdf', dpi=500, bbox_inches="tight", format='pdf', backend='cairo')

plt.show()


# +
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# # %matplotlib notebook
plt.rcParams['figure.figsize'] = [6, 6]

font_path = 'times-new-roman.ttf'
ticks_font = FontProperties(fname=font_path, size=14)
plt.xlabel("Epochs", fontsize=20, fontproperties=ticks_font)
plt.ylabel("Weights", fontsize=20, fontproperties=ticks_font)

# Set Times New Roman font for ticks
plt.xticks(fontsize=20, fontproperties=ticks_font)
plt.yticks(fontsize=20, fontproperties=ticks_font)
#plt.plot(x_test, u_test, label="Ground Truth",lw=2)
# plt.scatter(x_test[::100], u_test[::100], label="Ground Truth", color='blue', lw=2)
# plt.plot(x_test, u_test_pred.detach(), label="Prediction",color='red', lw=2)
plt.plot(loaded_wt_1, label='$w_1$')
plt.plot(loaded_wt_2, label='$w_2$')
plt.plot(loaded_wt_3, label='$w_3$')
plt.plot(loaded_wt_4, label='$w_4$')
plt.plot(loaded_wt_5, label='$w_5$')
plt.plot(loaded_wt_6, label='$w_6$')

# plt.xlim([-5, 10000])
# plt.ylim([0.1, 1000])

# Increase the size of the legend
legend_font = FontProperties(family='Times New Roman', style='normal', size=50)

# # Set x-axis and y-axis limits to define the zoomed-in portion
plt.xlim([3500, 6000])
plt.ylim([0.95, 1])
plt.legend(loc='lower right', frameon=False, prop={'size': 17})  # Adjust 'size' as needed


plt.savefig('fig2_A_W_epochs.pdf', dpi=500, bbox_inches="tight", format='pdf', backend='cairo')

plt.show()

# -


