# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
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

# +
loaded_loss = torch.load('final_error_new.pth')

loaded_loss_1 = loaded_loss['error_1'].detach().numpy()
loaded_loss_2 = loaded_loss['error_2'].detach().numpy()
loaded_loss_3 = loaded_loss['error_3'].detach().numpy()
loaded_loss_4 = loaded_loss['error_4'].detach().numpy()
loaded_loss_5 = loaded_loss['error_5'].detach().numpy()
loaded_loss_6 = loaded_loss['error_6'].detach().numpy()
loaded_loss_7 = loaded_loss['error_7'].detach().numpy()
loaded_loss_8 = loaded_loss['error_8'].detach().numpy()
loaded_loss_9 = loaded_loss['error_9'].detach().numpy()
loaded_loss_10 = loaded_loss['error_10'].detach().numpy()
##loaded_loss_6 = loaded_loss['error_6'].detach().numpy()

# +
# # %matplotlib notebook
plt.plot(loaded_wt_1, label='wt1')
plt.plot(loaded_wt_2, label='wt2')
plt.plot(loaded_wt_3, label='wt3')
plt.plot(loaded_wt_4, label='wt4')
plt.plot(loaded_wt_5, label='wt5')
plt.plot(loaded_wt_6, label='wt6')

# Increase ticks and labels size
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Increase axis label size
plt.xlabel('epochs', fontsize=14)
plt.ylabel('weights', fontsize=14)
plt.legend(frameon=False)

plt.show()

# +
import matplotlib.pyplot as plt

# Increase the number of labels, ticks, and legend position
plt.plot(loaded_wt_1, label='wt1')
plt.plot(loaded_wt_2, label='wt2')
plt.plot(loaded_wt_3, label='wt3')
plt.plot(loaded_wt_4, label='wt4')
plt.plot(loaded_wt_5, label='wt5')
plt.plot(loaded_wt_6, label='wt6')

# Increase ticks and labels size
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Increase axis label size
plt.xlabel('epochs', fontsize=16, fontname='times new roman')
plt.ylabel('weights', fontsize=16, fontname='times new roman')

# Increase legend size and move it to lower right
plt.legend(fontsize=12, loc='lower right', frameon=False)

plt.show()


# +
plt.rcParams['figure.figsize'] = [10, 4]

from matplotlib.font_manager import FontProperties
plt.rcParams['axes.unicode_minus'] = False

font_path = 'times-new-roman.ttf'
ticks_font = FontProperties(fname=font_path, size=14)
plt.xlabel("epochs", fontsize=20, fontproperties=ticks_font)
plt.ylabel("weights", fontsize=20, fontproperties=ticks_font)

# Set Times New Roman font for ticks
plt.xticks(fontsize=20, fontproperties=ticks_font)
plt.yticks(fontsize=20, fontproperties=ticks_font)
#plt.plot(x_test, u_test, label="Ground Truth",lw=2)
# plt.scatter(x_test[::100], u_test[::100], label="Ground Truth", color='blue', lw=2)
# plt.plot(x_test, u_test_pred.detach(), label="Prediction",color='red', lw=2)
plt.plot(loaded_wt_1, label='wt1')
plt.plot(loaded_wt_2, label='wt2')
plt.plot(loaded_wt_3, label='wt3')
plt.plot(loaded_wt_4, label='wt4')
plt.plot(loaded_wt_5, label='wt5')
plt.plot(loaded_wt_6, label='wt6')


legend_font = FontProperties(family='Times New Roman', style='normal', size=30)

#
# plt.legend(loc='lower left', frameon=False)
# plt.savefig('diff_ic_dotted_case2.pdf', dpi = 500, bbox_inches = "tight", format='pdf', backend='cairo')
# plt.show()


# +
import matplotlib.pyplot as plt

# Assuming loaded_loss_1, loaded_loss_2, ..., loaded_loss_10 are defined

# Set custom colors for each line
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']

plt.figure()

plt.loglog(loaded_loss_1, label='wt1', color=colors[0])
plt.loglog(loaded_loss_2, label='wt2', color=colors[1])
plt.loglog(loaded_loss_3, label='wt3', color=colors[2])
plt.loglog(loaded_loss_4, label='wt4', color=colors[3])
plt.loglog(loaded_loss_5, label='wt5', color=colors[4])
plt.loglog(loaded_loss_6, label='wt6', color=colors[5])
# plt.loglog(loaded_loss_7, label='wt7', color=colors[6])
# plt.loglog(loaded_loss_8, label='wt8', color=colors[7])
# plt.loglog(loaded_loss_9, label='wt9', color=colors[8])
# plt.loglog(loaded_loss_10, label='wt10', color=colors[9])

plt.legend()
plt.show()


# +
plt.rcParams['figure.figsize'] = [10, 4]

from matplotlib.font_manager import FontProperties
plt.rcParams['axes.unicode_minus'] = False

font_path = 'times-new-roman.ttf'
ticks_font = FontProperties(fname=font_path, size=14)
plt.xlabel("epochs", fontsize=20, fontproperties=ticks_font)
plt.ylabel("weights", fontsize=20, fontproperties=ticks_font)

# Set Times New Roman font for ticks
plt.xticks(fontsize=20, fontproperties=ticks_font)
plt.yticks(fontsize=20, fontproperties=ticks_font)
#plt.plot(x_test, u_test, label="Ground Truth",lw=2)
# plt.scatter(x_test[::100], u_test[::100], label="Ground Truth", color='blue', lw=2)
# plt.plot(x_test, u_test_pred.detach(), label="Prediction",color='red', lw=2)
plt.loglog(loaded_loss_1, label='wt1')
plt.loglog(loaded_loss_2, label='wt2')
plt.loglog(loaded_loss_3, label='wt3')
plt.loglog(loaded_loss_4, label='wt4')
plt.loglog(loaded_loss_5, label='wt5')
plt.loglog(loaded_loss_6, label='wt6')
#

legend_font = FontProperties(family='Times New Roman', style='normal', size=30)


# +
plt.rcParams['figure.figsize'] = [10, 4]

# Comment out the line below to use the default font for the minus sign
# plt.rcParams['axes.unicode_minus'] = False
font_path = 'times-new-roman.ttf'
ticks_font = FontProperties(fname=font_path, size=14)

# Use a system-installed Times New Roman font or another font that includes the minus sign
ticks_font = {'fontname': 'Times New Roman', 'size': 14}
plt.xlabel("epochs", fontsize=20, **ticks_font)
plt.ylabel("weights", fontsize=20, **ticks_font)

# Set Times New Roman font for ticks
plt.xticks(fontsize=20, **ticks_font)
plt.yticks(fontsize=20, **ticks_font)

# Your plot commands
plt.loglog(loaded_loss_1, label='wt1')
plt.loglog(loaded_loss_2, label='wt2')
plt.loglog(loaded_loss_3, label='wt3')
plt.loglog(loaded_loss_4, label='wt4')
plt.loglog(loaded_loss_5, label='wt5')
plt.loglog(loaded_loss_6, label='wt6')

# Set Times New Roman font for legend
legend_font = {'family': 'Times New Roman', 'style': 'normal', 'size': 10}
plt.legend(fontsize=5, prop=legend_font)

plt.show()

# -


