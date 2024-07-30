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
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# # %matplotlib notebook
plt.rcParams['figure.figsize'] = [6, 6]

font_path = 'times-new-roman.ttf'
ticks_font = FontProperties(fname=font_path, size=14)
plt.xlabel("Epochs", fontsize=20, fontproperties=ticks_font)
plt.ylabel("Error", fontsize=20, fontproperties=ticks_font)

# Set Times New Roman font for ticks
plt.xticks(fontsize=20, fontproperties=ticks_font)
plt.yticks(fontsize=20, fontproperties=ticks_font)
#plt.plot(x_test, u_test, label="Ground Truth",lw=2)
# plt.scatter(x_test[::100], u_test[::100], label="Ground Truth", color='blue', lw=2)
# plt.plot(x_test, u_test_pred.detach(), label="Prediction",color='red', lw=2)
plt.plot(loaded_loss_1, label='$w_1$')
plt.plot(loaded_loss_2, label='$w_2$')
plt.plot(loaded_loss_3, label='$w_3$')
plt.plot(loaded_loss_4, label='$w_4$')
plt.plot(loaded_loss_5, label='$w_5$')
plt.plot(loaded_loss_6, label='$w_6$')

# plt.xlim([-5, 10000])
# plt.ylim([0.1, 1000])

# Increase the size of the legend
legend_font = FontProperties(family='Times New Roman', style='normal', size=50)

plt.legend(loc='upper right', frameon=False, prop={'size': 17})  # Adjust 'size' as needed

# plt.savefig('weights_epochs.pdf', dpi=500, bbox_inches="tight", format='pdf', backend='cairo')
# # plt.show()
# Your existing code...

# # Set x-axis and y-axis limits to define the zoomed-in portion
# plt.xlim([6862, 6872])

# # plt.legend(loc='lower right', frameon=False)
# # # Save the figure
plt.savefig('first_weights_epochs.pdf', dpi=500, bbox_inches="tight", format='pdf', backend='cairo')

# # Optional: If you want to reset the limits to their original values, uncomment the following lines
# plt.xlim(original_xlim)
# plt.ylim(original_ylim)

# Display the zoomed-in portion
plt.show()


# +
# # %matplotlib notebook
plt.rcParams['figure.figsize'] = [6, 6]

from matplotlib.font_manager import FontProperties
plt.rcParams['axes.unicode_minus'] = False

font_path = 'times-new-roman.ttf'
ticks_font = FontProperties(fname=font_path, size=14)
plt.xlabel("Epochs", fontsize=20, fontproperties=ticks_font)
plt.ylabel("Error", fontsize=20, fontproperties=ticks_font)

# Set Times New Roman font for ticks
plt.xticks(fontsize=20, fontproperties=ticks_font)
plt.yticks(fontsize=20, fontproperties=ticks_font)
#plt.plot(x_test, u_test, label="Ground Truth",lw=2)
# plt.scatter(x_test[::100], u_test[::100], label="Ground Truth", color='blue', lw=2)
# plt.plot(x_test, u_test_pred.detach(), label="Prediction",color='red', lw=2)
plt.plot(loaded_loss_1, label='$w_1$')
plt.plot(loaded_loss_2, label='$w_2$')
plt.plot(loaded_loss_3, label='$w_3$')
plt.plot(loaded_loss_4, label='$w_4$')
plt.plot(loaded_loss_5, label='$w_5$')
plt.plot(loaded_loss_6, label='$w_6$')

# plt.xlim([-5, 10000])
# plt.ylim([0.1, 1000])

legend_font = FontProperties(family='Times New Roman', style='normal', size=30)


plt.legend(loc='upper right', frameon=False, prop={'size': 17})
# plt.savefig('weights_epochs.pdf', dpi = 500, bbox_inches = "tight", format='pdf', backend='cairo')
# # plt.show()
# Your existing code...

# # Set x-axis and y-axis limits to define the zoomed-in portion
plt.xlim([-1, 175])
plt.ylim([-5, 130])

# # plt.legend(loc='lower right', frameon=False)
# # # Save the figure
plt.savefig('second_weights_epochs.pdf', dpi=500, bbox_inches="tight", format='pdf', backend='cairo')

# # Optional: If you want to reset the limits to their original values, uncomment the following lines
# plt.xlim(original_xlim)
# plt.ylim(original_ylim)

# Display the zoomed-in portion
plt.show()


# +
# # %matplotlib notebook
plt.rcParams['figure.figsize'] = [6, 6]

from matplotlib.font_manager import FontProperties
plt.rcParams['axes.unicode_minus'] = False

font_path = 'times-new-roman.ttf'
ticks_font = FontProperties(fname=font_path, size=14)
plt.xlabel("Epochs", fontsize=20, fontproperties=ticks_font)
plt.ylabel("Error", fontsize=20, fontproperties=ticks_font)

# Set Times New Roman font for ticks
plt.xticks(fontsize=20, fontproperties=ticks_font)
plt.yticks(fontsize=20, fontproperties=ticks_font)
#plt.plot(x_test, u_test, label="Ground Truth",lw=2)
# plt.scatter(x_test[::100], u_test[::100], label="Ground Truth", color='blue', lw=2)
# plt.plot(x_test, u_test_pred.detach(), label="Prediction",color='red', lw=2)
plt.plot(loaded_loss_1, label='$w_1$')
plt.plot(loaded_loss_2, label='$w_2$')
plt.plot(loaded_loss_3, label='$w_3$')
plt.plot(loaded_loss_4, label='$w_4$')
plt.plot(loaded_loss_5, label='$w_5$')
plt.plot(loaded_loss_6, label='$w_6$')

# plt.xlim([-5, 10000])
# plt.ylim([0.1, 1000])

legend_font = FontProperties(family='Times New Roman', style='normal', size=30)

# #
plt.legend(loc='upper right', frameon=False, prop={'size': 17})


# # Set x-axis and y-axis limits to define the zoomed-in portion
plt.xlim([200, 1600])
plt.ylim([-10, 100])

plt.savefig('Third_weights_epochs.pdf', dpi=500, bbox_inches="tight", format='pdf', backend='cairo')

plt.show()

# -

[-1, 175][-5, 30],{first subplot} [200, 1600][-10, 70]Second sublot


