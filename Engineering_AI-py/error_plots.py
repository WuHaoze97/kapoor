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

# +
import matplotlib.pyplot as plt

# Define the data from your table
u_star = [5, 10, 12.5, 15, 17.5, 20]
with_TL = [0.03063, 0.03198, 0.04180, 0.06937, 0.222182, 0.23296]
without_TL = [117.7389, 45.65849, 59.42882, 19.7473, 48.75515, 29.50691]

# Create a line plot
plt.figure(figsize=(10, 6))
plt.plot(u_star, with_TL, label='with TL', marker='o')
plt.plot(u_star, without_TL, label='w/o TL', marker='x')

# Customize the plot
plt.title('Euler Bernoulli Beam: R at t=1 for different percentages of noise')
plt.xlabel('u*')
plt.ylabel('R')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()


# +
# import matplotlib.pyplot as plt

# # Define the data from your table
# u_star = [5, 10, 12.5, 15, 17.5, 20]
# with_TL = [0.03063, 0.03198, 0.04180, 0.06937, 0.222182, 0.23296]
# without_TL = [117.7389, 45.65849, 59.42882, 19.7473, 48.75515, 29.50691]

# # Create a line plot
# plt.figure(figsize=(10, 6))
# plt.plot(u_star, with_TL, label='with TL', marker='o')
# plt.plot(u_star, without_TL, label='w/o TL', marker='x')

# # Customize the plot with Times New Roman font
# plt.title('Euler Bernoulli Beam: R at t=1 for different percentages of noise', fontname='Times New Roman', fontsize=14)
# plt.xlabel('u*', fontname='Times New Roman', fontsize=12)
# plt.ylabel('R', fontname='Times New Roman', fontsize=12)
# plt.legend(fontname='Times New Roman', fontsize=12)
# plt.grid(True)

# # Customize the tick labels to be in Times New Roman
# plt.xticks(fontname='Times New Roman', fontsize=10)
# plt.yticks(fontname='Times New Roman', fontsize=10)

# # Show the plot
# plt.show()


# +
import matplotlib.pyplot as plt

# Define the data from your table
u_star = [5, 10, 12.5, 15, 17.5, 20]
with_TL = [0.03063, 0.03198, 0.04180, 0.06937, 0.222182, 0.23296]
without_TL = [117.7389, 45.65849, 59.42882, 19.7473, 48.75515, 29.50691]

# Create a line plot
plt.figure(figsize=(10, 6))

plt.rcParams['figure.figsize'] = [10, 4]

from matplotlib.font_manager import FontProperties
plt.rcParams['axes.unicode_minus'] = False

font_path = 'times-new-roman.ttf'
ticks_font = FontProperties(fname=font_path, size=14)
plt.xlabel("x", fontsize=20, fontproperties=ticks_font)
plt.ylabel("u", fontsize=20, fontproperties=ticks_font)

# Set Times New Roman font for ticks
plt.xticks(fontsize=20, fontproperties=ticks_font)
plt.yticks(fontsize=20, fontproperties=ticks_font)
plt.plot(u_star, with_TL, label='with TL', marker='o', lw=4)
plt.plot(u_star, without_TL, label='w/o TL', marker='x', lw=4)

#plt.plot(x_test, u_test, label="Ground Truth",lw=2)
# plt.scatter(x_test[::100], u_test[::100], label="Ground Truth", color='blue', lw=2)
# plt.plot(x_test, u_test_pred.detach(), label="Prediction",color='red', lw=2)
legend_font = FontProperties(family='Times New Roman', style='normal', size=30)

#
plt.legend(loc='upper right', frameon=False)
plt.grid(True)
#plt.savefig('diff_ic_dotted.pdf', dpi = 500, bbox_inches = "tight", format='pdf', backend='cairo')
plt.show()


# +
import matplotlib.pyplot as plt

# Define the data from your table
u_star = [5, 10, 12.5, 15, 17.5, 20]
with_TL = [0.03063, 0.03198, 0.04180, 0.06937, 0.222182, 0.23296]
without_TL = [117.7389, 45.65849, 59.42882, 19.7473, 48.75515, 29.50691]

# Create a line plot
plt.figure(figsize=(10, 6))

from matplotlib.font_manager import FontProperties
plt.rcParams['axes.unicode_minus'] = False

font_path = 'times-new-roman.ttf'
ticks_font = FontProperties(fname=font_path, size=14)
plt.xlabel("Noise percentage", fontsize=20, fontproperties=ticks_font)
plt.ylabel(r'$\mathcal{R}$', fontsize=20, fontproperties=ticks_font)


# Set Times New Roman font for ticks
plt.xticks(fontsize=20, fontproperties=ticks_font)
plt.yticks(fontsize=20, fontproperties=ticks_font)

# Increase the size of dots and markers
#plt.plot(u_star, with_TL, label='with TL', marker='o', markersize=10, lw=4)
plt.plot(u_star, without_TL, label='w/o TL',color = 'k', marker='x', markersize=15, lw=4)

legend_font = FontProperties(family='Times New Roman', style='normal', size=20)

# Customize legend
plt.legend(loc='upper right', frameon=False, prop=legend_font)

#plt.grid(True)
plt.savefig('Error_plot_wo_tl.pdf', dpi=500, bbox_inches="tight", format='pdf',  backend='cairo')
plt.show()


# +
import matplotlib.pyplot as plt

# Define the data from your table
u_star = [5, 10, 12.5, 15, 17.5, 20]
with_TL = [0.03063, 0.03198, 0.04180, 0.06937, 0.222182, 0.23296]
without_TL = [117.7389, 45.65849, 59.42882, 19.7473, 48.75515, 29.50691]

# Create a line plot
plt.figure(figsize=(10, 6))

from matplotlib.font_manager import FontProperties
plt.rcParams['axes.unicode_minus'] = False

font_path = 'times-new-roman.ttf'
ticks_font = FontProperties(fname=font_path, size=14)
plt.xlabel("Noise percentage", fontsize=20, fontproperties=ticks_font)
plt.ylabel(r'$\mathcal{R}$', fontsize=20, fontproperties=ticks_font)


# Set Times New Roman font for ticks
plt.xticks(fontsize=20, fontproperties=ticks_font)
plt.yticks(fontsize=20, fontproperties=ticks_font)

# Increase the size of dots and markers
plt.plot(u_star, with_TL, label='with TL', marker='o', color = 'r', markersize=10, lw=4)
#plt.plot(u_star, without_TL, label='w/o TL', marker='x', markersize=15, lw=4)

legend_font = FontProperties(family='Times New Roman', style='normal', size=20)

# Customize legend
plt.legend(loc='upper right', frameon=False, prop=legend_font)

# Set the y-axis range from 0 to 0.5
plt.ylim(0, 0.3)

#plt.grid(True)
plt.savefig('Error_plot_with_tl.pdf', dpi=500, bbox_inches="tight", format='pdf',  backend='cairo')
plt.show()


# +
import matplotlib.pyplot as plt

# Define the data from your tables
x = [5, 10, 12.5, 15, 17.5, 20]
dataset1 = [117.7389, 45.65849, 59.42882, 19.7473, 48.75515, 29.50691]
dataset2 = [0.03063, 0.03198, 0.04180, 0.06937, 0.222182, 0.23296]

# Create a figure and primary y-axis
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot the first dataset on the primary y-axis
ax1.plot(x, dataset1, color='b', label='Dataset 1', marker='o', markersize=10, lw=4)
ax1.set_xlabel('x', fontsize=20)
ax1.set_ylabel('Dataset 1', fontsize=20, color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax1.grid(True)

# Create a secondary y-axis and plot the second dataset
ax2 = ax1.twinx()
ax2.plot(x, dataset2, color='r', label='Dataset 2', marker='x', markersize=10, lw=4)
ax2.set_ylabel('Dataset 2', fontsize=20, color='r')
ax2.tick_params(axis='y', labelcolor='r')

# Add legend for both datasets
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
#ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.show()


# +
import matplotlib.pyplot as plt

# Define the data from your table
u_star = [5, 10, 12.5, 15, 17.5, 20]
with_TL = [0.03063, 0.03198, 0.04180, 0.06937, 0.222182, 0.23296]
without_TL = [117.7389, 45.65849, 59.42882, 19.7473, 48.75515, 29.50691]

# Create a figure and primary y-axis
fig, ax1 = plt.subplots(figsize=(10, 6))

from matplotlib.font_manager import FontProperties
plt.rcParams['axes.unicode_minus'] = False

font_path = 'times-new-roman.ttf'
ticks_font = FontProperties(fname=font_path, size=14)
ax1.set_xlabel("Noise percentage", fontsize=20, fontproperties=ticks_font)
ax1.set_ylabel(r'$\mathcal{R}$', fontsize=20, fontproperties=ticks_font)

# Set Times New Roman font for ticks
ax1.tick_params(axis='both', labelsize=20)
ax1.plot(u_star, with_TL, label='with TL', color='b', marker='o', markersize=10, lw=4)

# Create a secondary y-axis
ax2 = ax1.twinx()
ax2.set_ylabel(r'$\mathcal{R}$', fontsize=20, fontproperties=ticks_font)
ax2.plot(u_star, without_TL, label='w/o TL', color='r', marker='x', markersize=15, lw=4)

# Add legend for both datasets
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', prop=FontProperties(family='Times New Roman', style='normal', size=20))
ax2.set_ylim(0, 120)  # Set the range for the secondary y-axis

#plt.grid(True)
plt.show()

# -


