import numpy as np
import matplotlib.pyplot as plt


# # visualization 1.1
# zigong_poverty = [0.224109015, 0.174684428, 0.144866107, 0.115706496, 0.090707814, 0.081508537, 0.070997651, 0.062488989, 0.057344588]
# guangan_poverty = [0.24377122, 0.194286329, 0.162710306, 0.1293242, 0.103862965, 0.094635573, 0.082518422, 0.074101915, 0.067036182]
# year = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]

# # convert to percent
# zigong_percentage = [p * 100 for p in zigong_poverty]
# guangan_percentage = [p * 100 for p in guangan_poverty]

# # create a chart containing two subgraphs
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# # left: Zigong
# bars1 = ax1.bar(year, zigong_percentage, color='coral', edgecolor='black', width=0.6)
# for bar, pct in zip(bars1, zigong_percentage):
#     ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{pct:.2f}%', ha='center', va='bottom', fontsize=9)
# ax1.set_title('Zigong Poverty Rate Trend (2016-2024)', fontsize=14, fontweight='bold')
# ax1.set_xlabel('Year', fontsize=12)
# ax1.set_ylabel('Poverty Rate (%)', fontsize=12)
# ax1.set_xticks(year)
# ax1.set_ylim(0, 27)
# ax1.set_yticks([0, 5, 10, 15, 20, 25])
# ax1.tick_params(axis='both', labelsize=10)

# # right: Guangan
# bars2 = ax2.bar(year, guangan_percentage, color='steelblue', edgecolor='black', width=0.6)
# for bar, pct in zip(bars2, guangan_percentage):
#     ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{pct:.2f}%', ha='center', va='bottom', fontsize=9)
# ax2.set_title("Guang'an Poverty Rate Trend (2016-2024)", fontsize=14, fontweight='bold')
# ax2.set_xlabel('Year', fontsize=12)
# ax2.set_ylabel('Poverty Rate (%)', fontsize=12)
# ax2.set_xticks(year)
# ax2.set_ylim(0, 27)
# ax2.set_yticks([0, 5, 10, 15, 20, 25])
# ax2.tick_params(axis='both', labelsize=10)

# # adjust layout
# plt.tight_layout()

# # save image
# plt.savefig('visualization_outputs/poverty_comparison_bar_chart.png', dpi=300, bbox_inches='tight')

# # display chart
# plt.show()


# # visualization 1.2
# chengdu_tourist = [16840.25232, 18586.15622, 19998.03530, 21852.77342, 13050.43231, 13090.96614, 13425.30652, 16378.75457, 18288.56266]
# chengdu_highway = [5458.98045, 5509.69305, 5588.44578, 5632.29623, 2917.80507, 2654.53064, 2709.34089, 3432.17, 3476.16160]
# year = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]

# # create figure
# fig, ax = plt.subplots(figsize=(12, 7))

# # set bar width and positions
# bar_width = 0.35
# x = np.arange(len(year))

# # plot bars for both indicators
# bars1 = ax.bar(x - bar_width/2, chengdu_tourist, bar_width, color='coral', edgecolor='black', label='Number of Domestic Tourist Arrivals')
# bars2 = ax.bar(x + bar_width/2, chengdu_highway, bar_width, color='steelblue', edgecolor='black', label='Highway Passenger Traffic Volume')

# # add value labels on bars
# for bar in bars1:
#     ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100, f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8, rotation=0)
# for bar in bars2:
#     ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100, f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8, rotation=0)

# # set title and labels
# ax.set_title('Chengdu Tourist Arrivals and Highway Passenger Traffic (2016-2024)', fontsize=14, fontweight='bold')
# ax.set_xlabel('Year', fontsize=12)
# ax.set_ylabel('Value (10,000 person-times)', fontsize=12)
# ax.set_xticks(x)
# ax.set_xticklabels(year)
# ax.tick_params(axis='both', labelsize=10)

# # add legend
# ax.legend(loc='upper left', fontsize=10)

# # adjust layout
# plt.tight_layout()

# # save image
# plt.savefig('visualization_outputs/chengdu_tourist_highway_comparison.png', dpi=300, bbox_inches='tight')

# # display chart
# plt.show()


# visualization 1.3
meishan_unemployment = [3.158575, 3.147765, 3.108664, 3.105304, 3.070834, 3.07419, 3.079856, 3.070009, 3.042391]
year = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]

# calculate growth rate (from 2017 to 2024)
growth_rate = []
for i in range(1, len(meishan_unemployment)):
    rate = (meishan_unemployment[i] - meishan_unemployment[i-1]) / meishan_unemployment[i-1] * 100
    growth_rate.append(rate)
growth_year = year[1:]  # 2017-2024

# create a chart containing two subgraphs
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# left: Unemployment Rate Bar Chart
bars1 = ax1.bar(year, meishan_unemployment, color='coral', edgecolor='black', width=0.6, label='Unemployment Rate')
for bar, pct in zip(bars1, meishan_unemployment):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, f'{pct:.2f}%', ha='center', va='bottom', fontsize=9)
ax1.set_title('Meishan Unemployment Rate (2016-2024)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Year', fontsize=12)
ax1.set_ylabel('Unemployment Rate (%)', fontsize=12)
ax1.set_xticks(year)
ax1.set_ylim(0, 4.0)
ax1.tick_params(axis='both', labelsize=10)
ax1.legend(loc='upper right', fontsize=10)

# right: Growth Rate Line Chart
ax2.plot(growth_year, growth_rate, marker='o', color='steelblue', linewidth=2, markersize=8, label='Growth Rate')
for x, y in zip(growth_year, growth_rate):
    ax2.text(x, y + 0.05, f'{y:.2f}%', ha='center', va='bottom', fontsize=9)
ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1)  # add zero reference line
ax2.set_title('Meishan Unemployment Rate Growth Rate (2017-2024)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Year', fontsize=12)
ax2.set_ylabel('Growth Rate (%)', fontsize=12)
ax2.set_xticks(growth_year)
ax2.tick_params(axis='both', labelsize=10)
ax2.legend(loc='upper right', fontsize=10)

# adjust layout
plt.tight_layout()

# save image
plt.savefig('visualization_outputs/meishan_unemployment_rate.png', dpi=300, bbox_inches='tight')

# display chart
plt.show()
