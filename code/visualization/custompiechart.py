# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np


class CustomPieChart:
    def __init__(self, ratio, labels, colors=None, explode=None, startangle=90, shadow=True, legend_loc="upper center", legend_ncol=1):
        self.ratio = ratio
        self.labels = labels
        self.colors = colors if colors else ['firebrick', 'royalblue', 'green', 'gold']
        self.explode = explode if explode else (0,) * len(labels)
        self.startangle = startangle
        self.shadow = shadow
        self.legend_loc = legend_loc
        self.legend_ncol = legend_ncol
        plt.rcParams['figure.figsize'] = [5, 5]

    def func(self, pct, allvals):
        absolute = int(pct / 100. * np.sum(allvals))
        return "{:.0f}%\n({})".format(pct, absolute)

    def plot(self):
        plt.pie(self.ratio,
                labels=self.labels,
                autopct=lambda pct: self.func(pct, self.ratio),
                colors=self.colors,
                textprops={'fontsize': 15, 'weight': 'bold', 'color': 'white'},
                startangle=self.startangle,
                explode=self.explode,
                wedgeprops={'width': 1, "linewidth": 2, "edgecolor": 'black'},
                shadow=self.shadow,
                labeldistance=None)  # Removes the labels from next to the slices

        plt.legend(self.labels, loc=self.legend_loc, bbox_to_anchor=(0.5, -0.2), ncol=self.legend_ncol)
        plt.show()

# # Example usage:
# chart = CustomPieChart(
#     ratio=[423, 467],
#     labels=['Republican', 'Democrat'],
#     colors=['firebrick', 'royalblue'],
#     explode=(0.1, 0),  # Only explode the first slice ('Republican')
#     startangle=90
# )
# chart.plot()

# chart2 = CustomPieChart(
#     ratio=[21, 157],
#     labels=['Fail', 'Success'],
#     colors=['red', 'mediumseagreen'],
#     explode=(0.1, 0),  # Only explode the first slice ('Fail')
#     startangle=0
# )
# chart2.plot()