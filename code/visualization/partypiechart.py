
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


class PartyDistributionPieChart:
    def __init__(self, df):
        self.df = df

    def generate_pie_chart(self):
        # Count the occurrences of each party
        party_counts = self.df['party'].fillna('NaN (George Washington)').value_counts()

        # Create labels with party names and counts
        recipe = [f"{party} : {count}" for party, count in party_counts.items()]

        # Data for the pie chart
        data = party_counts.values

        # Create a new figure and axis with equal aspect ratio
        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(aspect="equal"))

        # Plot the pie chart
        wedges, texts = ax.pie(data, 
                           wedgeprops={'width':0.7, "linewidth" :2, "edgecolor":'black'}, 
                           startangle=30,
                           colors=['royalblue','firebrick','orangered','salmon','darkorange','gold','cadetblue'],
                           shadow=True,
                           textprops={'fontsize':14, 'color':'black'},
                           explode=(0.00,0.00,0,0,0,0,0)) 
        
        # Define properties for the bounding box of annotations
        bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
        kw = dict(arrowprops=dict(arrowstyle="-"),
                  bbox=bbox_props, zorder=0, va="center")

        # Annotate each pie wedge
        for i, p in enumerate(wedges):
            ang = (p.theta2 - p.theta1)/6. + p.theta1
            y = np.sin(np.deg2rad(ang))
            x = np.cos(np.deg2rad(ang))
            horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
            connectionstyle = "angle,angleA=0,angleB={}".format(ang)
            kw["arrowprops"].update({"connectionstyle": connectionstyle})
            ax.annotate(recipe[i], xy=(x, y), xytext=(1.5*np.sign(x), 1.4*y),
                        horizontalalignment=horizontalalignment, **kw,
                        fontweight='bold')

        # Set the title for the pie chart
        ax.set_title("Political Party : # of Speeches", fontweight='bold')

        # Display the pie chart
        plt.show()

# Example usage
# Assuming 'df' is your DataFrame
# pie_chart_generator = PartyDistributionPieChart(df)
# pie_chart_generator.generate_pie_chart()