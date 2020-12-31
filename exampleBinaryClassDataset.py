import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
 
# Create data frame. The last column is the needed color
df = pd.DataFrame(np.random.random((32,2)), columns=["Estimated Age","Average Whisker Nitrogen Isotope Value"])
 
# Add a column: the color depends of x and y values, but you can use whatever function.
# (df['Estimated Age'] < 0.5) &
value = ((df['Estimated Age'] < 0.4) &
         (df['Average Whisker Nitrogen Isotope Value'] > 0.3) )
df['color'] = np.where(value == True, "#FF0000", "#0000FF")
 
# plot
sns_plot = sns.regplot(data=df, x="Estimated Age", y="Average Whisker Nitrogen Isotope Value",
                       fit_reg=False, scatter_kws={'facecolors': df['color']})

plt.savefig('graph9.png')
