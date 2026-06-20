# Student FULL name: Huynh Thien Luan Dang

# Acknowledgement Statement by students: <br>
# 	I acknowledge that I have only used GenAI (e.g., ChatGPT) in drafting and proofreading this assignment, which is permitted in the assignment instructions. <br>
#

# Provide three interesting facts (no more than 4 lines or 70 words) about the diamond industry in Australia, with the link to the source where you got the info.

# 1. Diamonds were first recorded in Australia in 1851 in New South Wales, with early mining focused on alluvial deposits in areas like Copeton and Bingara. <br>
# 2. The Argyle mine in Western Australia, discovered in 1979, became the world's largest by volume and supplied over 90% of the world's pink diamonds. <br>
# 3. Argyle was famous for rare pink and red diamonds and closed in 2020, after which Australia's diamond production dropped sharply. <br>
#
# Source for fact 1: https://australian.museum/learn/minerals/gemstones/diamond/
#
# Source for fact 2: https://www.ga.gov.au/education/minerals-energy/australian-mineral-facts/diamond
#
# Source for fact 3: https://www.brilliyond.com.au/education/diamond-mining/australia

# Conduct Exploratory Data Analysis (EDA) as needed.

# Import essential libraries for all sections
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.stats import ttest_1samp, ttest_ind
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.special import inv_boxcox

# Load the CSV file into a DataFrame
diamond = pd.read_csv('data.csv')

# Display the first 5 rows of the dataset
diamond.head()

# Display summary info (columns, data types, non-null counts)
diamond.info()

# Count the total number of missing values in each column and show them
missing_count = diamond.isnull().sum()
missing_count

# Display all unique values in the 'carat' column
diamond['carat'].unique()

# Display all unique values in the 'cut' column
diamond['cut'].unique()

# Display all unique values in the 'color' column
diamond['color'].unique()

# Display all unique values in the 'clarity' column
diamond['clarity'].unique()

# Display all unique values in the 'depth' column
diamond['depth'].unique()

# # Display all unique values in the 'array' column
diamond['table'].unique()

# Display all unique values in the 'x' column
diamond['x'].unique()

# Display all unique values in the 'y' column
diamond['y'].unique()

# Display all unique values in the 'z' column
diamond['z'].unique()

# Display all unique values in the 'price' column
diamond['price'].unique()

# Reload the CSV, treating empty strings, spaces, NaN, null, NULL, None, na, and 0 as missing values
diamond = pd.read_csv('data.csv', na_values=['', ' ', 'NaN', 'null', 'NULL', 'None', 'na', 0])

# Display summary info (columns, data types, non-null counts)
diamond.info()

# Convert the data type of numeric columns from object to float64 for analysis
diamond['carat'] = pd.to_numeric(diamond['carat'], errors='coerce')
diamond['depth'] = pd.to_numeric(diamond['depth'], errors='coerce')
diamond['table'] = pd.to_numeric(diamond['table'], errors='coerce')
diamond['x'] = pd.to_numeric(diamond['x'], errors='coerce')
diamond['z'] = pd.to_numeric(diamond['z'], errors='coerce')
diamond['price'] = pd.to_numeric(diamond['price'], errors='coerce')

# Display the count of missing values per column
diamond.isnull().sum()

# Count missing values per row:
missing_values_per_row = diamond.isnull().sum(axis=1)

# Filter and display only rows that contain at least one missing value
rows_with_nulls = diamond[missing_values_per_row > 0]
print("Rows with at least one missing value:\n", rows_with_nulls)

# Drop rows where 'cut', 'color', or 'clarity' have missing values
diamond.dropna(subset=['cut', 'color', 'clarity'], inplace=True)

# Double check by showing rows with at least one null value again:
missing_values_per_row = diamond.isnull().sum(axis=1)
rows_with_nulls = diamond[missing_values_per_row > 0]
print("Rows with at least one missing value:\n", rows_with_nulls)

# Fill missing values in 'carat' with the column's median
diamond['carat'] = diamond['carat'].fillna(diamond['carat'].median())

# Fill missing values in 'depth' with the column's median
diamond['depth'] = diamond['depth'].fillna(diamond['depth'].median())

# Fill missing values in 'table' with the column's median
diamond['table'] = diamond['table'].fillna(diamond['table'].median())

# Fill missing values in 'x' with the column's median
diamond['x'] = diamond['x'].fillna(diamond['x'].median())

# Fill missing values in 'y' with the column's median
diamond['y'] = diamond['y'].fillna(diamond['y'].median())

# Fill missing values in 'z' with the column's median
diamond['z'] = diamond['z'].fillna(diamond['z'].median())

# Fill missing values in 'price' with the column's median
diamond['price'] = diamond['price'].fillna(diamond['price'].median())

# Count and display remaining missing values per column to verify
missing_values_per_column = diamond.isnull().sum()
print("Missing values per column:\n", missing_values_per_column)

# Returns the total number of duplicate rows
duplicate_count = diamond.duplicated().sum()
print(f"Number of duplicate rows: {duplicate_count}")

# Filters the dataframe to show only the duplicate records
duplicate_rows = diamond[diamond.duplicated(keep=False)] 
print(duplicate_rows)

# Removes duplicates
diamond.drop_duplicates(inplace=True)

# Check duplicates again
duplicate_rows = diamond[diamond.duplicated(keep=False)] 
print(duplicate_rows)

# Save the DataFrame with missing values treated to a new CSV file
diamond.to_csv('data_cleaned_missingvalues_duplicates.csv', index=False)

# Create a figure with specified size for displaying multiple boxplots
plt.figure(figsize=(15, 15))

# Carat boxplot
plt.subplot(4, 3, 1)
plt.boxplot(diamond['carat'],patch_artist=True)
plt.title('Boxplot of Carat')
plt.ylabel('Carat Value')

# Depth boxplot
plt.subplot(4, 3, 2)
plt.boxplot(diamond['depth'], patch_artist=True)
plt.title('Boxplot of Depth')
plt.ylabel('Depth Value')

# Table boxplot
plt.subplot(4, 3, 3)
plt.boxplot(diamond['table'], patch_artist=True)
plt.title('Boxplot of Table')
plt.ylabel('Table Value')

# Width boxplot
plt.subplot(4, 3, 4)
plt.boxplot(diamond['x'], patch_artist=True)
plt.title('Boxplot of Diamond Width')
plt.ylabel('Diamond Width Value')

# Length boxplot
plt.subplot(4, 3, 5)
plt.boxplot(diamond['y'], patch_artist=True)
plt.title('Boxplot of Diamond Length')
plt.ylabel('Diamond Length Value')

# Height boxplot
plt.subplot(4, 3, 6)
plt.boxplot(diamond['z'], patch_artist=True)
plt.title('Boxplot of Diamond Height')
plt.ylabel('Diamond Height Value')

# Price boxplot
plt.subplot(4, 3, 7)
plt.boxplot(diamond['price'], patch_artist=True)
plt.title('Boxplot of Diamond Price')
plt.ylabel('Diamond Price Value (USD)')

# Adjust spacing between subplots to prevent overlap
plt.tight_layout()

# Display all the boxplots
plt.show()

# Display summary statistics for all numeric columns
diamond.describe()

# Calculate the first quantile, the third quantile, and the interquartile range of carat
Q1_carat = diamond['carat'].quantile(0.25)
Q3_carat = diamond['carat'].quantile(0.75)
IQR_carat = Q3_carat - Q1_carat

# Calculate the upper and lower bound
lower_carat = Q1_carat - 1.5 * IQR_carat
upper_carat= Q3_carat + 1.5 * IQR_carat

# Calculate the median of carat
median_carat = diamond['carat'].median()

# Replace outliers (values below the lower bound and above the upper bound) with the median 
diamond.loc[(diamond['carat'] < lower_carat) | (diamond['carat'] > upper_carat), 'carat'] = median_carat

# Calculate the first quantile, the third quantile, and the interquartile range of depth
Q1_depth = diamond['depth'].quantile(0.25)
Q3_depth = diamond['depth'].quantile(0.75)
IQR_depth = Q3_depth - Q1_depth

# Calculate the upper and lower bound
lower_depth = Q1_depth - 1.5 * IQR_depth
upper_depth= Q3_depth + 1.5 * IQR_depth

# Calculate the median of depth
median_depth = diamond['depth'].median()

# Replace outliers (values below the lower bound and above the upper bound) with the median
diamond.loc[(diamond['depth'] < lower_depth) | (diamond['depth'] > upper_depth), 'depth'] = median_depth

# Calculate the first quantile, the third quantile, and the interquartile range of table
Q1_table = diamond['table'].quantile(0.25)
Q3_table = diamond['table'].quantile(0.75)
IQR_table = Q3_table - Q1_table

# Calculate the upper and lower bound
lower_table = Q1_table - 1.5 * IQR_table
upper_table= Q3_table + 1.5 * IQR_table

# Calculate the median of table
median_table = diamond['table'].median()

# Replace outliers (values below the lower bound and above the upper bound) with the median
diamond.loc[(diamond['table'] < lower_table) | (diamond['table'] > upper_table), 'table'] = median_table

# Calculate the first quantile, the third quantile, and the interquartile range of width
Q1_x = diamond['x'].quantile(0.25)
Q3_x = diamond['x'].quantile(0.75)
IQR_x = Q3_x - Q1_x

# Calculate the upper and lower bound
lower_x = Q1_x - 1.5 * IQR_x
upper_x= Q3_x + 1.5 * IQR_x

# Calculate the median of width
median_x = diamond['x'].median()

# Replace outliers (values below the lower bound and above the upper bound) with the median
diamond.loc[(diamond['x'] < lower_x) | (diamond['x'] > upper_x), 'x'] = median_x

# Calculate the first quantile, the third quantile, and the interquartile range of length
Q1_y = diamond['y'].quantile(0.25)
Q3_y = diamond['y'].quantile(0.75)
IQR_y = Q3_y - Q1_y

# Calculate the upper and lower bound
lower_y = Q1_y - 1.5 * IQR_y
upper_y= Q3_y + 1.5 * IQR_y

# Calculate the median of length
median_y = diamond['y'].median()

# Replace outliers (values below the lower bound and above the upper bound) with the median
diamond.loc[(diamond['y'] < lower_y) | (diamond['y'] > upper_y), 'y'] = median_y

# Calculate the first quantile, the third quantile, and the interquartile range of height
Q1_z = diamond['z'].quantile(0.25)
Q3_z = diamond['z'].quantile(0.75)
IQR_z = Q3_z - Q1_z

# Calculate the upper and lower bound
lower_z = Q1_z - 1.5 * IQR_z
upper_z= Q3_z + 1.5 * IQR_z

# Calculate the median of length
median_z = diamond['z'].median()

diamond.loc[(diamond['z'] < lower_z) | (diamond['z'] > upper_z), 'z'] = median_z

Q1_price = diamond['price'].quantile(0.25)
Q3_price = diamond['price'].quantile(0.75)
IQR_price = Q3_price - Q1_price

lower_price = Q1_price - 1.5 * IQR_price
upper_price = Q3_price + 1.5 * IQR_price

median_price = diamond['price'].median()

diamond.loc[(diamond['price'] < lower_price) | (diamond['price'] > upper_price), 'price'] = median_price

# Create a figure with specified size for displaying multiple boxplots
plt.figure(figsize=(15, 15))

# Carat boxplot
plt.subplot(4, 3, 1)
plt.boxplot(diamond['carat'],patch_artist=True)
plt.title('Boxplot of Carat')
plt.ylabel('Carat Value')

# Depth boxplot
plt.subplot(4, 3, 2)
plt.boxplot(diamond['depth'], patch_artist=True)
plt.title('Boxplot of Depth')
plt.ylabel('Depth Value')

# Table boxplot
plt.subplot(4, 3, 3)
plt.boxplot(diamond['table'], patch_artist=True)
plt.title('Boxplot of Table')
plt.ylabel('Table Value')

# Width boxplot
plt.subplot(4, 3, 4)
plt.boxplot(diamond['x'], patch_artist=True)
plt.title('Boxplot of Diamond Width')
plt.ylabel('Diamond Width Value')

# Length boxplot
plt.subplot(4, 3, 5)
plt.boxplot(diamond['y'], patch_artist=True)
plt.title('Boxplot of Diamond Length')
plt.ylabel('Diamond Length Value')

# Height boxplot
plt.subplot(4, 3, 6)
plt.boxplot(diamond['z'], patch_artist=True)
plt.title('Boxplot of Diamond Height')
plt.ylabel('Diamond Height Value')

# Price boxplot
plt.subplot(4, 3, 7)
plt.boxplot(diamond['price'], patch_artist=True)
plt.title('Boxplot of Diamond Price')
plt.ylabel('Diamond Price Value (USD)')

# Adjust spacing between subplots to prevent overlap
plt.tight_layout()

# Display all the boxplots
plt.show()

# Display summary statistics for all numeric columns
diamond.describe()

# Create a figure with specified size for displaying multiple histograms
plt.figure(figsize=(15, 15))

# Carat histogram
plt.subplot(4,3,1)
plt.hist(diamond['carat'], bins=30, color='red', edgecolor='black')
plt.title('Histogram of Carat')

# Depth histogram
plt.subplot(4,3,2)
plt.hist(diamond['depth'], bins=30, color='green', edgecolor='black')
plt.title('Histogram of Depth')

# Table histogram
plt.subplot(4,3,3)
plt.hist(diamond['table'], bins=30, color='darkblue', edgecolor='black')
plt.title('Histogram of Table')

# Width histogram
plt.subplot(4,3,4)
plt.hist(diamond['x'], bins=30, color='yellow', edgecolor='black')
plt.title('Histogram of Diamond Width')

# Length histogram
plt.subplot(4,3,5)
plt.hist(diamond['y'], bins=30, color='purple', edgecolor='black')
plt.title('Histogram of Diamond Length')

# Height histogram
plt.subplot(4,3,6)
plt.hist(diamond['z'], bins=30, color='orange', edgecolor='black')
plt.title('Histogram of Diamond Height')

# Price histogram
plt.subplot(4,3,7)
plt.hist(diamond['price'], bins=30, color='teal', edgecolor='black')
plt.title('Histogram of Price')

# Adjust spacing between subplots to prevent overlap
plt.tight_layout()

# Display all the histograms
plt.show()

# Calculate and show skewness of numeric columns
skewness = diamond.skew(numeric_only=True)

print("Skewness of each numerical variable:")
print(skewness)

# Fix skewness for columns with skewness values under -0.5 and above 0.5
# carat, price

# carat

# Apply log transformation to fix skewness
diamond['carat_log'] = np.log(diamond['carat'])

# Display histogram after log transformation
plt.figure(figsize=(10, 6))
plt.hist(diamond['carat_log'], bins=30, color='green', edgecolor='black')
plt.title('Histogram of Log Transformed Carat Data')
plt.xlabel('Log Values')
plt.ylabel('Frequency')
plt.show()

# Calculate and show skewness after log transformation
skewness_after_log_carat = diamond['carat_log'].skew()
skewness_after_log_carat

# Apply square root transformation to fix skewness
diamond['carat_sqrt'] = np.sqrt(diamond['carat'])

# Display histogram after square root transformation
plt.figure(figsize=(10, 6))
plt.hist(diamond['carat_sqrt'], bins=30, color='red', edgecolor='black')
plt.title('Histogram of Square Root Transformed Carat Data')
plt.xlabel('Square Root Values')
plt.ylabel('Frequency')
plt.show()

# Calculate and show skewness after square root transformation
skewness_after_sqrt_carat = diamond['carat_sqrt'].skew()
skewness_after_sqrt_carat

# Apply Box-Cox transformation to fix skewness
diamond['carat_boxcox'], _ = stats.boxcox(diamond['carat'])

# Display histogram after Box-Cox transformation
plt.figure(figsize=(10, 6))
plt.hist(diamond['carat_boxcox'], bins=30, color='purple', edgecolor='black')
plt.title('Histogram of Box-Cox Transformed Carat Data')
plt.xlabel('Box-Cox Values')
plt.ylabel('Frequency')
plt.show()

# Calculate and show skewness after box-cox transformation
skewness_after_boxcox_carat = diamond['carat_boxcox'].skew()
skewness_after_boxcox_carat

# Box cox transformation gives the best results for carat

# price

# Apply log transformation to fix skewness
diamond['price_log'] = np.log(diamond['price'])

# Display histogram after log transformation
plt.figure(figsize=(10, 6))
plt.hist(diamond['price_log'], bins=30, color='green', edgecolor='black')
plt.title('Histogram of Log Transformed Price Data')
plt.xlabel('Log Values')
plt.ylabel('Frequency')
plt.show()

# Calculate and show skewness after log transformation
skewness_after_log_price = diamond['price_log'].skew()
skewness_after_log_price

# Apply square root transformation to fix skewness
diamond['price_sqrt'] = np.sqrt(diamond['price'])

# Display histogram after square root transformation
plt.figure(figsize=(10, 6))
plt.hist(diamond['price_sqrt'], bins=30, color='red', edgecolor='black')
plt.title('Histogram of Square Root Transformed Price Data')
plt.xlabel('Square Root Values')
plt.ylabel('Frequency')
plt.show()

# Calculate skewness after square root transformation
skewness_after_sqrt_price = diamond['price_sqrt'].skew()
skewness_after_sqrt_price

# Apply Box-Cox transformation to fix skewness, save its lambda for later to inverse transform so that we can report the evaluation metrics of model training in the correct scale
diamond['price_boxcox'], lambda_price = stats.boxcox(diamond['price'])
print(f"Lambda for price: {lambda_price}")

# Display histogram after Box-Cox transformation
plt.figure(figsize=(10, 6))
plt.hist(diamond['price_boxcox'], bins=30, color='purple', edgecolor='black')
plt.title('Histogram of Box-Cox Transformed Price Data')
plt.xlabel('Box-Cox Values')
plt.ylabel('Frequency')
plt.show()

# Calculate and show skewness after box-cox transformation
skewness_after_boxcox_price = diamond['price_boxcox'].skew()
skewness_after_boxcox_price

# Boxcox transformation gives the best results for price

# Save the DataFrame with missing values, outliers, and skewness treated to a new CSV file
diamond.to_csv('data_cleaned_full.csv', index=False)

# Develop and visualise four descriptive questions that would drive insightful values for Aurora Gems.

# Visualisation 1: How does the carat weight of a diamond affect its price?
# Understanding this relationship helps Aurora Gems price their diamonds accurately based on weight and helps predict the value of new inventory.

# Load the cleaned diamond dataset into a DataFrame
diamond_visualisation = pd.read_csv('data_cleaned_missingvalues_duplicates.csv')

# Plot carat against price as a scatter, with transparency to show point density
diamond_visualisation.plot.scatter(x='carat', y='price', c='DarkBlue', alpha=0.2)

# Label the x-axis
plt.xlabel('Carat Value')

# Label the y-axis
plt.ylabel('Diamond Price (USD)')

# Add a title to the plot
plt.title('Relationship Between Carat Value and Diamond Price')

# Render and display the plot
plt.show()

# Insights and Recommendations
#
# The scatter plot shows a clear positive relationship between carat weight and price, with heavier diamonds commanding higher prices on average. However, the relationship is weak as a standalone predictor: at almost every carat level the price spread is very wide, with one-carat stones ranging from a few hundred dollars to near the top of the scale. This indicates that other attributes such as cut, clarity, and colour are doing substantial work in setting value. The vertical banding at 1.0, 1.5, and 2.0 carats reflects the market tendency to cut stones to land just above psychologically important weight thresholds, where price jumps disproportionately. Prices also appear to flatten near a 20,000 USD ceiling, and inventory above 3 carats is sparse but priced consistently at a premium.
#
# Management should therefore avoid pricing on carat weight alone and instead use a multivariate model incorporating cut, clarity, and colour to explain the variance left unaddressed here and reduce mispricing. The wide price dispersion at common 1.0 and 2.0 carat weights should be reviewed to confirm it stems from genuine quality differences rather than inconsistent pricing, and the threshold effects present a sourcing opportunity, since stones just below a round weight can be acquired more cheaply while appearing similar to buyers. Finally, the 20,000 USD ceiling should be validated before use in any valuation work, as it may reflect data capping rather than true market prices.

# Visualisation 2: What is the average price of diamonds across different cut types?
# This helps Aurora Gems understand the price that customers pay for better cut qualities. It can guide purchasing decisions like if one type of cuts yield significantly higher prices, Aurora Gem might want to stock more of it.

# Load the cleaned diamond dataset into a DataFrame
diamond_visualisation = pd.read_csv('data_cleaned_missingvalues_duplicates.csv')

# Calculate the average price for each cut type, sorted from highest to lowest
avg_price = diamond_visualisation.groupby('cut')['price'].mean().sort_values(ascending=False)

# Plot the average price per cut type as a bar chart
plt.bar(x=avg_price.index, height=avg_price.values)

# Label the x-axis
plt.xlabel('Cut Type')

# Label the y-axis
plt.ylabel('Average Price')

# Add a title to the plot
plt.title('Average Price per Cut Type')

# Render and display the plot
plt.show()

# Insights and Recommendations
#
# The chart reveals a counterintuitive pattern: the lowest-quality cut, Fair, commands the highest average price at around 5,300 USD, while the top-quality Ideal cut sits well below it at roughly 3,600 USD. The bars descend from Fair through Premium, Good, and Very Good in a way that runs opposite to the expected quality-price ordering, where better cuts should command higher prices. The most likely explanation is a confound with carat: Fair and Premium stones in this inventory tend to be larger, and the size premium outweighs the cut discount. The "Very Fair" category stands out at the bottom, sitting far below all others at under 500 USD, which suggests it may be a sparsely populated category or one dominated by very small stones.
#
# For management, the takeaway is that cut quality cannot be read as a standalone price driver from this chart, because the apparent trend is almost certainly being driven by differences in carat weight across the cut grades. Aurora Gems should control for size before drawing purchasing conclusions, ideally by comparing average price per carat within each cut grade to isolate the true effect of cut on value. The "Very Fair" category should also be examined more closely before this analysis informs any stocking decisions, as its extreme low average could reflect an unusual mix of stones rather than a genuine pricing signal.

# Visualisation 3: Which diamond colors are most abundant in Aurora Gems' inventory?
# Knowing this distribution helps with marketing and sales strategies. If Aurora Gems has a lot of stock of a specific colour, they can run targeted campaigns.

# Load the cleaned diamond dataset into a DataFrame
diamond_visualisation = pd.read_csv('data_cleaned_missingvalues_duplicates.csv')

# Count how many diamonds fall into each colour grade, sorted from most to least common
color_counts = diamond_visualisation['color'].value_counts()

# Plot the count for each colour grade as a purple bar chart
plt.bar(color_counts.index, color_counts.values, color = 'purple')

# Add a title to the plot
plt.title('Abundance of Diamonds by Color in Inventory')

# Label the x-axis
plt.xlabel('Diamond Color')

# Label the y-axis
plt.ylabel('Number of Diamonds (Count)')

# Render and display the plot
plt.show()

# Insights and recommendations
#
# The inventory is heavily concentrated in the middle of the colour scale. G is the single most abundant grade at over 11,000 stones, followed closely by E, F, and H, each holding between roughly 8,000 and 10,000 diamonds. These four grades dominate the inventory and together account for the large majority of stock. From D onward the counts fall steadily, with I, J, and Z holding progressively smaller quantities, and the S and N grades are effectively absent with negligible counts. The overall shape shows Aurora Gems is concentrated in the near-colourless to faint range rather than at the premium colourless end or the heavily tinted end.
#
# For marketing and sales, this distribution points to clear opportunities. The high-volume G, E, F, and H grades are the natural focus for targeted campaigns and promotions, since strong stock levels can support sustained demand without risk of running out, and bundling or featured-collection strategies would work well here. The thinly stocked grades such as J and Z are better suited to scarcity-based or premium positioning rather than volume campaigns. The near-empty S and N categories should be checked before any decisions are made, as their negligible counts may reflect genuinely rare stock or simply non-standard codes, and Aurora Gems will want to confirm which before factoring them into strategy.

# The fourth question: How does the physical width of a diamond relate to its carat weight?
# This helps Aurora Gems and customers understand if a heavier diamond actually looks wider from the top (visual size).
diamond_visualisation = pd.read_csv('data_cleaned_missingvalues_duplicates.csv')
diamond_visualisation.plot.scatter(x='x', y='carat', c='Green', alpha=0.2)
plt.xlabel('Diamond Width')
plt.ylabel('Carat Value')
plt.title('Relationship Between Diamond Width and Carat Value')
plt.show()

# Insights and recommendations
#
# The chart shows a strong positive relationship between physical width and carat weight: as a diamond's width increases, its carat weight rises as well. The relationship is curved rather than straight, which makes sense physically, since weight scales with volume (roughly the cube of width) while width is a single linear dimension. The dense band of points forms a clear, tight curve from the bottom-left upward, indicating that width is a reliable predictor of carat for the bulk of the inventory. There is a scatter of points sitting above the main band, particularly in the 4 to 9 width range, where stones have a higher carat than their width would suggest, likely reflecting differences in cut depth and proportions that add weight without adding width.
#
# For Aurora Gems and its customers, the key message is reassuring: a wider diamond does generally mean a heavier one, so the visual size a customer sees from the top broadly tracks the carat weight they pay for. However, the points above the main curve are worth flagging, because they represent stones that carry extra weight in their depth rather than their face-up spread. These diamonds look smaller from the top than their carat weight implies, which is a value consideration customers should understand, and one Aurora Gems could use to guide buyers toward stones that maximise visible size per carat. The very small number of stones with unusually large widths at the top of the chart should also be checked to confirm they are genuine rather than measurement errors.

# Develop one one-sample t-test and two two-sample t-tests (‘two-sided’ and ‘upper/lower-tailed’ test). Please make sure they are meaningful.

# **One-sample t-test**
#
# **Research question**: Is the average price of diamonds different from 3,300 USD?
#
# **H0:** The mean price of diamonds is equal to 3,300 USD.
#
# **H1:** The mean price of diamonds is not equal to 3,300 USD.
#
# We'll use a significance level of 0.05 (5%).

# Load the cleaned dataset with outliers handled
diamond_t_test = pd.read_csv('data_cleaned_full.csv')

# perform one-sample t-test
t_stat_1, p_value_1 = ttest_1samp(diamond_t_test['price'], 3300)

# Display t-statistic value
print("t-statistic:", t_stat_1)

# Display p-value
print("p-value:", p_value_1)

# Significance level
alpha = 0.05

# Compare p-value with alpha to determine whether to reject the null hypothesis
if p_value_1 < alpha:
    print("Reject the null hypothesis. The mean number of diamond price is significantly different from 3,300 USD.")
else:
    print("Fail to reject the null hypothesis. The data does not provide sufficient evidence that the mean number of diamond price is significantly different from 3,300 USD.")

# The one-sample t-test examined whether the average diamond price differs from 3,300 USD. The test returned a t-statistic of -3.38 and a p-value of approximately 0.0007, which is far below the 0.05 significance threshold. We therefore reject the null hypothesis and conclude that the mean diamond price is significantly different from 3,300 USD. The negative t-statistic indicates the direction of this difference: the true average price sits below 3,300 USD, not above it. In practical terms, the result is statistically robust, since a p-value this small means there is well under a 0.1% chance of observing a difference this large if the true mean really were 3,300 USD.
#
# For Aurora Gems, this confirms that 3,300 USD is not an accurate representation of the average value of their inventory, and that the genuine average lies meaningfully lower. Management should use the actual sample mean as the reference point for pricing benchmarks, valuation, and financial planning rather than a rounded assumption. It is worth noting that statistical significance does not by itself indicate the size of the gap, so Aurora Gems should also look at the actual mean price and the magnitude of the difference to judge whether it is large enough to matter commercially. Because price is strongly influenced by carat, cut, and colour, this overall average is best treated as a high-level benchmark rather than a guide to pricing any individual stone.

# **Two-sample t-test (two-sided)**
#
# **Research question**: Is there a significant difference in the mean depth between diamonds with color grade D and diamonds with color grade J?
#
# **H0:** The mean depth of color-D diamonds is equal to the mean depth of color-J diamonds.
#
# **H1:** The mean depth of color-D diamonds is not equal to the mean depth of color-J diamonds.
#
# We'll use a significance level of  0.05 (5%).

# Load the cleaned dataset with outliers handled
diamond_t_test = pd.read_csv('data_cleaned_full.csv')

# Filter depth values for diamonds with color 'D'
d_depth = diamond_t_test[diamond_t_test['color'] == 'D']['depth']

# Filter depth values for diamonds with color 'J'
j_depth = diamond_t_test[diamond_t_test['color'] == 'J']['depth']

# Perform two-sample t-test
t_stat_2a, p_value_2a = ttest_ind(d_depth, j_depth)

# Display t-statistic value
print("t-statistic:", t_stat_2a)

# Display p-value
print("p-value:", p_value_2a)

# Significance level
alpha = 0.05

# Compare p-value with alpha to determine whether to reject the null hypothesis
if p_value_2a < alpha:
    print("Reject the null hypothesis. The mean number of color-D diamonds' depth differs from color-J diamonds' depth.")
else:
    print("Fail to reject the null hypothesis. The data does not provide sufficient evidence that the mean number of color-D diamonds' depth differs from color-J diamonds' depth")

# The two-sample t-test compared the mean depth of colour-D diamonds against colour-J diamonds. The test produced a t-statistic of -5.91 and a p-value of approximately 0.0000000035 (3.5 × 10⁻⁹), which is far below the 0.05 significance level. We therefore reject the null hypothesis and conclude that the mean depth of colour-D diamonds is significantly different from that of colour-J diamonds. The negative t-statistic indicates the direction of the difference: colour-D diamonds have a lower average depth than colour-J diamonds. The extremely small p-value means this result is highly unlikely to have arisen by chance, making it a statistically strong finding.
#
# For Aurora Gems, this suggests that depth proportions are not uniform across colour grades, and that the two ends of this colour range differ systematically in their cut profile. This is useful context when interpreting earlier price patterns, since depth contributes to carat weight without adding to face-up size, so a consistent depth difference between grades could partly explain why colour and price did not move in the expected direction. Management should be cautious, however, since statistical significance does not measure how large the depth gap is in practical terms. The actual difference in average depth may be small even though it is highly significant, given the large sample size, so Aurora Gems should review the real mean depths of each grade before drawing any operational conclusions about cut quality or value.

# **Two-sample t-test (one-sided)**
#
# **Research question**: Is the mean price of Very Good-cut diamonds significantly greater than the mean price of Premium-cut diamonds?
#
# **H0:** The mean price of Very Good-cut diamonds is less than or equal to the mean price of Premium-cut diamonds.
#
# **H1:** The mean price of Very Good-cut diamonds is greater than the mean price of Premium-cut diamonds.
#
# We'll use a significance level of 0.05 (5%).

# Load the cleaned dataset with outliers handled
diamond_t_test = pd.read_csv('data_cleaned_full.csv')

# Filter price values for diamonds with 'Very Good' cut
verygood_price = diamond_t_test[diamond_t_test['cut'] == 'Very Good']['price']

# Filter price values for diamonds with 'Premium' cut
premium_price = diamond_t_test[diamond_t_test['cut'] == 'Premium']['price']

# Perform two-sample t-test (upper-tailed)
t_stat_2b, p_value_2b = ttest_ind(verygood_price, premium_price, alternative='greater')

# Display t-statistic value
print("t-statistic:", t_stat_2b)

# Display p-value
print("p-value:", p_value_2b)

# Significance level
alpha = 0.05

# Compare p-value with alpha to determine whether to reject the null hypothesis
if p_value_2b < alpha:
    print("Reject the null hypothesis. The mean price of Very Good-cut diamonds is significantly greater than Premium-cut diamonds.")
else:
    print("Fail to reject the null hypothesis. The data does not provide sufficient evidence that the mean price of Very Good-cut diamonds is greater than Premium-cut diamonds.")

# This one-sided two-sample t-test tested whether the mean price of Very Good-cut diamonds is greater than that of Premium-cut diamonds. The test returned a t-statistic of -7.32 and a p-value of approximately 0.9999, which is far above the 0.05 significance level. We therefore fail to reject the null hypothesis and conclude that there is no evidence that Very Good-cut diamonds are priced higher than Premium-cut diamonds. The result is in fact strongly the opposite of the alternative hypothesis: the large negative t-statistic shows that Very Good diamonds are on average priced lower than Premium diamonds, which is exactly why the upper-tailed p-value sits so close to 1. The test was looking for evidence in one direction, and the data points firmly in the other.
#
# For Aurora Gems, this confirms that a "Very Good" cut grade does not command a price premium over "Premium" cut in this inventory, and the average actually runs the other way. As with the earlier cut analysis, this is most likely driven by a carat confound rather than the cut grade itself, since Premium stones in this dataset tend to be larger and size dominates price. Management should therefore not assume cut grade alone justifies a higher price, and if they specifically want to test whether Premium is priced above Very Good, they could run the one-sided test in the reverse direction to confirm it statistically. As always, comparing price per carat within each cut grade would isolate the true effect of cut and give a cleaner basis for pricing decisions.

# Develop machine learning models to predict a meaningful outcome variable of your choice, which would contextually make sense.

# Load the cleaned dataset with outliers and skewness handled
diamond_cleaned = pd.read_csv('data_cleaned_full.csv')

# Display a summary of the dataset
diamond_cleaned.info()

# Choose columns to use for this question
diamond_prediction = diamond_cleaned[[
    'carat_boxcox',
    'cut',
    'color',
    'clarity',
    'depth',
    'table',
    'x',
    'y',
    'z',
    'price_boxcox']]

# Identify categorical columns
categorical_cols = [
    'cut',
    'color',
    'clarity']

# Perform one-hot encoding; dtype=int keeps continuous columns as floats
diamond_encoded = pd.get_dummies(diamond_prediction, columns=categorical_cols,
                                 drop_first=True, dtype=int)

# Display the first few rows of the encoded dataframe
diamond_encoded.head()

# Shuffle the rows to remove any ordering in the dataset (e.g. sorted by carat or price),
# which can cause the Durbin-Watson test to falsely detect autocorrelation in cross-sectional data.
# frac=1 returns all rows in random order; random_state=42 makes the shuffle reproducible;
# reset_index(drop=True) renumbers the now-jumbled index cleanly.
diamond_encoded = diamond_encoded.sample(frac=1, random_state=42).reset_index(drop=True)

# Compute the correlation matrix
correlation_matrix = diamond_encoded.corr()

# Set up the matplotlib figure
plt.figure(figsize=(30, 10))

# Generate a heat map
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)

plt.title('Correlation Matrix')
plt.show()

# Define the independent variables (features) and the dependent variable (target)
X = diamond_encoded.drop('price_boxcox', axis=1)
y = diamond_encoded['price_boxcox']

# Add a constant term to the model (for the intercept)
X = sm.add_constant(X)

# Calculate VIF for each feature
vif_data = pd.DataFrame()
vif_data['Feature'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# Display the VIF data
print(vif_data)

'''NOTE: Calculating Variance Inflation Factor (VIF) for categorical independent variables in multiple regression analysis is not standard practice, and it generally doesn't make much sense.'''

# VIF < 10: Generally acceptable; no significant multicollinearity.
#
# VIF > 10: Indicates multicollinearity issues.
#
# The results show no evidence of multicollinearity. All independent variables have VIF scores between 1.00 and 2.29, falling well below the standard of 5 for concern.
#
# note: The constant term is inherently collinear with any other variables in the model because it does not vary; it is the same (a constant value of 1) across all observations.

# Drop highly correlated variables to avoid multicollinearity
diamond_encoded = diamond_encoded.drop(columns=[
    'x',
    'y',
    'z'
])

# Compute the correlation matrix
correlation_matrix = diamond_encoded.corr()

# Set up the matplotlib figure
plt.figure(figsize=(30, 10))

# Generate a heat map
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)

plt.title('Correlation Matrix')
plt.show()

# Define the independent variables (features) and the dependent variable (target)
X = diamond_encoded.drop('price_boxcox', axis=1)
y = diamond_encoded['price_boxcox']

# Add a constant term to the model (for the intercept)
X = sm.add_constant(X)

# Calculate VIF for each feature
vif_data = pd.DataFrame()
vif_data['Feature'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# Display the VIF data
print(vif_data)

'''NOTE: Calculating Variance Inflation Factor (VIF) for categorical independent variables in multiple regression analysis is not standard practice, and it generally doesn't make much sense.'''

# VIF < 10: Generally acceptable; no significant multicollinearity.
#
# VIF > 10: Indicates multicollinearity issues.
#
# The results show no evidence of multicollinearity. All independent variables have VIF scores between 1.00 and 2.29, falling well below the standard of 5 for concern.
#
# note: The constant term is inherently collinear with any other variables in the model because it does not vary; it is the same (a constant value of 1) across all observations.

# After the redundant dimension variables (x, y, z) were removed in favour of carat as the single size measure, the remaining elevated VIFs are attributable to one-hot encoding of categorical features and were retained as legitimate predictors.

# Fit the regression model
model = sm.OLS(y, X).fit()

# Display the summary of the regression model
model_summary = model.summary()
model_summary

# Overall model fit
# The OLS regression model was statistically significant overall, with an F-statistic of 9,845 and a p-value of approximately zero, confirming that the predictors jointly explain diamond price far better than a model with no predictors. The model returned an R-squared of 0.809, meaning it accounts for roughly 81% of the variation in the Box-Cox transformed price, with an identical adjusted R-squared of 0.809 confirming the predictors all contribute meaningfully. This represents a strong fit for a linear model, indicating that the diamond's measured attributes explain the large majority of the variation in its price.
#
# Carat as the dominant predictor
# Carat was by far the most influential predictor, with a coefficient of 1.60 and an extremely large t-statistic of 460, significant at the 1% level. This confirms that diamond size is the overwhelming determinant of price, holding all other attributes constant. Table was statistically significant but carried a very small coefficient (0.004), and depth was not statistically significant (p = 0.238), indicating that neither dimension has a meaningful practical effect on price once carat is accounted for.
#
# Colour effects
# The colour variables were highly significant and showed the expected direction in earlier analysis. The coefficients become increasingly negative as colour grade declines from E through to J (color_E at -0.046 down to color_J at -0.310), correctly indicating that lower colour grades are associated with lower prices once size is controlled for. As before, the N and S categories were not statistically significant and carried very large standard errors, consistent with these being sparsely populated or non-standard grades that the model cannot estimate reliably.
#
# Cut and Clarity effects
# The cut variables were mostly significant and positive, with Ideal (0.101), Premium (0.072), Very Good (0.058), and Good (0.032) all reaching significance and ordering sensibly, so that better cuts are associated with higher prices once size is held constant. Only the sparse Very Fair category was insignificant. The clarity variables were also all highly significant and positive, with the highest-clarity grades commanding the largest premiums (clarity_IF at 0.478 and clarity_VVS1 at 0.430), which aligns with the expectation that superior clarity increases value.

# Checking regression assumptions

# Normality of residues

# Get the residuals
residuals = model.resid


# Histogram of residuals
plt.figure(figsize=(8, 6))
sns.histplot(residuals, kde=True, bins=30)
plt.title('Histogram of Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()

# The histogram of residuals assesses whether the model's errors are approximately normally distributed. The residuals are sharply centred on zero and form a tall, broadly symmetric peak, indicating that most predictions fall very close to the actual values. Some departures from normality are visible, including a small secondary cluster around -1.5 and a slightly longer left tail. Overall the normality assumption is reasonably satisfied, and given the large sample of 55,685 observations, these minor departures do not undermine the reliability of the model's estimates, as OLS remains robust to moderate non-normality at this scale.

# QQplot

stats.probplot(residuals, dist="norm", plot=plt)
plt.show()

# The Q-Q plot compares the model's residuals against a theoretical normal distribution, where perfect normality would place all points along the red diagonal. The residuals follow the line closely through the centre, confirming they are approximately normal for the bulk of the distribution, but both tails curve away from the line, indicating a heavy-tailed distribution where the most extreme residuals are larger than a normal distribution would predict. A mild stepped pattern is also visible, reflecting the discrete group structure of the categorical predictors. Overall the normality assumption is reasonably satisfied through the centre, and given the large sample of 55,685 observations, these tail departures do not undermine the reliability of the model's estimates, as OLS remains robust to moderate non-normality at this scale.

# Homoscedasticity (Constant Variance of Residuals)

# Get fitted values
fitted_vals = model.fittedvalues

# Scatter plot of residuals vs. fitted values
plt.figure(figsize=(8, 6))
sns.scatterplot(x=fitted_vals, y=residuals)
plt.axhline(0, color='red', linestyle='--')
plt.title('Residuals vs. Fitted Values')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.show()

# The residuals versus fitted values plot assesses whether the variance of the residuals stays constant across the range of predicted prices. The residuals are spread on both sides of the zero line, but the spread is not uniform: it widens through the middle of the fitted range and narrows at the upper end, producing a mild funnel shape rather than an even band. This indicates some heteroscedasticity, where the model's error variance is not perfectly constant across all price levels. There is also a faint diagonal banding visible, reflecting the discrete group structure of the categorical predictors. Overall the constant variance assumption is only partially satisfied, with some non-constant spread across the fitted range, though given the large sample of 55,685 observations the impact on the reliability of the model's estimates remains limited.

# Independence of errors

# Get the residuals
residuals = model.resid
residuals

# Perform Durbin-Watson test
dw_statistic = durbin_watson(residuals)
print(f'Durbin-Watson Statistic: {dw_statistic:.4f}')

# Interpretation
if 1.5 < dw_statistic < 2.5:
    print("Errors are likely independent (No autocorrelation).")
else:
    print("Potential autocorrelation detected.")

# Linearity

# Pairplot to visualise relationships
sns.pairplot(diamond_encoded, x_vars=X.columns.drop('const'),
             y_vars='price_boxcox', kind='reg', height=4, aspect=1)
plt.show()

# The pairplot assesses whether each continuous predictor shares an approximately linear relationship with the Box-Cox transformed price. Carat shows a strong, clear positive linear trend, confirming its appropriately linear association with price and its role as the dominant predictor in the model. Table displays a weak positive linear relationship, while depth shows an essentially flat, formless cloud with no discernible trend, indicating little to no linear association with price. This is consistent with the regression output, where depth was statistically insignificant. The remaining panels correspond to the one-hot encoded categorical predictors, which take only values of zero and one, so their fitted lines simply connect two group means rather than describing a continuous relationship and are linear by construction. Overall the linearity assumption is well supported for carat and adequately met for table, while depth contributes little linear signal, which explains why carat dominates the model's explanatory power.

# Training the linear regression model using sklearn

# Features and target variable
X = diamond_encoded.drop(columns=['price_boxcox'])
y = diamond_encoded['price_boxcox']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the shape of the training and testing sets
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# Initialise the Linear Regression model
linear_model = LinearRegression()
# Train the model on the training data
linear_model.fit(X_train, y_train)
# Predict on the test set
y_pred = linear_model.predict(X_test)

# Evaluate the model on Box-Cox scale
mse = round(mean_squared_error(y_test, y_pred), 2)
r2 = round(r2_score(y_test, y_pred), 2)
mae = round(mean_absolute_error(y_test, y_pred), 2)
rmse = round(mse ** 0.5, 2)

# Display evaluation metrics on Box-Cox scale
print("=== Metrics on Box-Cox Scale ===")
print(f"MSE: {mse}")
print(f"R²: {r2}")
print(f"MAE: {mae}")
print(f"RMSE: {rmse}")

# Inverse transform predictions back to original scale
y_pred_original = inv_boxcox(y_pred, lambda_price)
y_test_original = inv_boxcox(y_test, lambda_price)

# Evaluate on original scale (compute roots before rounding)
mse_original = mean_squared_error(y_test_original, y_pred_original)
rmse_original = round(mse_original ** 0.5, 2)
mae_original = round(mean_absolute_error(y_test_original, y_pred_original), 2)
mape = round((abs((y_test_original - y_pred_original) / y_test_original).mean()) * 100, 2)
r2_original = round(r2_score(y_test_original, y_pred_original), 2)
mse_original = round(mse_original, 2)

print("\n=== Metrics on Original Scale (USD) ===")
print(f"MSE: ${mse_original}")
print(f"MAE: ${mae_original}")
print(f"RMSE: ${rmse_original}")
print(f"MAPE: {mape:.2f}%")
print(f"R²: {r2_original}")

# Build a results table comparing actual vs predicted prices on the original scale
results = pd.DataFrame({
    'Actual Price (USD)': y_test_original.round(2),
    'Predicted Price (USD)': y_pred_original.round(2)
})
# Add the absolute and percentage error for each prediction
results['Error (USD)'] = (results['Predicted Price (USD)'] - results['Actual Price (USD)']).round(2)
results['Error (%)'] = ((results['Error (USD)'] / results['Actual Price (USD)']) * 100).round(2)

# Show the first 10 predictions
print("\n=== Sample Predictions (First 10) ===")
print(results.head(10).to_string())

# The linear regression model achieves a strong fit on the Box-Cox scale, with an R² of 0.80, but more limited predictive accuracy on the original dollar scale, with an R² of 0.47 and an average prediction error of around 30%. While this represents a usable level of performance and offers clear interpretability of how each attribute influences price, the substantial overshoots seen on individual stones reflect the inherent limitation of a linear model applied to a nonlinear pricing relationship, indicating that a more flexible modelling approach would be better suited to accurate diamond valuation at Aurora Gems.

# Training a Random Forest model

# Rechose columns to use for this question since Random Forest doesn't require Box-Cox transformation
diamond_prediction_rf = diamond_cleaned[[
    'carat',
    'cut',
    'color',
    'clarity',
    'depth',
    'table',
    'price']]

# Identify categorical columns
categorical_cols = [
    'cut',
    'color',
    'clarity']

# Perform one-hot encoding; keep all categories since Random Forest is unaffected by the dummy variable trap, dtype=int makes only the new dummy columns integers, leaving carat, depth and table as floats
diamond_encoded_rf = pd.get_dummies(diamond_prediction_rf, columns=categorical_cols, dtype=int)

# Display the first few rows of the encoded dataframe
diamond_encoded_rf.head()

# Features and target variable
X = diamond_encoded_rf.drop(columns=['price'])
y = diamond_encoded_rf['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialise the Random Forest Regressor model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
# Train the model on the training data (raw price target)
rf_model.fit(X_train, y_train)
# Predict on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model on the original scale (USD), computing the root before rounding
mse = mean_squared_error(y_test, y_pred)
rmse = round(mse ** 0.5, 2)
mae = round(mean_absolute_error(y_test, y_pred), 2)
mape = round((abs((y_test - y_pred) / y_test).mean()) * 100, 2)
r2 = round(r2_score(y_test, y_pred), 2)
mse = round(mse, 2)

# Display evaluation metrics
print("=== Metrics on Original Scale (USD) ===")
print(f"MSE: ${mse}")
print(f"MAE: ${mae}")
print(f"RMSE: ${rmse}")
print(f"MAPE: {mape:.2f}%")
print(f"R²: {r2}")

# Build a results table comparing actual vs predicted prices
results = pd.DataFrame({
    'Actual Price (USD)': y_test.reset_index(drop=True).round(2),
    'Predicted Price (USD)': pd.Series(y_pred).round(2)
})
# Add the absolute and percentage error for each prediction
results['Error (USD)'] = (results['Predicted Price (USD)'] - results['Actual Price (USD)']).round(2)
results['Error (%)'] = ((results['Error (USD)'] / results['Actual Price (USD)']) * 100).round(2)

# Show the first 10 predictions
print("\n=== Sample Predictions (First 10) ===")
print(results.head(10).to_string())

# The Random Forest model delivers strong and practically useful predictive performance, explaining 85% of the variation in diamond price with an average error of around 15%. By capturing the nonlinear relationships and interactions between a diamond's attributes without requiring any transformation, it substantially outperforms the linear regression in dollar-scale accuracy, making it the more reliable model for diamond valuation at Aurora Gems.
