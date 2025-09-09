"""
df.info()
df.describe()
df.isnull().sum()
sns.histplot(df[target])    # if regression
"""

"""
# Correlation Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()
"""


"""
# Histogram for a numeric column
sns.histplot(df["age"], kde=True, bins=30)
plt.title("Distribution of Age")
plt.show()
"""

"""
# Countplot for categorical column 
sns.countplot(x=df["city"])
plt.title("City Distribution")
plt.xticks(rotation=45)
plt.show()
"""

"""
# in-case problem is classification problem
sns.countplot(x=df["target_column"])
plt.title("Class Distribution")
plt.show()
"""

"""
# Boxplot for numeric column
sns.boxplot(x=df["salary"])
plt.title("Boxplot of Salary")
plt.show()
"""

"""
# Scatterplot of feature vs target
sns.scatterplot(x=df["age"], y=df["salary"], alpha=0.6)
plt.title("Age vs Salary")
plt.show()
"""

"""
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
"""


